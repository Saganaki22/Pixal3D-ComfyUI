from typing import *
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict

from ..basic import BasicTrainer
from ...pipelines import samplers 
from ...utils.general_utils import dict_reduce
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.text_conditioned import TextConditionedMixin
from .mixins.image_conditioned import ImageConditionedMixin
from .mixins.image_conditioned_proj import ImageConditionedProjMixin


class FlowMatchingTrainer(BasicTrainer):
    """
    Trainer for diffusion model with flow matching objective.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
    """
    def __init__(
        self,
        *args,
        t_schedule: dict = {
            'name': 'logitNormal',
            'args': {
                'mean': 0.0,
                'std': 1.0,
            }
        },
        sigma_min: float = 1e-5,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.t_schedule = t_schedule
        self.sigma_min = sigma_min

    def diffuse(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            t: The [N] tensor of diffusion steps [0-1].
            noise: If specified, use this noise instead of generating new noise.

        Returns:
            x_t, the noisy version of x_0 under timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        assert noise.shape == x_0.shape, "noise must have same shape as x_0"

        t = t.view(-1, *[1 for _ in range(len(x_0.shape) - 1)])
        x_t = (1 - t) * x_0 + (self.sigma_min + (1 - self.sigma_min) * t) * noise

        return x_t

    def reverse_diffuse(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Get original image from noisy version under timestep t.
        """
        assert noise.shape == x_t.shape, "noise must have same shape as x_t"
        t = t.view(-1, *[1 for _ in range(len(x_t.shape) - 1)])
        x_0 = (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * noise) / (1 - t)
        return x_0

    def get_v(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the velocity of the diffusion process at time t.
        """
        return (1 - self.sigma_min) * noise - x_0

    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        """
        return cond
    
    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        """
        return {'cond': cond, **kwargs}

    def get_sampler(self, **kwargs) -> samplers.FlowEulerSampler:
        """
        Get the sampler for the diffusion process.
        """
        return samplers.FlowEulerSampler(self.sigma_min)
    
    def vis_cond(self, **kwargs):
        """
        Visualize the conditioning data.
        """
        return {}

    def sample_t(self, batch_size: int) -> torch.Tensor:
        """
        Sample timesteps.
        """
        if self.t_schedule['name'] == 'uniform':
            t = torch.rand(batch_size)
        elif self.t_schedule['name'] == 'logitNormal':
            mean = self.t_schedule['args']['mean']
            std = self.t_schedule['args']['std']
            t = torch.sigmoid(torch.randn(batch_size) * std + mean)
        else:
            raise ValueError(f"Unknown t_schedule: {self.t_schedule['name']}")
        return t

    def training_losses(
        self,
        x_0: torch.Tensor,
        cond=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            cond: The [N x ...] tensor of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **kwargs)
        
        pred = self.training_models['denoiser'](x_t, t * 1000, cond, **kwargs)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)
        terms = edict()
        terms["mse"] = F.mse_loss(pred, target)
        terms["loss"] = terms["mse"]

        # log loss with time bins
        mse_per_instance = np.array([
            F.mse_loss(pred[i], target[i]).item()
            for i in range(x_0.shape[0])
        ])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}

        return terms, {}
    
    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        # Use current step as seed to ensure different samples for each snapshot
        import random
        snapshot_seed = self.step
        random.seed(snapshot_seed)
        np.random.seed(snapshot_seed)
        
        g = torch.Generator()
        g.manual_seed(snapshot_seed)
        
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
            generator=g,
        )

        # inference
        sampler = self.get_sampler()
        sample_gt = []
        sample = []
        cond_vis = []
        sample_metadata = []
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            data = {k: v[:batch].cuda() if isinstance(v, torch.Tensor) else v[:batch] for k, v in data.items()}
            
            # Collect metadata (dataset_name and sha256) for wandb display
            if '_dataset_name' in data and '_sha256' in data:
                for j in range(batch):
                    sample_metadata.append(f"{data['_dataset_name'][j]}/{data['_sha256'][j]}")
            
            # Remove metadata fields before inference
            data.pop('_dataset_name', None)
            data.pop('_sha256', None)
            
            noise = torch.randn_like(data['x_0'])
            sample_gt.append(data['x_0'])
            cond_vis.append(self.vis_cond(**data))
            del data['x_0']
            args = self.get_inference_cond(**data)
            res = sampler.sample(
                self.models['denoiser'],
                noise=noise,
                **args,
                steps=50, guidance_strength=3.0, verbose=verbose,
            )
            sample.append(res.samples)

        sample_gt = torch.cat(sample_gt, dim=0)
        sample = torch.cat(sample, dim=0)
        sample_dict = {
            'sample_gt': {'value': sample_gt, 'type': 'sample'},
            'sample': {'value': sample, 'type': 'sample'},
        }
        if sample_metadata:
            sample_dict['_metadata'] = sample_metadata
        sample_dict.update(dict_reduce(cond_vis, None, {
            'value': lambda x: torch.cat(x, dim=0),
            'type': lambda x: x[0],
        }))
        
        return sample_dict

    
class FlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, FlowMatchingTrainer):
    """
    Trainer for diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
    """
    pass


class TextConditionedFlowMatchingCFGTrainer(TextConditionedMixin, FlowMatchingCFGTrainer):
    """
    Trainer for text-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        text_cond_model(str): Text conditioning model.
    """
    pass


class ImageConditionedFlowMatchingCFGTrainer(ImageConditionedMixin, FlowMatchingCFGTrainer):
    """
    Trainer for image-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        image_cond_model (str): Image conditioning model.
    """
    pass


class ImageConditionedProjFlowMatchingCFGTrainer(ImageConditionedProjMixin, FlowMatchingCFGTrainer):
    """
    Trainer for image-conditioned diffusion model with view-aligned projection.
    
    Uses ImageConditionedProjMixin for 3D-to-2D feature projection with camera parameters.
    CFG dropout is handled by ClassifierFreeGuidanceMixin (via p_uncond parameter).
    
    Args:
        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        image_cond_model (dict): Image conditioning model config (DinoV3ProjFeatureExtractor).
        run_projection_test (bool): Whether to run projection visualization test before training.
    """
    
    def __init__(self, *args, run_projection_test: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_projection_test = run_projection_test
    
    def training_losses(
        self,
        x_0: torch.Tensor,
        cond=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.
        
        Overridden to avoid passing extra kwargs to the model.

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            cond: The [N x ...] tensor of additional conditions.
            kwargs: Additional arguments (camera info, view_idx, etc.) for conditioning.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **kwargs)
        
        # Note: SparseStructureFlowModel.forward() only accepts (x, t, cond)
        # Do not pass extra kwargs to avoid unexpected keyword argument errors
        pred = self.training_models['denoiser'](x_t, t * 1000, cond)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)
        terms = edict()
        terms["mse"] = F.mse_loss(pred, target)
        terms["loss"] = terms["mse"]

        # log loss with time bins
        mse_per_instance = np.array([
            F.mse_loss(pred[i], target[i]).item()
            for i in range(x_0.shape[0])
        ])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}

        return terms, {}
    
    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        """
        Run snapshot with camera parameters for GT view rendering.
        
        Overrides parent to include camera parameters in sample dicts for
        visualizing the GT camera view alongside the standard 4-view rendering.
        """
        # Use current step as seed to ensure different samples for each snapshot
        import random
        snapshot_seed = self.step
        random.seed(snapshot_seed)
        np.random.seed(snapshot_seed)
        
        g = torch.Generator()
        g.manual_seed(snapshot_seed)
        
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
            generator=g,
        )

        # inference
        sampler = self.get_sampler()
        sample_gt_list = []
        sample_list = []
        cond_vis = []
        sample_metadata = []
        
        # Camera params for GT view rendering
        camera_distances = []
        camera_angles = []
        mesh_scales = []
        
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            data = {k: v[:batch].cuda() if isinstance(v, torch.Tensor) else v[:batch] for k, v in data.items()}
            
            # Collect metadata (dataset_name and sha256) for wandb display
            if '_dataset_name' in data and '_sha256' in data:
                for j in range(batch):
                    sample_metadata.append(f"{data['_dataset_name'][j]}/{data['_sha256'][j]}")
            
            # Remove metadata fields before inference
            data.pop('_dataset_name', None)
            data.pop('_sha256', None)
            
            noise = torch.randn_like(data['x_0'])
            
            # Save GT sample
            sample_gt_list.append(data['x_0'])
            cond_vis.append(self.vis_cond(**data))
            
            # Save camera parameters for GT view rendering (if available)
            if 'camera_distance' in data:
                camera_distances.append(data['camera_distance'])
            if 'camera_angle_x' in data:
                camera_angles.append(data['camera_angle_x'])
            if 'mesh_scale' in data:
                mesh_scales.append(data['mesh_scale'])
            
            # Remove x_0 before inference
            del data['x_0']
            args = self.get_inference_cond(**data)
            res = sampler.sample(
                self.models['denoiser'],
                noise=noise,
                **args,
                steps=50, guidance_strength=3.0, verbose=verbose,
            )
            sample_list.append(res.samples)

        # Concatenate samples
        sample_gt = torch.cat(sample_gt_list, dim=0)
        sample = torch.cat(sample_list, dim=0)
        
        # Build sample dicts with camera info for GT view rendering
        sample_gt_value = {'x_0': sample_gt}
        sample_value = {'x_0': sample}
        
        # Add camera params if available
        if len(camera_distances) > 0:
            camera_distance = torch.cat(camera_distances, dim=0)
            sample_gt_value['camera_distance'] = camera_distance
            sample_value['camera_distance'] = camera_distance
        if len(camera_angles) > 0:
            camera_angle_x = torch.cat(camera_angles, dim=0)
            sample_gt_value['camera_angle_x'] = camera_angle_x
            sample_value['camera_angle_x'] = camera_angle_x
        if len(mesh_scales) > 0:
            mesh_scale = torch.cat(mesh_scales, dim=0)
            sample_gt_value['mesh_scale'] = mesh_scale
            sample_value['mesh_scale'] = mesh_scale
        
        sample_dict = {
            'sample_gt': {'value': sample_gt_value, 'type': 'sample'},
            'sample': {'value': sample_value, 'type': 'sample'},
        }
        if sample_metadata:
            sample_dict['_metadata'] = sample_metadata
        sample_dict.update(dict_reduce(cond_vis, None, {
            'value': lambda x: torch.cat(x, dim=0),
            'type': lambda x: x[0],
        }))
        
        return sample_dict
    
    @torch.no_grad()
    def visualize_sample(self, sample):
        """
        Convert a sample to images, including GT camera view if available.
        
        Args:
            sample: Either a tensor or dict containing:
                - 'x_0': latent tensor [B, C, D, H, W]
                - 'camera_angle_x': [B] (optional)
                - 'camera_distance': [B] (optional)
                - 'mesh_scale': [B] (optional)
                
        Returns:
            dict with visualization images or tensor
        """
        if hasattr(self.dataset, 'visualize_sample'):
            if isinstance(sample, dict):
                # Extract camera params if available
                camera_angle_x = sample.get('camera_angle_x')
                camera_distance = sample.get('camera_distance')
                mesh_scale = sample.get('mesh_scale')
                x_0 = sample.get('x_0', sample)
                
                return self.dataset.visualize_sample(
                    x_0,
                    camera_angle_x=camera_angle_x,
                    camera_distance=camera_distance,
                    mesh_scale=mesh_scale,
                )
            else:
                return self.dataset.visualize_sample(sample)
        else:
            if isinstance(sample, dict):
                return sample.get('x_0', sample)
            return sample
    
    def run(self):
        """
        Run training with projection visualization test before starting.
        """
        # Run projection visualization test before training starts (if enabled)
        if self.run_projection_test and self.is_master:
            print('\n' + '='*60)
            print('Running projection visualization test...')
            print('='*60)
            self._run_projection_visualization_test()
            
        super().run()
    
    @torch.no_grad()
    def _run_projection_visualization_test(self, num_samples: int = 4):
        """
        Run projection visualization test on a few samples before training starts.
        
        This helps verify that the 3D-to-2D projection is working correctly.
        """
        import os
        from torch.utils.data import DataLoader
        
        # Create a small dataloader
        dataloader = DataLoader(
            self.dataset,
            batch_size=min(num_samples, self.snapshot_batch_size),
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )
        
        # Get one batch
        data = next(iter(dataloader))
        data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        
        # Extract condition image
        cond = data.get('cond')
        if cond is None:
            print("Warning: No 'cond' field in data, skipping projection visualization test")
            return
        
        # Save directory
        save_dir = os.path.join(self.output_dir, 'samples', 'projection_test')
        
        # Call visualization method
        if hasattr(self, 'visualize_projection_test'):
            # Need to pass camera info as kwargs
            kwargs = {k: v for k, v in data.items() if k != 'cond' and k != 'x_0'}
            self.visualize_projection_test(
                cond=cond,
                save_dir=save_dir,
                prefix="proj_test",
                **kwargs
            )
            print(f"Projection visualization saved to: {save_dir}")
        else:
            print("Warning: visualize_projection_test not available")
