from typing import *
import os
import copy
import functools
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict

from ...modules import sparse as sp
from ...utils.general_utils import dict_reduce
from ...utils.data_utils import recursive_to_device, cycle, BalancedResumableSampler
from .flow_matching import FlowMatchingTrainer
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.text_conditioned import TextConditionedMixin
from .mixins.image_conditioned import ImageConditionedMixin, MultiImageConditionedMixin
from .mixins.image_conditioned_proj import ImageConditionedProjMixin


class SparseFlowMatchingTrainer(FlowMatchingTrainer):
    """
    Trainer for sparse diffusion model with flow matching objective.
    
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
    
    def prepare_dataloader(self, **kwargs):
        """
        Prepare dataloader.
        """
        self.data_sampler = BalancedResumableSampler(
            self.dataset,
            shuffle=True,
            batch_size=self.batch_size_per_gpu,
        )
        if self.num_workers is None or self.num_workers == -1:
            num_workers = max(1, int(np.ceil((os.cpu_count() - 16) / torch.cuda.device_count())))
        else:
            num_workers = self.num_workers
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size_per_gpu,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            collate_fn=functools.partial(self.dataset.collate_fn, split_size=self.batch_split),
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

    def training_losses(
        self,
        x_0: sp.SparseTensor,
        cond=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x ... x C] sparse tensor of the inputs.
            cond: The [N x ...] tensor of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        noise = x_0.replace(torch.randn_like(x_0.feats))
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **kwargs)
        
        pred = self.training_models['denoiser'](x_t, t * 1000, cond, **kwargs)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)
        terms = edict()
        terms["mse"] = F.mse_loss(pred.feats, target.feats)
        terms["loss"] = terms["mse"]

        # log loss with time bins
        mse_per_instance = np.array([
            F.mse_loss(pred.feats[x_0.layout[i]], target.feats[x_0.layout[i]]).item()
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
            batch_size=num_samples,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
            generator=g,
        )
        data = next(iter(dataloader))

        # Collect metadata (dataset_name and sha256) for wandb display
        sample_metadata = []
        if '_dataset_name' in data and '_sha256' in data:
            for j in range(min(num_samples, len(data['_dataset_name']))):
                sample_metadata.append(f"{data['_dataset_name'][j]}/{data['_sha256'][j]}")
        # Remove metadata fields before inference
        data.pop('_dataset_name', None)
        data.pop('_sha256', None)

        # inference
        sampler = self.get_sampler()
        sample = []
        cond_vis = []
        for i in range(0, num_samples, batch_size):
            batch_data = {k: v[i:i+batch_size] for k, v in data.items()}
            batch_data = recursive_to_device(batch_data, 'cuda')
            noise = batch_data['x_0'].replace(torch.randn_like(batch_data['x_0'].feats))
            cond_vis.append(self.vis_cond(**batch_data))
            del batch_data['x_0']
            args = self.get_inference_cond(**batch_data)
            res = sampler.sample(
                self.models['denoiser'],
                noise=noise,
                **args,
                steps=12, guidance_strength=3.0, verbose=verbose,
            )
            sample.append(res.samples)
        sample = sp.sparse_cat(sample)
        
        sample_gt = {k: v for k, v in data.items()}
        sample = {k: v if k != 'x_0' else sample for k, v in data.items()}
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


class SparseFlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, SparseFlowMatchingTrainer):
    """
    Trainer for sparse diffusion model with flow matching objective and classifier-free guidance.
    
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


class TextConditionedSparseFlowMatchingCFGTrainer(TextConditionedMixin, SparseFlowMatchingCFGTrainer):
    """
    Trainer for sparse text-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
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


class ImageConditionedSparseFlowMatchingCFGTrainer(ImageConditionedMixin, SparseFlowMatchingCFGTrainer):
    """
    Trainer for sparse image-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
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


class MultiImageConditionedSparseFlowMatchingCFGTrainer(MultiImageConditionedMixin, SparseFlowMatchingCFGTrainer):
    """
    Trainer for sparse image-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
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


class ImageConditionedProjSparseFlowMatchingCFGTrainer(ImageConditionedProjMixin, SparseFlowMatchingCFGTrainer):
    """
    Trainer for sparse image-conditioned diffusion model with view-aligned projection.
    
    Uses ImageConditionedProjMixin for 3D-to-2D feature projection with camera parameters.
    CFG dropout is handled by ClassifierFreeGuidanceMixin (via p_uncond parameter).
    
    The projection grid outputs a full [B, R, R, R, D] tensor, and this trainer extracts
    features at sparse coordinates using advanced indexing.
    
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
        x_0: sp.SparseTensor,
        cond=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.
        
        Overridden to pass coords from x_0 to get_cond for sparse feature extraction.

        Args:
            x_0: The [N x ... x C] sparse tensor of the inputs.
            cond: The [N x ...] tensor of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        noise = x_0.replace(torch.randn_like(x_0.feats))
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        
        # Pass coords to get_cond for sparse feature extraction from full grid
        kwargs['coords'] = x_0.coords
        cond = self.get_cond(cond, **kwargs)
        
        # Pass concat_cond to denoiser if present (needed for PBR/texture training
        # where shape latent is concatenated with PBR latent as input)
        denoiser_kwargs = {}
        if 'concat_cond' in kwargs:
            denoiser_kwargs['concat_cond'] = kwargs['concat_cond']
        pred = self.training_models['denoiser'](x_t, t * 1000, cond, **denoiser_kwargs)
        
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)
        terms = edict()
        terms["mse"] = F.mse_loss(pred.feats, target.feats)
        terms["loss"] = terms["mse"]

        # log loss with time bins
        mse_per_instance = np.array([
            F.mse_loss(pred.feats[x_0.layout[i]], target.feats[x_0.layout[i]]).item()
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
        Run snapshot with coords passed to get_inference_cond for sparse feature extraction.
        
        For projection mode, we need to pass coords to properly extract features at
        sparse positions from the full projection grid.
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
            batch_size=num_samples,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
            generator=g,
        )
        data = next(iter(dataloader))

        # Collect metadata (dataset_name and sha256) for wandb display
        sample_metadata = []
        if '_dataset_name' in data and '_sha256' in data:
            for j in range(min(num_samples, len(data['_dataset_name']))):
                sample_metadata.append(f"{data['_dataset_name'][j]}/{data['_sha256'][j]}")
        # Remove metadata fields before inference
        data.pop('_dataset_name', None)
        data.pop('_sha256', None)

        # inference
        sampler = self.get_sampler()
        sample = []
        cond_vis = []
        for i in range(0, num_samples, batch_size):
            batch_data = {k: v[i:i+batch_size] for k, v in data.items()}
            batch_data = recursive_to_device(batch_data, 'cuda')
            noise = batch_data['x_0'].replace(torch.randn_like(batch_data['x_0'].feats))
            cond_vis.append(self.vis_cond(**batch_data))
            
            # Save coords before deleting x_0 (needed for projection feature extraction)
            coords = batch_data['x_0'].coords
            del batch_data['x_0']
            
            # Pass coords to get_inference_cond for sparse feature extraction
            batch_data['coords'] = coords
            args = self.get_inference_cond(**batch_data)
            
            res = sampler.sample(
                self.models['denoiser'],
                noise=noise,
                **args,
                steps=12, guidance_strength=3.0, verbose=verbose,
            )
            sample.append(res.samples)
        sample = sp.sparse_cat(sample)
        
        sample_gt = {k: v for k, v in data.items()}
        sample = {k: v if k != 'x_0' else sample for k, v in data.items()}
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
    
    @torch.no_grad()
    def visualize_sample(self, sample):
        """
        Convert a sample to images, including GT camera view if available.
        
        Args:
            sample: Either a SparseTensor or dict containing:
                - 'x_0': SparseTensor
                - 'camera_angle_x': [B] (optional)
                - 'camera_distance': [B] (optional)
                - 'mesh_scale': [B] (optional)
                
        Returns:
            dict with visualization images or tensor
        """
        if hasattr(self.dataset, 'visualize_sample'):
            if isinstance(sample, dict):
                # Extract camera params and pass them explicitly, since some
                # dataset.visualize_sample() (e.g. SLatShapeVisMixin) expect
                # separate keyword arguments rather than a single dict.
                camera_kwargs = {}
                for k in ('camera_angle_x', 'camera_distance', 'mesh_scale'):
                    if k in sample:
                        camera_kwargs[k] = sample[k]
                
                # Try passing camera kwargs explicitly first; fall back to
                # passing the entire dict if the dataset method doesn't accept them
                # (e.g. SLatPbrVisMixin expects a dict with 'x_0' + 'concat_cond').
                import inspect
                sig = inspect.signature(self.dataset.visualize_sample)
                params = list(sig.parameters.keys())
                if 'camera_angle_x' in params:
                    # Shape-style: visualize_sample(x_0, camera_angle_x=, ...)
                    x_0 = sample.get('x_0', sample)
                    return self.dataset.visualize_sample(x_0, **camera_kwargs)
                else:
                    # Tex/PBR-style: visualize_sample(sample_dict)
                    return self.dataset.visualize_sample(sample)
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
