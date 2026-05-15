from abc import abstractmethod
import os
import time
import json
import copy
import threading
from functools import partial
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

from torchvision import utils

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .utils import *
from ..utils.general_utils import *
from ..utils.data_utils import recursive_to_device, cycle, ResumableSampler
from ..utils.dist_utils import *
from ..utils import grad_clip_utils, elastic_utils


class BasicTrainer:
    """
    Trainer for basic training loop.
    
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
        mix_precision_mode (str):
            - None: No mixed precision.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        mix_precision_dtype (str): Mixed precision dtype.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        parallel_mode (str): Parallel mode. Options are 'ddp'.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.
    """
    def __init__(self,
        models,
        dataset,
        *,
        output_dir,
        load_dir,
        step,
        max_steps,
        batch_size=None,
        batch_size_per_gpu=None,
        batch_split=None,
        optimizer={},
        lr_scheduler=None,
        elastic=None,
        grad_clip=None,
        ema_rate=0.9999,
        fp16_mode=None,
        mix_precision_mode='inflat_all',
        mix_precision_dtype='float16',
        fp16_scale_growth=1e-3,
        parallel_mode='ddp',
        finetune_ckpt=None,
        log_param_stats=False,
        prefetch_data=True,
        snapshot_batch_size=4,
        snapshot_num_samples=64,
        num_workers=None,
        debug=False,
        i_print=1000,
        i_log=500,
        i_sample=10000,
        i_save=10000,
        i_ddpcheck=10000,
        wandb_run=None,  # wandb run object
        **kwargs
    ):
        assert batch_size is not None or batch_size_per_gpu is not None, 'Either batch_size or batch_size_per_gpu must be specified.'

        self.models = models
        self.dataset = dataset
        self.batch_split = batch_split if batch_split is not None else 1
        self.max_steps = max_steps
        self.debug = debug
        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler
        self.elastic_controller_config = elastic
        self.grad_clip = grad_clip
        self.ema_rate = [ema_rate] if isinstance(ema_rate, float) else ema_rate
        if fp16_mode is not None:
            mix_precision_dtype = 'float16'
            mix_precision_mode = fp16_mode
        self.mix_precision_mode = mix_precision_mode
        self.mix_precision_dtype = str_to_dtype(mix_precision_dtype)
        self.fp16_scale_growth = fp16_scale_growth
        self.parallel_mode = parallel_mode
        self.log_param_stats = log_param_stats
        self.prefetch_data = prefetch_data
        self.snapshot_batch_size = snapshot_batch_size
        self.snapshot_num_samples = snapshot_num_samples
        self.num_workers = num_workers
        self.log = []
        if self.prefetch_data:
            self._data_prefetched = None

        self.output_dir = output_dir
        from datetime import datetime
        self._log_file = os.path.join(self.output_dir, f'log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        self.i_print = i_print
        self.i_log = i_log
        self.i_sample = i_sample
        self.i_save = i_save
        self.i_ddpcheck = i_ddpcheck        

        if dist.is_initialized():
            # Multi-GPU params
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.local_rank = dist.get_rank() % torch.cuda.device_count()
            self.is_master = self.rank == 0
        else:
            # Single-GPU params
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0
            self.is_master = True

        self.batch_size = batch_size if batch_size_per_gpu is None else batch_size_per_gpu * self.world_size
        self.batch_size_per_gpu = batch_size_per_gpu if batch_size_per_gpu is not None else batch_size // self.world_size
        assert self.batch_size % self.world_size == 0, 'Batch size must be divisible by the number of GPUs.'
        assert self.batch_size_per_gpu % self.batch_split == 0, 'Batch size per GPU must be divisible by batch split.'

        self.init_models_and_more(**kwargs)
        self.prepare_dataloader(**kwargs)
        
        # Load checkpoint
        self.step = 0
        if load_dir is not None and step is not None:
            self.load(load_dir, step)
        elif finetune_ckpt is not None:
            self.finetune_from(finetune_ckpt)
        
        if self.is_master:
            os.makedirs(os.path.join(self.output_dir, 'ckpts'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'samples'), exist_ok=True)
            self.writer = None  # TensorBoard disabled (S3 FUSE does not support append)
            # Initialize wandb
            self.wandb_run = wandb_run
            if self.wandb_run is not None:
                print(f'Wandb logging enabled: {self.wandb_run.url}')

        if self.parallel_mode == 'ddp' and self.world_size > 1:
            self.check_ddp()
            
        if self.is_master:
            print('\n\nTrainer initialized.')
            print(self)

    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f'  - Models:')
        for name, model in self.models.items():
            lines.append(f'    - {name}: {model.__class__.__name__}')
        lines.append(f'  - Dataset: {indent(str(self.dataset), 2)}')
        lines.append(f'  - Dataloader:')
        lines.append(f'    - Sampler: {self.dataloader.sampler.__class__.__name__}')
        lines.append(f'    - Num workers: {self.dataloader.num_workers}')
        lines.append(f'  - Number of steps: {self.max_steps}')
        lines.append(f'  - Number of GPUs: {self.world_size}')
        lines.append(f'  - Batch size: {self.batch_size}')
        lines.append(f'  - Batch size per GPU: {self.batch_size_per_gpu}')
        lines.append(f'  - Batch split: {self.batch_split}')
        lines.append(f'  - Optimizer: {self.optimizer.__class__.__name__}')
        lines.append(f'  - Learning rate: {self.optimizer.param_groups[0]["lr"]}')
        if self.lr_scheduler_config is not None:
            lines.append(f'  - LR scheduler: {self.lr_scheduler.__class__.__name__}')
        if self.elastic_controller_config is not None:
            lines.append(f'  - Elastic memory: {indent(str(self.elastic_controller), 2)}')
        if self.grad_clip is not None:
            lines.append(f'  - Gradient clip: {indent(str(self.grad_clip), 2)}')
        lines.append(f'  - EMA rate: {self.ema_rate}')
        lines.append(f'  - Mixed precision dtype: {self.mix_precision_dtype}')
        lines.append(f'  - Mixed precision mode: {self.mix_precision_mode}')
        if self.mix_precision_mode == 'amp' and self.mix_precision_dtype == torch.float16:
            lines.append(f'  - FP16 scale growth: {self.fp16_scale_growth}')
        lines.append(f'  - Parallel mode: {self.parallel_mode}')
        return '\n'.join(lines)

    @property
    def device(self):
        for _, model in self.models.items():
            if hasattr(model, 'device'):
                return model.device
        return next(list(self.models.values())[0].parameters()).device
            
    def init_models_and_more(self, **kwargs):
        """
        Initialize models and more.
        """
        if self.world_size > 1:
            # Prepare distributed data parallel
            self.training_models = {
                name: DDP(
                    model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    bucket_cap_mb=128,
                    find_unused_parameters=False
                )
                for name, model in self.models.items()
            }
        else:
            self.training_models = self.models

        # Build master params
        self.model_params = sum(
            [[p for p in model.parameters() if p.requires_grad] for model in self.models.values()]
        , [])
        if self.mix_precision_mode == 'amp':
            self.master_params = self.model_params
            if self.mix_precision_dtype == torch.float16:
                self.scaler = torch.GradScaler()
        elif self.mix_precision_mode == 'inflat_all':
            self.master_params = make_master_params(self.model_params)
            if self.mix_precision_dtype == torch.float16:
                self.log_scale = 20.0
        elif self.mix_precision_mode is None:
            self.master_params = self.model_params
        else:
            raise NotImplementedError(f'Mix precision mode {self.mix_precision_mode} is not implemented.')

        # Build EMA params
        if self.is_master:
            self.ema_params = [copy.deepcopy(self.master_params) for _ in self.ema_rate]

        # Initialize optimizer
        if hasattr(torch.optim, self.optimizer_config['name']):
            self.optimizer = getattr(torch.optim, self.optimizer_config['name'])(self.master_params, **self.optimizer_config['args'])
        else:
            self.optimizer = globals()[self.optimizer_config['name']](self.master_params, **self.optimizer_config['args'])
        
        # Initalize learning rate scheduler
        if self.lr_scheduler_config is not None:
            if hasattr(torch.optim.lr_scheduler, self.lr_scheduler_config['name']):
                self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_config['name'])(self.optimizer, **self.lr_scheduler_config['args'])
            else:
                self.lr_scheduler = globals()[self.lr_scheduler_config['name']](self.optimizer, **self.lr_scheduler_config['args'])

        # Initialize elastic memory controller
        if self.elastic_controller_config is not None:
            assert any([isinstance(model, (elastic_utils.ElasticModule, elastic_utils.ElasticModuleMixin)) for model in self.models.values()]), \
                'No elastic module found in models, please inherit from ElasticModule or ElasticModuleMixin'
            self.elastic_controller = getattr(elastic_utils, self.elastic_controller_config['name'])(**self.elastic_controller_config['args'])
            for model in self.models.values():
                if isinstance(model, (elastic_utils.ElasticModule, elastic_utils.ElasticModuleMixin)):
                    model.register_memory_controller(self.elastic_controller)

        # Initialize gradient clipper
        if self.grad_clip is not None:
            if isinstance(self.grad_clip, (float, int)):
                self.grad_clip = float(self.grad_clip)
            else:
                self.grad_clip = getattr(grad_clip_utils, self.grad_clip['name'])(**self.grad_clip['args'])

    def prepare_dataloader(self, **kwargs):
        """
        Prepare dataloader.
        """
        self.data_sampler = ResumableSampler(
            self.dataset,
            shuffle=True,
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
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
            sampler=self.data_sampler,
        )
        self.data_iterator = cycle(self.dataloader)

    def _master_params_to_state_dicts(self, master_params):
        """
        Convert master params to dict of state_dicts.
        """
        if self.mix_precision_mode == 'inflat_all':
            master_params = unflatten_master_params(self.model_params, master_params)
        state_dicts = {name: model.state_dict() for name, model in self.models.items()}
        master_params_names = sum(
            [[(name, n) for n, p in model.named_parameters() if p.requires_grad] for name, model in self.models.items()]
        , [])
        for i, (model_name, param_name) in enumerate(master_params_names):
            state_dicts[model_name][param_name] = master_params[i]
        return state_dicts

    def _state_dicts_to_master_params(self, master_params, state_dicts):
        """
        Convert a state_dict to master params.
        """
        master_params_names = sum(
            [[(name, n) for n, p in model.named_parameters() if p.requires_grad] for name, model in self.models.items()]
        , [])
        params = [state_dicts[name][param_name] for name, param_name in master_params_names]
        if self.mix_precision_mode == 'inflat_all':
            model_params_to_master_params(params, master_params)
        else:
            for i, param in enumerate(params):
                master_params[i].data.copy_(param.data)

    def load(self, load_dir, step=0):
        """
        Load a checkpoint.
        Should be called by all processes.
        """
        if self.is_master:
            print(f'\nLoading checkpoint from step {step}...', end='')
            
        model_ckpts = {}
        for name, model in self.models.items():
            model_ckpt = torch.load(read_file_dist(os.path.join(load_dir, 'ckpts', f'{name}_step{step:07d}.pt')), map_location=self.device, weights_only=True)
            model_ckpts[name] = model_ckpt
            model.load_state_dict(model_ckpt)
        self._state_dicts_to_master_params(self.master_params, model_ckpts)
        del model_ckpts

        if self.is_master:
            for i, ema_rate in enumerate(self.ema_rate):
                ema_ckpts = {}
                for name, model in self.models.items():
                    ema_ckpt = torch.load(os.path.join(load_dir, 'ckpts', f'{name}_ema{ema_rate}_step{step:07d}.pt'), map_location=self.device, weights_only=True)
                    ema_ckpts[name] = ema_ckpt
                self._state_dicts_to_master_params(self.ema_params[i], ema_ckpts)
                del ema_ckpts
        
        misc_ckpt = torch.load(read_file_dist(os.path.join(load_dir, 'ckpts', f'misc_step{step:07d}.pt')), map_location=torch.device('cpu'), weights_only=False)
        self.optimizer.load_state_dict(misc_ckpt['optimizer'])
        self.step = misc_ckpt['step']
        self.data_sampler.load_state_dict(misc_ckpt['data_sampler'])
        if self.mix_precision_mode == 'amp' and self.mix_precision_dtype == torch.float16:
            self.scaler.load_state_dict(misc_ckpt['scaler'])
        elif self.mix_precision_mode == 'inflat_all' and self.mix_precision_dtype == torch.float16:
            self.log_scale = misc_ckpt['log_scale']
        if self.lr_scheduler_config is not None:
            self.lr_scheduler.load_state_dict(misc_ckpt['lr_scheduler'])
        if self.elastic_controller_config is not None:
            self.elastic_controller.load_state_dict(misc_ckpt['elastic_controller'])
        if self.grad_clip is not None and not isinstance(self.grad_clip, float):
            self.grad_clip.load_state_dict(misc_ckpt['grad_clip'])
        del misc_ckpt

        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print(' Done.')

        if self.world_size > 1:
            self.check_ddp()

    def save(self, non_blocking=True):
        """
        Save a checkpoint.
        Should be called only by the rank 0 process.
        """
        assert self.is_master, 'save() should be called only by the rank 0 process.'
        print(f'\nSaving checkpoint at step {self.step}...', end='')
        
        model_ckpts = self._master_params_to_state_dicts(self.master_params)
        for name, model_ckpt in model_ckpts.items():
            model_ckpt = {k: v.cpu() for k, v in model_ckpt.items()}  # Move to CPU for saving
            if non_blocking:
                threading.Thread(
                    target=torch.save,
                    args=(model_ckpt, os.path.join(self.output_dir, 'ckpts', f'{name}_step{self.step:07d}.pt')),
                ).start()
            else:
                torch.save(model_ckpt, os.path.join(self.output_dir, 'ckpts', f'{name}_step{self.step:07d}.pt'))
        
        for i, ema_rate in enumerate(self.ema_rate):
            ema_ckpts = self._master_params_to_state_dicts(self.ema_params[i])
            for name, ema_ckpt in ema_ckpts.items():
                ema_ckpt = {k: v.cpu() for k, v in ema_ckpt.items()}  # Move to CPU for saving
                if non_blocking:
                    threading.Thread(
                        target=torch.save,
                        args=(ema_ckpt, os.path.join(self.output_dir, 'ckpts', f'{name}_ema{ema_rate}_step{self.step:07d}.pt')),
                    ).start()
                else:
                    torch.save(ema_ckpt, os.path.join(self.output_dir, 'ckpts', f'{name}_ema{ema_rate}_step{self.step:07d}.pt'))

        misc_ckpt = {
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'data_sampler': self.data_sampler.state_dict(),
        }
        if self.mix_precision_mode == 'amp' and self.mix_precision_dtype == torch.float16:
            misc_ckpt['scaler'] = self.scaler.state_dict()
        elif self.mix_precision_mode == 'inflat_all' and self.mix_precision_dtype == torch.float16:
            misc_ckpt['log_scale'] = self.log_scale
        if self.lr_scheduler_config is not None:
            misc_ckpt['lr_scheduler'] = self.lr_scheduler.state_dict()
        if self.elastic_controller_config is not None:
            misc_ckpt['elastic_controller'] = self.elastic_controller.state_dict()
        if self.grad_clip is not None and not isinstance(self.grad_clip, float):
            misc_ckpt['grad_clip'] = self.grad_clip.state_dict()
        if non_blocking:
            threading.Thread(
                target=torch.save,
                args=(misc_ckpt, os.path.join(self.output_dir, 'ckpts', f'misc_step{self.step:07d}.pt')),
            ).start()
        else:
            torch.save(misc_ckpt, os.path.join(self.output_dir, 'ckpts', f'misc_step{self.step:07d}.pt'))
        print(' Done.')

    def _remap_checkpoint_keys(self, model_ckpt, model_state_dict):
        """
        Remap checkpoint keys to match model state dict.
        
        Handles structural changes like:
        - cross_attn.xxx -> cross_attn.cross_attn_block.xxx (for ProjectAttention wrapper)
        
        Args:
            model_ckpt: Checkpoint state dict
            model_state_dict: Model state dict
            
        Returns:
            Remapped checkpoint dict
        """
        remapped_ckpt = {}
        remapped_count = 0
        
        for ckpt_key, ckpt_value in model_ckpt.items():
            # Check if key exists directly
            if ckpt_key in model_state_dict:
                remapped_ckpt[ckpt_key] = ckpt_value
                continue
            
            # Try remapping: cross_attn.xxx -> cross_attn.cross_attn_block.xxx
            # This handles the case when cross_attn is wrapped by ProjectAttention
            if '.cross_attn.' in ckpt_key:
                # Split at .cross_attn.
                parts = ckpt_key.split('.cross_attn.')
                if len(parts) == 2:
                    new_key = f'{parts[0]}.cross_attn.cross_attn_block.{parts[1]}'
                    if new_key in model_state_dict:
                        remapped_ckpt[new_key] = ckpt_value
                        remapped_count += 1
                        continue
            
            # Key not remapped, keep original (will be handled by missing key logic)
            remapped_ckpt[ckpt_key] = ckpt_value
        
        if remapped_count > 0 and self.is_master:
            print(f'Info: Remapped {remapped_count} cross_attn keys to cross_attn.cross_attn_block')
        
        return remapped_ckpt

    def finetune_from(self, finetune_ckpt):
        """
        Finetune from a checkpoint.
        Should be called by all processes.
        """
        # Allow missing keys (e.g., register_buffer parameters)
        ALLOWED_MISSING_KEYS = {'rope_phases'}
        
        if self.is_master:
            print('\nFinetuning from:')
            for name, path in finetune_ckpt.items():
                print(f'  - {name}: {path}')
        
        model_ckpts = {}
        for name, model in self.models.items():
            model_state_dict = model.state_dict()
            if name in finetune_ckpt:
                model_ckpt = torch.load(read_file_dist(finetune_ckpt[name]), map_location=self.device, weights_only=True)
                
                # Remap checkpoint keys to handle structural changes (e.g., ProjectAttention wrapper)
                model_ckpt = self._remap_checkpoint_keys(model_ckpt, model_state_dict)
                
                # Check extra keys (in ckpt but not in model)
                for k, v in model_ckpt.items():
                    if k not in model_state_dict:
                        if self.is_master:
                            print(f'Warning: {k} not found in model_state_dict, skipped.')
                        model_ckpt[k] = None
                    elif model_ckpt[k].shape != model_state_dict[k].shape:
                        if self.is_master:
                            print(f'Warning: {k} shape mismatch, {model_ckpt[k].shape} vs {model_state_dict[k].shape}, skipped.')
                        model_ckpt[k] = model_state_dict[k]
                model_ckpt = {k: v for k, v in model_ckpt.items() if v is not None}
                
                # Check missing keys (in model but not in ckpt)
                missing_keys = set(model_state_dict.keys()) - set(model_ckpt.keys())
                unexpected_missing = missing_keys - ALLOWED_MISSING_KEYS
                if unexpected_missing and self.is_master:
                    print(f'Error: Missing keys in checkpoint: {unexpected_missing}')
                    raise RuntimeError(f'Missing keys in checkpoint: {unexpected_missing}')
                if missing_keys & ALLOWED_MISSING_KEYS and self.is_master:
                    print(f'Info: Using model initialized values for: {missing_keys & ALLOWED_MISSING_KEYS}')
                
                # Fill in missing keys (using model initialized values)
                for k in missing_keys:
                    model_ckpt[k] = model_state_dict[k]
                
                model_ckpts[name] = model_ckpt
                model.load_state_dict(model_ckpt)
            else:
                if self.is_master:
                    print(f'Warning: {name} not found in finetune_ckpt, skipped.')
                model_ckpts[name] = model_state_dict
        self._state_dicts_to_master_params(self.master_params, model_ckpts)
        if self.is_master:
            for i, ema_rate in enumerate(self.ema_rate):
                self._state_dicts_to_master_params(self.ema_params[i], model_ckpts)
        del model_ckpts

        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            print('Done.')

        if self.world_size > 1:
            self.check_ddp()

    @abstractmethod
    def run_snapshot(self, num_samples, batch_size=4, verbose=False, **kwargs):
        """
        Run a snapshot of the model.
        """
        pass

    @torch.no_grad()
    def visualize_sample(self, sample):
        """
        Convert a sample to an image.
        """
        if hasattr(self.dataset, 'visualize_sample'):
            return self.dataset.visualize_sample(sample)
        else:
            return sample

    @torch.no_grad()
    def snapshot_dataset(self, num_samples=100, batch_size=4):
        """
        Sample images from the dataset.
        """
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=True,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )
        save_cfg = {}
        for i in range(0, num_samples, batch_size):
            data = next(iter(dataloader))
            data = {k: v[:min(num_samples - i, batch_size)] for k, v in data.items()}
            data = recursive_to_device(data, self.device)
            try:
                vis = self.visualize_sample(data)
            except (RuntimeError, Exception) as e:
                print(f'\033[93m[WARN] snapshot_dataset visualize_sample failed (batch {i}), skipping: {e}\033[0m')
                torch.cuda.empty_cache()
                continue
            if isinstance(vis, dict):
                for k, v in vis.items():
                    if f'dataset_{k}' not in save_cfg:
                        save_cfg[f'dataset_{k}'] = []
                    save_cfg[f'dataset_{k}'].append(v)
            else:
                if 'dataset' not in save_cfg:
                    save_cfg['dataset'] = []
                save_cfg['dataset'].append(vis)
        for name, image in save_cfg.items():
            utils.save_image(
                torch.cat(image, dim=0),
                os.path.join(self.output_dir, 'samples', f'{name}.jpg'),
                nrow=int(np.sqrt(num_samples)),
                normalize=True,
                value_range=self.dataset.value_range,
            )

    @torch.no_grad()
    def snapshot(self, suffix=None, num_samples=64, batch_size=4, verbose=False):
        """
        Sample images from the model.
        NOTE: When num_samples >= 4, this function should be called by all processes.
              When num_samples < 4, only master runs snapshot (other ranks skip via barrier).
        """
        # Free cached GPU memory before snapshot to avoid OOM / illegal address errors
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        if self.is_master:
            print(f'\nSampling {num_samples} images...', end='')

        if suffix is None:
            suffix = f'step{self.step:07d}'

        # When num_samples < 4, only master runs snapshot to avoid multi-rank gather issues
        master_only = num_samples < 4
        
        sample_metadata = None  # Will hold list of "dataset_name/sha256" strings

        if master_only and self.world_size > 1:
            if not self.is_master:
                # Non-master ranks just wait at barrier
                dist.barrier()
                return

            # Master runs snapshot alone
            amp_context = partial(torch.autocast, device_type='cuda', dtype=self.mix_precision_dtype) if self.mix_precision_mode == 'amp' else nullcontext
            with amp_context():
                samples = self.run_snapshot(num_samples, batch_size=batch_size, verbose=verbose)

            # Extract metadata before preprocessing
            sample_metadata = samples.pop('_metadata', None)

            # Free GPU memory after sampling, before decode + render
            torch.cuda.empty_cache()

            # Preprocess images
            for key in list(samples.keys()):
                if samples[key]['type'] == 'sample':
                    try:
                        vis = self.visualize_sample(samples[key]['value'])
                    except RuntimeError as e:
                        print(f"[Snapshot] WARNING: visualize_sample failed for '{key}': {e}")
                        # Reset CUDA error state and skip this sample
                        try:
                            torch.cuda.synchronize()
                        except RuntimeError:
                            pass
                        torch.cuda.empty_cache()
                        del samples[key]
                        continue
                    if isinstance(vis, dict):
                        for k, v in vis.items():
                            samples[f'{key}_{k}'] = {'value': v, 'type': 'image'}
                        del samples[key]
                    else:
                        samples[key] = {'value': vis, 'type': 'image'}

            # No gather needed, master already has all samples
            dist.barrier()
        else:
            # Distribute sampling across all ranks
            num_samples_per_process = int(np.ceil(num_samples / self.world_size))
            amp_context = partial(torch.autocast, device_type='cuda', dtype=self.mix_precision_dtype) if self.mix_precision_mode == 'amp' else nullcontext
            
            with amp_context():
                samples = self.run_snapshot(num_samples_per_process, batch_size=batch_size, verbose=verbose)

            # Extract metadata before preprocessing
            local_metadata = samples.pop('_metadata', None)

            # Free GPU memory after sampling, before decode + render
            torch.cuda.empty_cache()

            # Preprocess images
            for key in list(samples.keys()):
                if samples[key]['type'] == 'sample':
                    try:
                        vis = self.visualize_sample(samples[key]['value'])
                    except RuntimeError as e:
                        print(f"[Snapshot] WARNING: visualize_sample failed for '{key}': {e}")
                        torch.cuda.synchronize()
                        del samples[key]
                        continue
                    if isinstance(vis, dict):
                        for k, v in vis.items():
                            samples[f'{key}_{k}'] = {'value': v, 'type': 'image'}
                        del samples[key]
                    else:
                        samples[key] = {'value': vis, 'type': 'image'}

            # Gather results
            if self.world_size > 1:
                for key in samples.keys():
                    samples[key]['value'] = samples[key]['value'].contiguous()
                    if self.is_master:
                        all_images = [torch.empty_like(samples[key]['value']) for _ in range(self.world_size)]
                    else:
                        all_images = []
                    dist.gather(samples[key]['value'], all_images, dst=0)
                    if self.is_master:
                        samples[key]['value'] = torch.cat(all_images, dim=0)[:num_samples]
                
                # Gather metadata across ranks
                if local_metadata is not None:
                    all_metadata = [None] * self.world_size
                    dist.all_gather_object(all_metadata, local_metadata)
                    if self.is_master:
                        sample_metadata = sum(all_metadata, [])[:num_samples]
                else:
                    sample_metadata = None
            else:
                sample_metadata = local_metadata

        # Save images
        if self.is_master:
            os.makedirs(os.path.join(self.output_dir, 'samples', suffix), exist_ok=True)
            wandb_images = {}  # Collect images for wandb logging
            nrow = int(np.sqrt(num_samples))
            vr = self.dataset.value_range
            
            # Build metadata caption string for wandb
            metadata_caption = ''
            if sample_metadata:
                metadata_caption = '\n' + ' | '.join(sample_metadata)
                # Also save metadata to file
                with open(os.path.join(self.output_dir, 'samples', suffix, 'metadata.txt'), 'w') as f:
                    for i, m in enumerate(sample_metadata):
                        f.write(f'{i}: {m}\n')

            # Helper: make a normalized grid tensor from a batch of images
            def _make_grid(tensor):
                return utils.make_grid(tensor, nrow=nrow, normalize=True, value_range=vr)

            # Helper: resize grid to target height (keep aspect ratio)
            def _resize_to_height(grid, target_h):
                import torch.nn.functional as F
                _, h, w = grid.shape
                if h == target_h:
                    return grid
                target_w = int(round(w * target_h / h))
                return F.interpolate(grid.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(0)

            # --- Save individual images (original behavior) ---
            for key in samples.keys():
                if samples[key]['type'] == 'image':
                    image_path = os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}.jpg')
                    utils.save_image(
                        samples[key]['value'],
                        image_path,
                        nrow=nrow,
                        normalize=True,
                        value_range=vr,
                    )
                    # Collect for wandb
                    if self.wandb_run is not None:
                        grid = _make_grid(samples[key]['value'])
                        grid_np = grid.permute(1, 2, 0).cpu().numpy()
                        grid_np = (grid_np * 255).clip(0, 255).astype(np.uint8)
                        wandb_images[f'samples/{key}'] = wandb.Image(grid_np, caption=f'{key} at step {self.step}{metadata_caption}')
                elif samples[key]['type'] == 'number':
                    val_min = samples[key]['value'].min()
                    val_max = samples[key]['value'].max()
                    images = (samples[key]['value'] - val_min) / (val_max - val_min)
                    images = utils.make_grid(
                        images,
                        nrow=nrow,
                        normalize=False,
                    )
                    save_image_with_notes(
                        images,
                        os.path.join(self.output_dir, 'samples', suffix, f'{key}_{suffix}.jpg'),
                        notes=f'{key} min: {val_min}, max: {val_max}',
                    )

            # --- Save combined images ---
            sample_keys = set(samples.keys())

            # Combined 1: image + sample_gt_view + sample_gt_gt_view (shape)
            #             image + sample_gt_view_{attr} + sample_gt_gt_view_{attr} (tex, per attribute)
            # Detect gt_view attribute suffixes from sample keys
            gt_view_attrs = set()
            for k in sample_keys:
                if k.startswith('sample_gt_view_'):
                    attr = k[len('sample_gt_view_'):]
                    gt_view_attrs.add(attr)
            
            if gt_view_attrs:
                # Tex mode: generate combined view for each PBR attribute
                for attr in sorted(gt_view_attrs):
                    combo1_keys = ['image', f'sample_gt_view_{attr}', f'sample_gt_gt_view_{attr}']
                    combo1_present = [k for k in combo1_keys if k in sample_keys and samples[k]['type'] == 'image']
                    if len(combo1_present) >= 2:
                        grids = [_make_grid(samples[k]['value']) for k in combo1_present]
                        target_h = max(g.shape[1] for g in grids)
                        grids = [_resize_to_height(g, target_h) for g in grids]
                        combined = torch.cat(grids, dim=2)
                        combined_path = os.path.join(self.output_dir, 'samples', suffix, f'combined_views_{attr}_{suffix}.jpg')
                        utils.save_image(combined, combined_path, normalize=False)
                        if self.wandb_run is not None:
                            grid_np = combined.permute(1, 2, 0).cpu().numpy()
                            grid_np = (grid_np * 255).clip(0, 255).astype(np.uint8)
                            label = ' | '.join(combo1_present)
                            wandb_images[f'samples/combined_views_{attr}'] = wandb.Image(grid_np, caption=f'{label} at step {self.step}{metadata_caption}')
            else:
                # Shape mode: single gt_view
                combo1_keys = ['image', 'sample_gt_view', 'sample_gt_gt_view']
                combo1_present = [k for k in combo1_keys if k in sample_keys and samples[k]['type'] == 'image']
                if len(combo1_present) >= 2:
                    grids = [_make_grid(samples[k]['value']) for k in combo1_present]
                    target_h = max(g.shape[1] for g in grids)
                    grids = [_resize_to_height(g, target_h) for g in grids]
                    combined = torch.cat(grids, dim=2)
                    combined_path = os.path.join(self.output_dir, 'samples', suffix, f'combined_views_{suffix}.jpg')
                    utils.save_image(combined, combined_path, normalize=False)
                    if self.wandb_run is not None:
                        grid_np = combined.permute(1, 2, 0).cpu().numpy()
                        grid_np = (grid_np * 255).clip(0, 255).astype(np.uint8)
                        label = ' | '.join(combo1_present)
                        wandb_images[f'samples/combined_views'] = wandb.Image(grid_np, caption=f'{label} at step {self.step}{metadata_caption}')

            # Combined 2: sample_multiview + sample_gt_multiview
            combo2_keys = ['sample_multiview', 'sample_gt_multiview']
            combo2_present = [k for k in combo2_keys if k in sample_keys and samples[k]['type'] == 'image']
            if len(combo2_present) >= 2:
                grids = [_make_grid(samples[k]['value']) for k in combo2_present]
                target_h = max(g.shape[1] for g in grids)
                grids = [_resize_to_height(g, target_h) for g in grids]
                combined = torch.cat(grids, dim=2)  # concatenate along width
                combined_path = os.path.join(self.output_dir, 'samples', suffix, f'combined_multiview_{suffix}.jpg')
                utils.save_image(combined, combined_path, normalize=False)
                if self.wandb_run is not None:
                    grid_np = combined.permute(1, 2, 0).cpu().numpy()
                    grid_np = (grid_np * 255).clip(0, 255).astype(np.uint8)
                    label = ' | '.join(combo2_present)
                    wandb_images[f'samples/combined_multiview'] = wandb.Image(grid_np, caption=f'{label} at step {self.step}{metadata_caption}')

            # Log images to wandb
            if self.wandb_run is not None and wandb_images:
                self.wandb_run.log(wandb_images, step=self.step)

        if self.is_master:
            print(' Done.')

    def update_ema(self):
        """
        Update exponential moving average.
        Should only be called by the rank 0 process.
        """
        assert self.is_master, 'update_ema() should be called only by the rank 0 process.'
        for i, ema_rate in enumerate(self.ema_rate):
            for master_param, ema_param in zip(self.master_params, self.ema_params[i]):
                ema_param.detach().mul_(ema_rate).add_(master_param, alpha=1.0 - ema_rate)

    def check_ddp(self):
        """
        Check if DDP is working properly.
        Should be called by all process.
        """
        if self.is_master:
            print('\nPerforming DDP check...')

        if self.is_master:
            print('Checking if parameters are consistent across processes...')
        dist.barrier()
        try:
            for p in self.master_params:
                # split to avoid OOM
                for i in range(0, p.numel(), 10000000):
                    sub_size = min(10000000, p.numel() - i)
                    sub_p = p.detach().view(-1)[i:i+sub_size]
                    # gather from all processes
                    sub_p_gather = [torch.empty_like(sub_p) for _ in range(self.world_size)]
                    dist.all_gather(sub_p_gather, sub_p)
                    # check if equal
                    assert all([torch.equal(sub_p, sub_p_gather[i]) for i in range(self.world_size)]), 'parameters are not consistent across processes'
        except AssertionError as e:
            if self.is_master:
                print(f'\n\033[91mError: {e}\033[0m')
                print('DDP check failed.')
            raise e

        dist.barrier()
        if self.is_master:
            print('Done.')

    def _verify_gradient_sync(self):
        """
        Verify that DDP gradient synchronization is working correctly.
        DDP's backward automatically performs all_reduce on gradients; after sync all ranks should have identical gradients.
        
        Verification method:
        1. Compute total gradient norm across all parameters
        2. Gather gradient norms from all ranks
        3. If DDP sync is working, all ranks should have identical gradient norms
        4. If not synced, gradient norms will differ (since each rank processes different data)
        """
        # Compute total gradient norm on this rank
        total_grad_norm_sq = 0.0
        grad_count = 0
        for p in self.model_params:
            if p.grad is not None:
                total_grad_norm_sq += p.grad.detach().float().norm().item() ** 2
                grad_count += 1
        
        if grad_count == 0:
            return
        
        local_grad_norm = total_grad_norm_sq ** 0.5
        
        # Ensure all processes reach the same point
        dist.barrier()
        
        # Gather gradient norms from all ranks
        grad_norm_tensor = torch.tensor([local_grad_norm], dtype=torch.float64, device=self.device)
        all_grad_norms = [torch.zeros(1, dtype=torch.float64, device=self.device) for _ in range(self.world_size)]
        dist.all_gather(all_grad_norms, grad_norm_tensor)
        all_grad_norms = [g.item() for g in all_grad_norms]
        
        # Verify all ranks have the same gradient norm (relative error tolerance: 0.1%)
        ref_norm = all_grad_norms[0]
        if ref_norm > 0:
            is_synced = all(abs(g - ref_norm) / ref_norm < 1e-3 for g in all_grad_norms)
        else:
            is_synced = all(abs(g) < 1e-10 for g in all_grad_norms)
        
        if self.is_master:
            print(f'\n{"="*60}')
            print(f'[Step {self.step}] DDP Gradient Sync Verification:')
            for i, g in enumerate(all_grad_norms):
                print(f'  Rank {i} grad_norm: {g:.8f}')
            if is_synced:
                print(f'  \033[92m✓ PASS: All gradients are synchronized!\033[0m')
            else:
                max_diff = max(abs(g - ref_norm) for g in all_grad_norms)
                print(f'  \033[91m✗ FAIL: Gradients are NOT synchronized! Max diff: {max_diff:.8f}\033[0m')
            print(f'{"="*60}\n')

    @abstractmethod
    def training_losses(**mb_data):
        """
        Compute training losses.
        """
        pass

    def load_data(self):
        """
        Load data.
        """
        if self.prefetch_data:
            if self._data_prefetched is None:
                self._data_prefetched = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
            data = self._data_prefetched
            self._data_prefetched = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
        else:
            data = recursive_to_device(next(self.data_iterator), self.device, non_blocking=True)
        
        # if the data is a dict, we need to split it into multiple dicts with batch_size_per_gpu
        if isinstance(data, dict):
            if self.batch_split == 1:
                data_list = [data]
            else:
                batch_size = list(data.values())[0].shape[0]
                data_list = [
                    {k: v[i * batch_size // self.batch_split:(i + 1) * batch_size // self.batch_split] for k, v in data.items()}
                    for i in range(self.batch_split)
                ]
        elif isinstance(data, list):
            data_list = data
        else:
            raise ValueError('Data must be a dict or a list of dicts.')
        
        return data_list

    def run_step(self, data_list):
        """
        Run a training step.
        """
        step_log = {'loss': {}, 'status': {}}
        amp_context = partial(torch.autocast, device_type='cuda', dtype=self.mix_precision_dtype) if self.mix_precision_mode == 'amp' else nullcontext
        elastic_controller_context = self.elastic_controller.record if self.elastic_controller_config is not None else nullcontext

        # Train
        losses = []
        statuses = []
        elastic_controller_logs = []
        zero_grad(self.model_params)
        for i, mb_data in enumerate(data_list):
            ## sync at the end of each batch split
            sync_contexts = [self.training_models[name].no_sync for name in self.training_models] if i != len(data_list) - 1 and self.world_size > 1 else [nullcontext]
            with nested_contexts(*sync_contexts), elastic_controller_context():
                with amp_context():
                    loss, status = self.training_losses(**mb_data)
                    l = loss['loss'] / len(data_list)
                    
                    # DEBUG: Print loss for each rank
                    if self.debug:
                        print(f'[Rank {self.rank}/{self.world_size}] Step {self.step} batch {i}: loss={loss["loss"].item():.6f}')
                    
                ## backward
                if self.mix_precision_mode == 'amp' and self.mix_precision_dtype == torch.float16:
                    self.scaler.scale(l).backward()
                elif self.mix_precision_mode == 'inflat_all' and self.mix_precision_dtype == torch.float16:
                    scaled_l = l * (2 ** self.log_scale)
                    scaled_l.backward()
                else:
                    l.backward()
            ## log
            losses.append(dict_foreach(loss, lambda x: x.item() if isinstance(x, torch.Tensor) else x))
            statuses.append(dict_foreach(status, lambda x: x.item() if isinstance(x, torch.Tensor) else x))
            if self.elastic_controller_config is not None:
                elastic_controller_logs.append(self.elastic_controller.log())
        
        # ============================================================
        # DEBUG: Verify DDP gradient synchronization
        # Check if gradients are consistent across ranks after backward
        # DDP automatically all_reduces gradients during the last batch_split's backward
        # After sync, all ranks should have identical gradients
        # ============================================================
        if self.debug and self.world_size > 1:
            self._verify_gradient_sync()
        
        ## gradient clip
        if self.grad_clip is not None:
            if self.mix_precision_mode == 'amp' and self.mix_precision_dtype == torch.float16:
                self.scaler.unscale_(self.optimizer)
            elif self.mix_precision_mode == 'inflat_all':
                model_grads_to_master_grads(self.model_params, self.master_params)
                if self.mix_precision_dtype == torch.float16:
                    self.master_params[0].grad.mul_(1.0 / (2 ** self.log_scale))
            if isinstance(self.grad_clip, float):
                grad_norm = torch.nn.utils.clip_grad_norm_(self.master_params, self.grad_clip)
            else:
                grad_norm = self.grad_clip(self.master_params)
            if torch.isfinite(grad_norm):
                statuses[-1]['grad_norm'] = grad_norm.item()
        ## step
        if self.mix_precision_mode == 'amp' and self.mix_precision_dtype == torch.float16:
            prev_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        elif self.mix_precision_mode == 'inflat_all':
            if self.mix_precision_dtype == torch.float16:
                prev_scale = 2 ** self.log_scale
                if not any(not p.grad.isfinite().all() for p in self.model_params):
                    if self.grad_clip is None:
                        model_grads_to_master_grads(self.model_params, self.master_params)
                        self.master_params[0].grad.mul_(1.0 / (2 ** self.log_scale))
                    self.optimizer.step()
                    master_params_to_model_params(self.model_params, self.master_params)
                    self.log_scale += self.fp16_scale_growth
                else:
                    self.log_scale -= 1
            else:
                prev_scale = 1.0
                if self.grad_clip is None:
                    model_grads_to_master_grads(self.model_params, self.master_params)
                if not any(not p.grad.isfinite().all() for p in self.master_params):
                    self.optimizer.step()
                    master_params_to_model_params(self.model_params, self.master_params)
                else:
                    print('\n\033[93mWarning: NaN detected in gradients. Skipping update.\033[0m')
        else:
            prev_scale = 1.0
            if not any(not p.grad.isfinite().all() for p in self.model_params):
                self.optimizer.step()
            else:
                print('\n\033[93mWarning: NaN detected in gradients. Skipping update.\033[0m') 
        ## adjust learning rate
        if self.lr_scheduler_config is not None:
            statuses[-1]['lr'] = self.lr_scheduler.get_last_lr()[0]
            self.lr_scheduler.step()

        # Logs
        step_log['loss'] = dict_reduce(losses, lambda x: np.mean(x))
        step_log['status'] = dict_reduce(statuses, lambda x: np.mean(x), special_func={'min': lambda x: np.min(x), 'max': lambda x: np.max(x)})
        if self.elastic_controller_config is not None:
            step_log['elastic'] = dict_reduce(elastic_controller_logs, lambda x: np.mean(x))
        if self.grad_clip is not None:
            step_log['grad_clip'] = self.grad_clip if isinstance(self.grad_clip, float) else self.grad_clip.log()
            
        # Check grad and norm of each param
        if self.log_param_stats:
            param_norms = {}
            param_grads = {}
            for model_name, model in self.models.items():
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        param_norms[f'{model_name}.{name}'] = param.norm().item()
                        if param.grad is not None and torch.isfinite(param.grad).all():
                            param_grads[f'{model_name}.{name}'] = param.grad.norm().item() / prev_scale
            step_log['param_norms'] = param_norms
            step_log['param_grads'] = param_grads

        # Update exponential moving average
        if self.is_master:
            self.update_ema()

        return step_log

    def save_logs(self):
        log_str = '\n'.join([
            f'{step}: {json.dumps(dict_foreach(log, lambda x: float(x)))}' for step, log in self.log
        ])
        
        # Accumulate logs in memory and overwrite file each time (S3 FUSE does not support append)
        if not hasattr(self, '_log_buffer'):
            self._log_buffer = []
        self._log_buffer.append(log_str)
        try:
            with open(self._log_file, 'w') as log_file:
                log_file.write('\n'.join(self._log_buffer) + '\n')
        except Exception as e:
            print(f'\033[93m[WARN] Failed to write log file: {e}\033[0m')

        # show with mlflow
        log_show = [l for _, l in self.log if not dict_any(l, lambda x: np.isnan(x))]
        log_show = dict_reduce(log_show, lambda x: np.mean(x))
        log_show = dict_flatten(log_show, sep='/')
        if self.writer is not None:
            for key, value in log_show.items():
                self.writer.add_scalar(key, value, self.step)
        
        # Log to wandb
        if self.wandb_run is not None:
            wandb_log = {key: value for key, value in log_show.items()}
            wandb_log['step'] = self.step
            self.wandb_run.log(wandb_log, step=self.step)
        
        self.log = []
        
    def check_abort(self):
        """
        Check if training should be aborted due to certain conditions.
        """
        # 1. If log_scale in inflat_all mode is less than 0
        if self.mix_precision_dtype == torch.float16 and \
           self.mix_precision_mode == 'inflat_all' and \
           self.log_scale < 0:
            if self.is_master:
                print ('\n\n\033[91m')
                print (f'ABORT: log_scale in inflat_all mode is less than 0 at step {self.step}.')
                print ('This indicates that the model is diverging. You should look into the model and the data.')
                print ('\033[0m')
                self.save(non_blocking=False)
                self.save_logs()
            if self.world_size > 1:
                dist.barrier()
            raise ValueError('ABORT: log_scale in inflat_all mode is less than 0.')

    def run(self):
        """
        Run training.
        """
        if self.is_master:
            print('\nStarting training...')
            if self.i_sample != -1:
                try:
                    self.snapshot_dataset(num_samples=self.snapshot_num_samples, batch_size=self.snapshot_batch_size)
                except (RuntimeError, Exception) as e:
                    print(f'\033[93m[WARN] snapshot_dataset failed, skipping: {e}\033[0m')
                    torch.cuda.empty_cache()
            else:
                print('[INFO] i_sample=-1, all snapshots disabled.')
        if self.i_sample != -1:
            if self.step == 0:
                try:
                    self.snapshot(suffix='init', num_samples=self.snapshot_num_samples, batch_size=self.snapshot_batch_size)
                except (RuntimeError, Exception) as e:
                    print(f'\033[93m[WARN] snapshot (init) failed, skipping: {e}\033[0m')
                    torch.cuda.empty_cache()
            else: # resume
                try:
                    self.snapshot(suffix=f'resume_step{self.step:07d}', num_samples=self.snapshot_num_samples, batch_size=self.snapshot_batch_size)
                except (RuntimeError, Exception) as e:
                    print(f'\033[93m[WARN] snapshot (resume) failed, skipping: {e}\033[0m')
                    torch.cuda.empty_cache()

        time_last_print = 0.0
        time_elapsed = 0.0
        while self.step < self.max_steps:
            time_start = time.time()

            data_list = self.load_data()
            step_log = self.run_step(data_list)

            time_end = time.time()
            time_elapsed += time_end - time_start

            self.step += 1

            # Print progress
            if self.is_master and self.step % self.i_print == 0:
                speed = self.i_print / (time_elapsed - time_last_print) * 3600
                columns = [
                    f'Step: {self.step}/{self.max_steps} ({self.step / self.max_steps * 100:.2f}%)',
                    f'Elapsed: {time_elapsed / 3600:.2f} h',
                    f'Speed: {speed:.2f} steps/h',
                    f'ETA: {(self.max_steps - self.step) / speed:.2f} h',
                ]
                print(' | '.join([c.ljust(25) for c in columns]), flush=True)
                time_last_print = time_elapsed

            # Check ddp
            if self.parallel_mode == 'ddp' and self.world_size > 1 and self.i_ddpcheck is not None and self.step % self.i_ddpcheck == 0:
                self.check_ddp()

            # Sample images
            if self.i_sample != -1 and self.step % self.i_sample == 0:
                try:
                    self.snapshot(num_samples=self.snapshot_num_samples, batch_size=self.snapshot_batch_size)
                except (RuntimeError, Exception) as e:
                    if self.is_master:
                        print(f'\033[93m[WARN] snapshot at step {self.step} failed, skipping: {e}\033[0m')
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

            if self.is_master:
                self.log.append((self.step, {}))

                # Log time
                self.log[-1][1]['time'] = {
                    'step': time_end - time_start,
                    'elapsed': time_elapsed,
                }

                # Log losses
                if step_log is not None:
                    self.log[-1][1].update(step_log)

                # Log scale
                if self.mix_precision_dtype == torch.float16:
                    if self.mix_precision_mode == 'amp':
                        self.log[-1][1]['scale'] = self.scaler.get_scale()
                    elif self.mix_precision_mode == 'inflat_all':
                        self.log[-1][1]['log_scale'] = self.log_scale

                # Save log
                if self.step % self.i_log == 0:
                    self.save_logs()

                # Save checkpoint
                if self.step % self.i_save == 0:
                    self.save()
                    
            # Check abort
            self.check_abort()

        if self.i_sample != -1:
            try:
                self.snapshot(suffix='final', num_samples=self.snapshot_num_samples, batch_size=self.snapshot_batch_size)
            except (RuntimeError, Exception) as e:
                if self.is_master:
                    print(f'\033[93m[WARN] snapshot (final) failed, skipping: {e}\033[0m')
            torch.cuda.empty_cache()
        if self.world_size > 1:
            dist.barrier()
        if self.is_master:
            self.writer.close()
            print('Training finished.')
            
    def profile(self, wait=2, warmup=3, active=5):
        """
        Profile the training loop.
        """
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(self.output_dir, 'profile')),
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(wait + warmup + active):
                self.run_step()
                prof.step()
