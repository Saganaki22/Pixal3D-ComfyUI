from typing import *
import json
from abc import abstractmethod
import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class StandardDatasetBase(Dataset):
    """
    Base class for standard datasets.

    Args:
        roots (str): paths to the dataset
        skip_list (str, optional): path to a file containing sha256 hashes to skip (one per line)
                                   Format: "dataset/sha256" (e.g., "ABO/6a79dbb5...")
        skip_aesthetic_score_datasets (list, optional): list of dataset names to skip aesthetic score check
                                                        (e.g., ["texverse"] for datasets without aesthetic_score)
    """

    def __init__(self,
        roots: str,
        skip_list: Optional[str] = None,
        skip_aesthetic_score_datasets: Optional[List[str]] = None,
    ):
        super().__init__()
        
        # Datasets to skip aesthetic score check
        self.skip_aesthetic_score_datasets = set(skip_aesthetic_score_datasets or [])
        
        # Load skip list if provided
        self.skip_set = set()
        if skip_list is not None and os.path.exists(skip_list):
            with open(skip_list, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        self.skip_set.add(line)
            print(f'Loaded {len(self.skip_set)} items from skip_list: {skip_list}')
        
        try:
            self.roots = json.loads(roots)
            root_type = 'obj'
        except:
            self.roots = roots.split(',')
            root_type = 'list'
        self.instances = []
        self.metadata = pd.DataFrame()
        
        self._stats = {}
        if root_type == 'obj':
            for key, root in self.roots.items():
                self._stats[key] = {}
                metadata = pd.DataFrame(columns=['sha256']).set_index('sha256')
                
                # Only merge key fields from ss_latent and render_cond
                # Exclude base, because cond_rendered=False in base/metadata.csv would incorrectly overwrite real values
                for sub_key, r in root.items():
                    if sub_key == 'base':
                        continue  # Skip base directory
                    metadata_file = os.path.join(r, 'metadata.csv')
                    if os.path.exists(metadata_file):
                        metadata = metadata.combine_first(pd.read_csv(metadata_file).set_index('sha256'))
                
                # Read aesthetic_score separately from base (avoid reading other potentially conflicting columns)
                if 'base' in root:
                    base_metadata_file = os.path.join(root['base'], 'metadata.csv')
                    if os.path.exists(base_metadata_file):
                        base_df = pd.read_csv(base_metadata_file).set_index('sha256')
                        if 'aesthetic_score' in base_df.columns and 'aesthetic_score' not in metadata.columns:
                            metadata['aesthetic_score'] = base_df['aesthetic_score']
                
                self._stats[key]['Total'] = len(metadata)
                metadata, stats = self.filter_metadata(metadata, dataset_name=key)
                self._stats[key].update(stats)
                
                # Filter out items in skip_list
                skipped_count = 0
                for sha256 in metadata.index.values:
                    skip_key = f'{key}/{sha256}'
                    if skip_key in self.skip_set:
                        skipped_count += 1
                    else:
                        self.instances.append((root, sha256, key))
                if skipped_count > 0:
                    self._stats[key]['Skipped (skip_list)'] = skipped_count
                    self._stats[key]['After skip_list'] = len(metadata) - skipped_count
                
                self.metadata = pd.concat([self.metadata, metadata])
        else:
            for root in self.roots:
                key = os.path.basename(root)
                self._stats[key] = {}
                metadata = pd.read_csv(os.path.join(root, 'metadata.csv'))
                self._stats[key]['Total'] = len(metadata)
                metadata, stats = self.filter_metadata(metadata, dataset_name=key)
                self._stats[key].update(stats)
                
                # Filter out items in skip_list
                skipped_count = 0
                for sha256 in metadata['sha256'].values:
                    skip_key = f'{key}/{sha256}'
                    if skip_key in self.skip_set:
                        skipped_count += 1
                    else:
                        self.instances.append((root, sha256, key))
                if skipped_count > 0:
                    self._stats[key]['Skipped (skip_list)'] = skipped_count
                    self._stats[key]['After skip_list'] = len(metadata) - skipped_count
                metadata.set_index('sha256', inplace=True)
                self.metadata = pd.concat([self.metadata, metadata])
            
    @abstractmethod
    def filter_metadata(self, metadata: pd.DataFrame, dataset_name: str = None) -> Tuple[pd.DataFrame, Dict[str, int]]:
        pass
    
    @abstractmethod
    def get_instance(self, root, instance: str) -> Dict[str, Any]:
        pass
        
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index) -> Dict[str, Any]:
        try:
            root, instance, dataset_name = self.instances[index]
            pack = self.get_instance(root, instance)
            pack['_dataset_name'] = dataset_name
            pack['_sha256'] = instance
            return pack
        except Exception as e:
            print(f'Error loading {self.instances[index][1]}: {e}')
            return self.__getitem__(np.random.randint(0, len(self)))
        
    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f'  - Total instances: {len(self)}')
        lines.append(f'  - Sources:')
        for key, stats in self._stats.items():
            lines.append(f'    - {key}:')
            for k, v in stats.items():
                lines.append(f'      - {k}: {v}')
        return '\n'.join(lines)


class ImageConditionedMixin:
    def __init__(self, roots, *, image_size=518, **kwargs):
        self.image_size = image_size
        super().__init__(roots, **kwargs)
    
    def filter_metadata(self, metadata, dataset_name=None):
        metadata, stats = super().filter_metadata(metadata, dataset_name=dataset_name)
        metadata = metadata[metadata['cond_rendered'].notna()]
        stats['Cond rendered'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
       
        image_root = os.path.join(root['render_cond'], instance)
        with open(os.path.join(image_root, 'transforms.json')) as f:
            metadata = json.load(f)
        n_views = len(metadata['frames'])
        view = np.random.randint(n_views)
        metadata = metadata['frames'][view]

        image_path = os.path.join(image_root, metadata['file_path'])
        image = Image.open(image_path)

        alpha = np.array(image.getchannel(3))
        bbox = np.array(alpha).nonzero()
        bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
        aug_hsize = hsize
        aug_center_offset = [0, 0]
        aug_center = [center[0] + aug_center_offset[0], center[1] + aug_center_offset[1]]
        aug_bbox = [int(aug_center[0] - aug_hsize), int(aug_center[1] - aug_hsize), int(aug_center[0] + aug_hsize), int(aug_center[1] + aug_hsize)]
        image = image.crop(aug_bbox)

        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        image = image * alpha.unsqueeze(0)
        pack['cond'] = image
       
        return pack


class ViewImageConditionedMixin:
    """
    Mixin for view-based image-conditioned datasets.
    
    This mixin is designed for datasets where ss_latent is stored per-view (view{XX}.npz),
    and needs to load the corresponding view image and scale from view{XX}_scale.json.
    
    Args:
        image_size: Target image size
        load_camera_info: Whether to load camera information for view-aligned conditioning
    """
    def __init__(self, roots, *, image_size=518, load_camera_info=False, **kwargs):
        self.image_size = image_size
        # self.load_camera_info = load_camera_info
        super().__init__(roots, **kwargs)
    
    def filter_metadata(self, metadata, dataset_name=None):
        metadata, stats = super().filter_metadata(metadata, dataset_name=dataset_name)
        metadata = metadata[metadata['cond_rendered'].notna()]
        stats['Cond rendered'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        """
        Get instance with view-aligned image and camera info.
        
        Expects parent class to set:
            - pack['x_0']: the latent tensor
            - self._current_view_idx: the selected view index
            - self._current_latent_dir: the latent directory path
        """
        pack = super().get_instance(root, instance)
        
        # Get view_idx from parent class (set by SparseStructureLatentView)
        if not hasattr(self, '_current_view_idx'):
            raise RuntimeError("Parent class must set '_current_view_idx' before calling ViewImageConditionedMixin.get_instance")
        if not hasattr(self, '_current_latent_dir'):
            raise RuntimeError("Parent class must set '_current_latent_dir' before calling ViewImageConditionedMixin.get_instance")
        view_idx = self._current_view_idx
        latent_dir = self._current_latent_dir
        
        # Load image metadata
        image_root = os.path.join(root['render_cond'], instance)
        with open(os.path.join(image_root, 'transforms.json')) as f:
            metadata = json.load(f)
        
        # Load corresponding image for this view
        frame_metadata = metadata['frames'][view_idx]
        image_path = os.path.join(image_root, frame_metadata['file_path'])
        image = Image.open(image_path)

        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        image = image * alpha.unsqueeze(0)
        pack['cond'] = image
        
        # Load camera info if requested
   
        # camera_angle_x: check frame first, then root metadata
        if 'camera_angle_x' in frame_metadata:
            camera_angle_x = float(frame_metadata['camera_angle_x'])
        elif 'camera_angle_x' in metadata:
            camera_angle_x = float(metadata['camera_angle_x'])
        else:
            raise KeyError(f"'camera_angle_x' not found in transforms.json for {instance}")
        pack['camera_angle_x'] = torch.tensor(camera_angle_x, dtype=torch.float32)
        
        # transform_matrix
        if 'transform_matrix' not in frame_metadata:
            raise KeyError(f"'transform_matrix' not found in frame {view_idx} for {instance}")
        transform_matrix = torch.tensor(frame_metadata['transform_matrix'], dtype=torch.float32)
        distance = torch.norm(transform_matrix[:3, 3]).item()
            
        pack['camera_distance'] = torch.tensor(distance, dtype=torch.float32)
        # NOTE: Do NOT pass transform_matrix to ProjGrid.
        # shape_latent space objects are already rotated to front-view by transform_mesh,
        # so ProjGrid should use the default front_view_transform_matrix + distance.
        # pack['transform_matrix'] = transform_matrix
        
        # Load mesh_scale from ss_latent directory's view{XX}_scale.json
        scale_json_path = os.path.join(latent_dir, f'view{view_idx:02d}_scale.json')
        if not os.path.exists(scale_json_path):
            raise FileNotFoundError(f"Scale file not found: {scale_json_path}")
        with open(scale_json_path) as f:
            scale_data = json.load(f)
        if 'total_scale' not in scale_data:
            raise KeyError(f"'total_scale' not found in {scale_json_path}")
        pack['mesh_scale'] = torch.tensor(float(scale_data['total_scale']), dtype=torch.float32)
       
        return pack


class MultiImageConditionedMixin:
    def __init__(self, roots, *, image_size=518, max_image_cond_view = 4, **kwargs):
        self.image_size = image_size
        self.max_image_cond_view = max_image_cond_view
        super().__init__(roots, **kwargs)

    def filter_metadata(self, metadata, dataset_name=None):
        metadata, stats = super().filter_metadata(metadata, dataset_name=dataset_name)
        metadata = metadata[metadata['cond_rendered'].notna()]
        stats['Cond rendered'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
       
        image_root = os.path.join(root['render_cond'], instance)
        with open(os.path.join(image_root, 'transforms.json')) as f:
            metadata = json.load(f)

        n_views = len(metadata['frames'])
        n_sample_views = np.random.randint(1, self.max_image_cond_view+1)

        assert n_views >= n_sample_views, f'Not enough views to sample {n_sample_views} unique images.'

        sampled_views = np.random.choice(n_views, size=n_sample_views, replace=False)

        cond_images = []
        for v in sampled_views:
            frame_info = metadata['frames'][v]
            image_path = os.path.join(image_root, frame_info['file_path'])
            image = Image.open(image_path)

            alpha = np.array(image.getchannel(3))
            bbox = np.array(alpha).nonzero()
            bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
            center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
            aug_hsize = hsize
            aug_center = center
            aug_bbox = [
                int(aug_center[0] - aug_hsize),
                int(aug_center[1] - aug_hsize),
                int(aug_center[0] + aug_hsize),
                int(aug_center[1] + aug_hsize),
            ]

            img = image.crop(aug_bbox)
            img = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            alpha = img.getchannel(3)
            img = img.convert('RGB')
            img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
            alpha = torch.tensor(np.array(alpha)).float() / 255.0
            img = img * alpha.unsqueeze(0)

            cond_images.append(img)

        pack['cond'] = [torch.stack(cond_images, dim=0)]  # (V,3,H,W)
        return pack
