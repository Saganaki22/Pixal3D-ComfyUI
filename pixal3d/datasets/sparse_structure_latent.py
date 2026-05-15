import os
import json
from typing import *
import numpy as np
import torch
import utils3d
from PIL import Image
from ..representations import Voxel
from ..renderers import VoxelRenderer
from .components import StandardDatasetBase, ImageConditionedMixin, ViewImageConditionedMixin
from .. import models
from ..utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics


class SparseStructureLatentVisMixin:
    def __init__(
        self,
        *args,
        pretrained_ss_dec: str = 'JeffreyXiang/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.json',
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ss_dec = None
        self.pretrained_ss_dec = pretrained_ss_dec
        self.ss_dec_path = ss_dec_path
        self.ss_dec_ckpt = ss_dec_ckpt
        
    def _loading_ss_dec(self):
        if self.ss_dec is not None:
            return
        if self.ss_dec_path is not None:
            cfg = json.load(open(os.path.join(self.ss_dec_path, 'config.json'), 'r'))
            decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
            ckpt_path = os.path.join(self.ss_dec_path, 'ckpts', f'decoder_{self.ss_dec_ckpt}.pt')
            decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        else:
            decoder = models.from_pretrained(self.pretrained_ss_dec)
        self.ss_dec = decoder.cuda().eval()

    def _delete_ss_dec(self):
        del self.ss_dec
        self.ss_dec = None

    @torch.no_grad()
    def decode_latent(self, z, batch_size=4):
        self._loading_ss_dec()
        ss = []
        if self.normalization:
            z = z * self.std.to(z.device) + self.mean.to(z.device)
        for i in range(0, z.shape[0], batch_size):
            ss.append(self.ss_dec(z[i:i+batch_size]))
        ss = torch.cat(ss, dim=0)
        self._delete_ss_dec()
        return ss

    @torch.no_grad()
    def visualize_sample(
        self, 
        x_0: Union[torch.Tensor, dict],
        camera_angle_x: Optional[torch.Tensor] = None,
        camera_distance: Optional[torch.Tensor] = None,
        mesh_scale: Optional[torch.Tensor] = None,
    ):
        """
        Visualize sparse structure samples.
        
        Args:
            x_0: Latent tensor [B, C, D, H, W] or dict containing 'x_0'
            camera_angle_x: Optional [B] camera FOV angle in radians
            camera_distance: Optional [B] camera distance for GT view rendering
            mesh_scale: Optional [B] mesh scale factor for coordinate alignment
            
        Returns:
            dict with:
                'multiview': [B, 3, 1024, 1024] - 4 fixed views rendered in 2x2 grid
                'gt_view': [B, 3, 512, 512] - GT camera view (if camera params provided)
        """
        x_0 = x_0 if isinstance(x_0, torch.Tensor) else x_0['x_0']
        x_0 = self.decode_latent(x_0.cuda())
        
        renderer = VoxelRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.ssaa = 4
        
        # Build fixed camera views (4 views: 0°, 90°, 180°, 270°)
        yaw = [0, np.pi/2, np.pi, 3*np.pi/2]
        yaw_offset = -16 / 180 * np.pi
        yaw = [y + yaw_offset for y in yaw]
        pitch = [20 / 180 * np.pi for _ in range(4)]
        fixed_exts, fixed_ints = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, 2, 30)

        # Check if we have GT camera parameters for front view rendering
        # GT view uses the fixed front_view_transform_matrix from image_conditioned_proj.py
        has_gt_camera = (
            camera_angle_x is not None and 
            camera_distance is not None and 
            mesh_scale is not None
        )
        
        multiview_images = []
        gt_view_images = []
        
        # Build each representation
        x_0 = x_0.cuda()
        for i in range(x_0.shape[0]):
            coords = torch.nonzero(x_0[i, 0] > 0, as_tuple=False)
            resolution = x_0.shape[-1]
            color = coords / resolution
            
            # Standard voxel for fixed multiview rendering (origin at [-0.5, -0.5, -0.5])
            rep = Voxel(
                origin=[-0.5, -0.5, -0.5],
                voxel_size=1/resolution,
                coords=coords,
                attrs=color,
                layout={
                    'color': slice(0, 3),
                }
            )
            
            # Render 4 fixed views (2x2 grid)
            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(fixed_exts, fixed_ints)):
                res = renderer.render(rep, ext, intr, colors_overwrite=color)
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
            multiview_images.append(image)
            
            # Render GT camera view using the fixed front view from image_conditioned_proj.py
            if has_gt_camera:
                # The GT view should match exactly how ProjGrid projects 3D points to 2D.
                # 
                # In image_conditioned_proj.py (ProjGrid.forward):
                # 1. grid_points are in [-1, 1]^3 (from torch.linspace(-1, 1, res))
                # 2. grid_points are rotated by rotation_matrix (Y-Z swap): x'=x, y'=-z, z'=y
                # 3. grid_points are scaled: grid_points / mesh_scale / 2
                # 4. Points are projected using front_view_transform_matrix with distance
                #
                # front_view_transform_matrix (camera-to-world):
                # [[1, 0, 0, 0],
                #  [0, 0, -1, -distance],
                #  [0, 1, 0, 0],
                #  [0, 0, 0, 1]]
                # 
                # Camera is at (0, -distance, 0) in Blender coords (Z-up), looking at origin.
                #
                # To match this in VoxelRenderer:
                # 1. Voxel coords [0, res-1] map to positions via: pos = (coords + 0.5) * voxel_size + origin
                # 2. We need these positions to match ProjGrid's transformed grid_points
                # 3. Apply rotation by swapping/flipping coords, then scale voxel_size and origin
                
                scale = mesh_scale[i].item()
                distance = camera_distance[i].item()
                fov = camera_angle_x[i].item()
                
                # Coordinate transformation to match ProjGrid's rotation (x'=x, y'=-z, z'=y)
                # new_coords maps to rotated positions in the same grid structure
                new_coords = torch.zeros_like(coords)
                new_coords[:, 0] = coords[:, 0]                       # x stays
                new_coords[:, 1] = (resolution - 1) - coords[:, 2]    # y' = -z (flip for negation)
                new_coords[:, 2] = coords[:, 1]                       # z' = y
                
                # Voxel position calculation:
                # Original: pos = (coords + 0.5) / res - 0.5  -> range [-0.5, 0.5]
                # We need:  pos = (coords + 0.5) * 2 / res - 1 -> range [-1, 1] (like ProjGrid)
                # Then:     pos_final = pos / scale / 2        -> range [-0.5/scale, 0.5/scale]
                #
                # Combined: pos_final = ((coords + 0.5) * 2 / res - 1) / scale / 2
                #                     = (coords + 0.5) / res / scale - 0.5 / scale
                #                     = (coords + 0.5) * voxel_size + origin
                # where:    voxel_size = 1 / res / scale
                #           origin = -0.5 / scale
                
                scaled_voxel_size = 1.0 / resolution / scale
                scaled_origin = [-0.5 / scale, -0.5 / scale, -0.5 / scale]
                
                rep_scaled = Voxel(
                    origin=scaled_origin,
                    voxel_size=scaled_voxel_size,
                    coords=new_coords,
                    attrs=color,
                    layout={
                        'color': slice(0, 3),
                    }
                )
                
                # Build the fixed front view camera (same as front_view_transform_matrix)
                # Camera at (0, -distance, 0), looking at origin, up is Z
                cam_pos = torch.tensor([0.0, -distance, 0.0], device=coords.device)
                look_at = torch.tensor([0.0, 0.0, 0.0], device=coords.device)
                cam_up = torch.tensor([0.0, 0.0, 1.0], device=coords.device)
                
                gt_ext = utils3d.torch.extrinsics_look_at(cam_pos, look_at, cam_up)
                gt_int = utils3d.torch.intrinsics_from_fov_xy(
                    torch.tensor(fov, device=coords.device),
                    torch.tensor(fov, device=coords.device)
                )
                
                # Ensure tensors are on the correct device (utils3d may not preserve device)
                gt_ext = gt_ext.to(coords.device)
                gt_int = gt_int.to(coords.device)
                
                gt_res = renderer.render(rep_scaled, gt_ext, gt_int, colors_overwrite=color)
                gt_view_images.append(gt_res['color'])
        
        result = {
            'multiview': torch.stack(multiview_images),
        }
        
        if has_gt_camera and len(gt_view_images) > 0:
            result['gt_view'] = torch.stack(gt_view_images)
            
        return result


class SparseStructureLatent(SparseStructureLatentVisMixin, StandardDatasetBase):
    """
    Sparse structure latent dataset
    
    Args:
        roots (str): path to the dataset
        min_aesthetic_score (float): minimum aesthetic score
        normalization (dict): normalization stats
        pretrained_ss_dec (str): name of the pretrained sparse structure decoder
        ss_dec_path (str): path to the sparse structure decoder, if given, will override the pretrained_ss_dec
        ss_dec_ckpt (str): name of the sparse structure decoder checkpoint
        skip_list (str, optional): path to a file containing sha256 hashes to skip
        skip_aesthetic_score_datasets (list, optional): list of dataset names to skip aesthetic score check
    """
    def __init__(self,
        roots: str,
        *,
        min_aesthetic_score: float = 5.0,
        normalization: Optional[dict] = None,
        pretrained_ss_dec: str = 'JeffreyXiang/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16',
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
        skip_list: Optional[str] = None,
        skip_aesthetic_score_datasets: Optional[list] = None,
    ):
        self.min_aesthetic_score = min_aesthetic_score
        self.normalization = normalization
        self.value_range = (0, 1)
        
        super().__init__(
            roots,
            pretrained_ss_dec=pretrained_ss_dec,
            ss_dec_path=ss_dec_path,
            ss_dec_ckpt=ss_dec_ckpt,
            skip_list=skip_list,
            skip_aesthetic_score_datasets=skip_aesthetic_score_datasets,
        )
        
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(-1, 1, 1, 1)
            self.std = torch.tensor(self.normalization['std']).reshape(-1, 1, 1, 1)
  
    def filter_metadata(self, metadata, dataset_name=None):
        stats = {}
        metadata = metadata[metadata['ss_latent_encoded'] == True]
        stats['With latent'] = len(metadata)
        # Skip aesthetic score check for specified datasets (e.g., texverse) or if column doesn't exist
        skip_aesthetic = (
            (dataset_name and dataset_name.lower() in [d.lower() for d in self.skip_aesthetic_score_datasets]) or
            ('aesthetic_score' not in metadata.columns)
        )
        if skip_aesthetic:
            stats[f'Aesthetic score check skipped'] = len(metadata)
        else:
            metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
            stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        return metadata, stats
                
    def get_instance(self, root, instance):
        latent = np.load(os.path.join(root['ss_latent'], f'{instance}.npz'))
        z = torch.tensor(latent['z']).float()
        if self.normalization is not None:
            z = (z - self.mean) / self.std

        pack = {
            'x_0': z,
        }
        return pack


class ImageConditionedSparseStructureLatent(ImageConditionedMixin, SparseStructureLatent):
    """
    Image-conditioned sparse structure dataset
    """
    pass


class SparseStructureLatentView(SparseStructureLatentVisMixin, StandardDatasetBase):
    """
    View-based sparse structure latent dataset.
    
    Data format: {sha256}/view{XX}.npz where each npz contains 'z' key.
    
    Args:
        num_views (int): Number of views to use (0 to num_views-1). Default is 2.
        skip_list (str, optional): path to a file containing sha256 hashes to skip
        skip_aesthetic_score_datasets (list, optional): list of dataset names to skip aesthetic score check
    """
    def __init__(self,
        roots: str,
        *,
        min_aesthetic_score: float = 5.0,
        normalization: Optional[dict] = None,
        num_views: int = 2,
        pretrained_ss_dec: str = 'JeffreyXiang/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16',
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
        skip_list: Optional[str] = None,
        skip_aesthetic_score_datasets: Optional[list] = None,
    ):
        self.min_aesthetic_score = min_aesthetic_score
        self.normalization = normalization
        self.num_views = num_views
        self.value_range = (0, 1)
        
        super().__init__(
            roots,
            pretrained_ss_dec=pretrained_ss_dec,
            ss_dec_path=ss_dec_path,
            ss_dec_ckpt=ss_dec_ckpt,
            skip_list=skip_list,
            skip_aesthetic_score_datasets=skip_aesthetic_score_datasets,
        )
        
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(-1, 1, 1, 1)
            self.std = torch.tensor(self.normalization['std']).reshape(-1, 1, 1, 1)
  
    def filter_metadata(self, metadata, dataset_name=None):
        stats = {}
        # View-based ss_latent uses columns like:
        #   ss_latent_view00_encoded, ss_latent_view01_encoded, ... (view format)
        #   ss_latent_view_scale00_encoded, ss_latent_view_scale01_encoded, ... (view_scale format)
        # Check both formats and use whichever exists (prefer view_scale over view)
        required_view_cols = [f'ss_latent_view_scale{i:02d}_encoded' for i in range(self.num_views)]
        existing_view_cols = [col for col in required_view_cols if col in metadata.columns]
        
        if not existing_view_cols:
            # Fallback to view format
            required_view_cols = [f'ss_latent_view{i:02d}_encoded' for i in range(self.num_views)]
            existing_view_cols = [col for col in required_view_cols if col in metadata.columns]
        
        if existing_view_cols:
            # Filter rows where all required views are encoded
            # Note: NaN should be treated as False, so use == True for explicit comparison
            has_all_views = (metadata[existing_view_cols] == True).all(axis=1)
            metadata = metadata[has_all_views]
            stats[f'With {self.num_views} view latents'] = len(metadata)
        else:
            # Fallback: check ss_latent_encoded column
            if 'ss_latent_encoded' in metadata.columns:
                metadata = metadata[metadata['ss_latent_encoded'] == True]
                stats['With latent'] = len(metadata)
            else:
                raise ValueError(f'No view columns found in metadata: {metadata.columns.tolist()}')
        # Skip aesthetic score check for specified datasets (e.g., texverse) or if column doesn't exist
        skip_aesthetic = (
            (dataset_name and dataset_name.lower() in [d.lower() for d in self.skip_aesthetic_score_datasets]) or
            ('aesthetic_score' not in metadata.columns)
        )
        if skip_aesthetic:
            stats[f'Aesthetic score check skipped'] = len(metadata)
        else:
            metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
            stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        return metadata, stats
                
    def get_instance(self, root, instance):
        # View-based format: directory with view{XX}.npz files
        latent_dir = os.path.join(root['ss_latent'], instance)
        
        # Randomly select a view from the configured range
        view_idx = np.random.randint(0, self.num_views)
        view_file = f'view{view_idx:02d}.npz'
        
        # Store view info for ViewImageConditionedMixin
        self._current_view_idx = view_idx
        self._current_latent_dir = latent_dir
        
        latent = np.load(os.path.join(latent_dir, view_file))
        z = torch.tensor(latent['z']).float()
        if self.normalization is not None:
            z = (z - self.mean) / self.std

        pack = {
            'x_0': z,
            'view_idx': view_idx,
        }
        return pack


class ViewImageConditionedSparseStructureLatentView(ViewImageConditionedMixin, SparseStructureLatentView):
    """
    Image-conditioned view-based sparse structure dataset.
    
    Loads ss_latent from {sha256}/view{XX}.npz format and pairs with 
    corresponding view from render_cond.
    
    Uses ViewImageConditionedMixin which reads mesh_scale from view{XX}_scale.json.
    """
    pass
