import os
import json
from typing import *
import numpy as np
import torch
import utils3d
from .. import models
from .components import ImageConditionedMixin, ViewImageConditionedMixin
from ..modules.sparse import SparseTensor
from .structured_latent import SLatVisMixin, SLat
from ..utils.render_utils import get_renderer, yaw_pitch_r_fov_to_extrinsics_intrinsics
from ..utils.data_utils import load_balanced_group_indices


class SLatShapeVisMixin(SLatVisMixin):
    def _loading_slat_dec(self):
        if self.slat_dec is not None:
            return
        if self.slat_dec_path is not None:
            cfg = json.load(open(os.path.join(self.slat_dec_path, 'config.json'), 'r'))
            decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
            ckpt_path = os.path.join(self.slat_dec_path, 'ckpts', f'decoder_{self.slat_dec_ckpt}.pt')
            decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        else:
            decoder = models.from_pretrained(self.pretrained_slat_dec)
        decoder.set_resolution(self.resolution)
        self.slat_dec = decoder.cuda().eval()

    @torch.no_grad()
    def visualize_sample(
        self, 
        x_0: Union[SparseTensor, dict],
        camera_angle_x: Optional[torch.Tensor] = None,
        camera_distance: Optional[torch.Tensor] = None,
        mesh_scale: Optional[torch.Tensor] = None,
    ):
        """
        Visualize shape samples.
        
        Args:
            x_0: SparseTensor or dict containing 'x_0'
            camera_angle_x: Optional [B] camera FOV angle in radians
            camera_distance: Optional [B] camera distance for GT view rendering
            mesh_scale: Optional [B] mesh scale factor for coordinate alignment
            
        Returns:
            dict with:
                'multiview': [B, 3, 1024, 1024] - 4 fixed views rendered in 2x2 grid (normal)
                'gt_view': [B, 3, 512, 512] - GT camera view (if camera params provided)
        """
        x_0 = x_0 if isinstance(x_0, SparseTensor) else x_0['x_0']
        reps = self.decode_latent(x_0.cuda())
        
        # build fixed camera views (4 views: 0°, 90°, 180°, 270°)
        yaw = [0, np.pi/2, np.pi, 3*np.pi/2]
        yaw_offset = -16 / 180 * np.pi
        yaw = [y + yaw_offset for y in yaw]
        pitch = [20 / 180 * np.pi for _ in range(4)]
        fixed_exts, fixed_ints = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, 2, 30)
        
        # Check if we have GT camera parameters for GT view rendering
        has_gt_camera = (
            camera_angle_x is not None and 
            camera_distance is not None and 
            mesh_scale is not None
        )
        
        # render
        renderer = get_renderer(reps[0])
        multiview_images = []
        gt_view_images = []
        
        for i, representation in enumerate(reps):
            # Render 4 fixed views (2x2 grid)
            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            
            # Validate mesh data before rasterization
            verts = representation.vertices
            faces = representation.faces
            if verts.shape[0] == 0 or faces.shape[0] == 0:
                print(f"[visualize_sample] Warning: sample {i} has empty mesh, skipping")
                multiview_images.append(image)
                continue
            if faces.max() >= verts.shape[0]:
                print(f"[visualize_sample] Warning: sample {i} has out-of-bound face indices "
                      f"(max face idx={faces.max().item()}, num verts={verts.shape[0]}), skipping")
                multiview_images.append(image)
                continue
            if torch.isnan(verts).any() or torch.isinf(verts).any():
                print(f"[visualize_sample] Warning: sample {i} has NaN/Inf vertices, skipping")
                multiview_images.append(image)
                continue
            
            try:
                for j, (ext, intr) in enumerate(zip(fixed_exts, fixed_ints)):
                    res = renderer.render(representation, ext, intr)
                    image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['normal']
            except RuntimeError as e:
                print(f"[visualize_sample] Warning: render failed for sample {i}: {e}")
                image = torch.zeros(3, 1024, 1024).cuda()
            multiview_images.append(image)
            
            # Render GT camera view using the fixed front view (same as sparse_structure_latent.py)
            if has_gt_camera:
                # The GT view should match exactly how ProjGrid projects 3D points to 2D.
                # 
                # In image_conditioned_proj.py (ProjGrid.forward):
                # 1. grid_points are in [-1, 1]^3 (from torch.linspace(-1, 1, res))
                # 2. grid_points are rotated by rotation_matrix (Y-Z swap): x'=x, y'=-z, z'=y
                # 3. grid_points are scaled: grid_points / mesh_scale / 2
                # 4. Points are projected using front_view_transform_matrix with distance
                #
                # Mesh vertices are in [-0.5, 0.5]^3. To match ProjGrid's coordinate space,
                # we need to scale them: vertices / mesh_scale -> [-0.5/s, 0.5/s]^3
                # This is equivalent to ProjGrid's: [-1,1]^3 / scale / 2 -> [-0.5/s, 0.5/s]^3
                #
                # Camera position: ProjGrid camera is at (0, -distance, 0) in Blender coords (Z-up).
                # After inverse rotation to mesh space, camera is at (0, 0, distance).
                
                scale = mesh_scale[i].item()
                distance = camera_distance[i].item()
                fov = camera_angle_x[i].item()
                device = representation.vertices.device
                
                # Scale mesh vertices to match ProjGrid's projection space
                from ..representations import Mesh
                scaled_rep = Mesh(
                    vertices=representation.vertices / scale,
                    faces=representation.faces,
                )
                
                cam_pos = torch.tensor([0.0, 0.0, distance], device=device)
                look_at = torch.tensor([0.0, 0.0, 0.0], device=device)
                cam_up = torch.tensor([0.0, 1.0, 0.0], device=device)
                
                gt_ext = utils3d.torch.extrinsics_look_at(cam_pos, look_at, cam_up)
                gt_int = utils3d.torch.intrinsics_from_fov_xy(
                    torch.tensor(fov, device=device),
                    torch.tensor(fov, device=device)
                )
                
                gt_ext = gt_ext.to(device)
                gt_int = gt_int.to(device)
                
                # Use scaled mesh renderer with appropriate near/far for smaller mesh
                mesh_half_size = 0.5 / scale
                renderer.rendering_options.near = max(0.01, distance - mesh_half_size - 0.5)
                renderer.rendering_options.far = distance + mesh_half_size + 0.5
                
                try:
                    gt_res = renderer.render(scaled_rep, gt_ext, gt_int)
                    gt_view_images.append(gt_res['normal'])
                except RuntimeError as e:
                    print(f"[visualize_sample] Warning: GT view render failed for sample {i}: {e}")
                    gt_view_images.append(torch.full((3, 512, 512), 0.5, device=device))
        
        result = {
            'multiview': torch.stack(multiview_images),
        }
        
        if has_gt_camera and len(gt_view_images) > 0:
            result['gt_view'] = torch.stack(gt_view_images)
            
        return result
    
    
class SLatShape(SLatShapeVisMixin, SLat):
    """
    structured latent for shape generation
    
    Args:
        roots (str): path to the dataset
        resolution (int): resolution of the shape
        min_aesthetic_score (float): minimum aesthetic score
        max_tokens (int): maximum number of tokens
        latent_key (str): key of the latent to be used
        normalization (dict): normalization stats
        pretrained_slat_dec (str): name of the pretrained slat decoder
        slat_dec_path (str): path to the slat decoder, if given, will override the pretrained_slat_dec
        slat_dec_ckpt (str): name of the slat decoder checkpoint
        skip_list (str, optional): path to a file containing sha256 hashes to skip
        skip_aesthetic_score_datasets (list, optional): list of dataset names to skip aesthetic score check
    """
    def __init__(self,
        roots: str,
        *,
        resolution: int,
        min_aesthetic_score: float = 5.0,
        max_tokens: int = 32768,
        normalization: Optional[dict] = None,
        pretrained_slat_dec: str = 'microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16',
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
        skip_list: Optional[str] = None,
        skip_aesthetic_score_datasets: Optional[list] = None,
    ):
        super().__init__(
            roots,
            min_aesthetic_score=min_aesthetic_score,
            max_tokens=max_tokens,
            latent_key='shape_latent',
            normalization=normalization,
            pretrained_slat_dec=pretrained_slat_dec,
            slat_dec_path=slat_dec_path,
            slat_dec_ckpt=slat_dec_ckpt,
            skip_list=skip_list,
            skip_aesthetic_score_datasets=skip_aesthetic_score_datasets,
        )
        self.resolution = resolution


class ImageConditionedSLatShape(ImageConditionedMixin, SLatShape):
    """
    Image conditioned structured latent for shape generation
    """
    pass


class SLatShapeView(SLatShapeVisMixin, SLat):
    """
    View-based structured latent for shape generation.
    
    Data format: {sha256}/view{XX}.npz where each npz contains 'coords' and 'feats' keys.
    
    Args:
        roots (str): path to the dataset
        resolution (int): resolution of the shape
        min_aesthetic_score (float): minimum aesthetic score
        max_tokens (int): maximum number of tokens
        num_views (int): Number of views to use (0 to num_views-1). Default is 2.
        normalization (dict): normalization stats
        pretrained_slat_dec (str): name of the pretrained slat decoder
        slat_dec_path (str): path to the slat decoder, if given, will override the pretrained_slat_dec
        slat_dec_ckpt (str): name of the slat decoder checkpoint
        skip_list (str, optional): path to a file containing sha256 hashes to skip
        skip_aesthetic_score_datasets (list, optional): list of dataset names to skip aesthetic score check
    """
    def __init__(self,
        roots: str,
        *,
        resolution: int,
        min_aesthetic_score: float = 5.0,
        max_tokens: int = 32768,
        num_views: int = 2,
        normalization: Optional[dict] = None,
        pretrained_slat_dec: str = 'microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16',
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
        skip_list: Optional[str] = None,
        skip_aesthetic_score_datasets: Optional[list] = None,
    ):
        self.normalization = normalization
        self.min_aesthetic_score = min_aesthetic_score
        self.max_tokens = max_tokens
        self.num_views = num_views
        self.latent_key = 'shape_latent'
        self.value_range = (0, 1)
        
        # Initialize parent with SLatVisMixin parameters
        from .components import StandardDatasetBase
        SLatVisMixin.__init__(
            self,
            roots,
            pretrained_slat_dec=pretrained_slat_dec,
            slat_dec_path=slat_dec_path,
            slat_dec_ckpt=slat_dec_ckpt,
        )
        StandardDatasetBase.__init__(self, roots, skip_list=skip_list, skip_aesthetic_score_datasets=skip_aesthetic_score_datasets)
        
        self.resolution = resolution
        
        # Calculate loads for load balancing
        self.loads = []
        for _, sha256, _ in self.instances:
            if f'{self.latent_key}_tokens' in self.metadata.columns:
                try:
                    self.loads.append(self.metadata.loc[sha256, f'{self.latent_key}_tokens'])
                except:
                    self.loads.append(self.max_tokens)
            else:
                self.loads.append(self.max_tokens)
        
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(1, -1)
            self.std = torch.tensor(self.normalization['std']).reshape(1, -1)

    def filter_metadata(self, metadata, dataset_name=None):
        stats = {}
        # View-based shape_latent uses columns like shape_latent_view00_encoded, shape_latent_view01_encoded, etc.
        required_view_cols = [f'shape_latent_view{i:02d}_encoded' for i in range(self.num_views)]
        existing_view_cols = [col for col in required_view_cols if col in metadata.columns]
        
        if existing_view_cols:
            # Filter rows where all required views are encoded
            # Note: NaN should be treated as False, so use == True for explicit comparison
            has_all_views = (metadata[existing_view_cols] == True).all(axis=1)
            metadata = metadata[has_all_views]
            stats[f'With {self.num_views} view latents'] = len(metadata)
        else:
            # Fallback: check shape_latent_encoded column
            if f'{self.latent_key}_encoded' in metadata.columns:
                metadata = metadata[metadata[f'{self.latent_key}_encoded'] == True]
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
        
        # Filter by max_tokens if column exists
        tokens_col = f'{self.latent_key}_tokens'
        if tokens_col in metadata.columns:
            metadata = metadata[metadata[tokens_col] <= self.max_tokens]
            stats[f'Num tokens <= {self.max_tokens}'] = len(metadata)
        
        return metadata, stats

    def get_instance(self, root, instance):
        # View-based format: directory with view{XX}.npz files
        latent_dir = os.path.join(root[self.latent_key], instance)
        
        # Randomly select a view from the configured range
        view_idx = np.random.randint(0, self.num_views)
        view_file = f'view{view_idx:02d}.npz'
        
        # Store view info for ViewImageConditionedMixin
        self._current_view_idx = view_idx
        self._current_latent_dir = latent_dir
        
        data = np.load(os.path.join(latent_dir, view_file))
        coords = torch.tensor(data['coords']).int()
        feats = torch.tensor(data['feats']).float()
        if self.normalization is not None:
            feats = (feats - self.mean) / self.std
        return {
            'coords': coords,
            'feats': feats,
            'view_idx': view_idx,
        }

    @staticmethod
    def collate_fn(batch, split_size=None):
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices([b['coords'].shape[0] for b in batch], split_size)
        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            pack = {}
            coords = []
            feats = []
            layout = []
            start = 0
            for i, b in enumerate(sub_batch):
                coords.append(torch.cat([torch.full((b['coords'].shape[0], 1), i, dtype=torch.int32), b['coords']], dim=-1))
                feats.append(b['feats'])
                layout.append(slice(start, start + b['coords'].shape[0]))
                start += b['coords'].shape[0]
            coords = torch.cat(coords)
            feats = torch.cat(feats)
            pack['x_0'] = SparseTensor(
                coords=coords,
                feats=feats,
            )
            pack['x_0']._shape = torch.Size([len(group), *sub_batch[0]['feats'].shape[1:]])
            pack['x_0'].register_spatial_cache('layout', layout)
            
            # collate other data
            keys = [k for k in sub_batch[0].keys() if k not in ['coords', 'feats']]
            for k in keys:
                if isinstance(sub_batch[0][k], torch.Tensor):
                    pack[k] = torch.stack([b[k] for b in sub_batch])
                elif isinstance(sub_batch[0][k], list):
                    pack[k] = sum([b[k] for b in sub_batch], [])
                else:
                    pack[k] = [b[k] for b in sub_batch]
                    
            packs.append(pack)
          
        if split_size is None:
            return packs[0]
        return packs


class ViewImageConditionedSLatShapeView(ViewImageConditionedMixin, SLatShapeView):
    """
    Image-conditioned view-based structured latent for shape generation.
    
    Loads shape_latent from {sha256}/view{XX}.npz format and pairs with 
    corresponding view from render_cond.
    
    Uses ViewImageConditionedMixin which reads mesh_scale from view{XX}_scale.json.
    """
    pass
