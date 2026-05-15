import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import json
from typing import *
import numpy as np
import torch
import cv2
import utils3d
from .. import models
from .components import StandardDatasetBase, ImageConditionedMixin, ViewImageConditionedMixin
from ..modules.sparse import SparseTensor, sparse_cat
from ..representations import MeshWithVoxel
from ..renderers import PbrMeshRenderer, EnvMap
from ..utils.data_utils import load_balanced_group_indices
from ..utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics


class SLatPbrVisMixin:
    def __init__(
        self,
        *args,
        pretrained_pbr_slat_dec: str = 'JeffreyXiang/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16',
        pbr_slat_dec_path: Optional[str] = None,
        pbr_slat_dec_ckpt: Optional[str] = None,
        pretrained_shape_slat_dec: str = 'JeffreyXiang/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16',
        shape_slat_dec_path: Optional[str] = None,
        shape_slat_dec_ckpt: Optional[str] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.pbr_slat_dec = None
        self.pretrained_pbr_slat_dec = pretrained_pbr_slat_dec
        self.pbr_slat_dec_path = pbr_slat_dec_path
        self.pbr_slat_dec_ckpt = pbr_slat_dec_ckpt
        self.shape_slat_dec = None
        self.pretrained_shape_slat_dec = pretrained_shape_slat_dec
        self.shape_slat_dec_path = shape_slat_dec_path
        self.shape_slat_dec_ckpt = shape_slat_dec_ckpt
        
    def _loading_slat_dec(self):
        if self.pbr_slat_dec is not None and self.shape_slat_dec is not None:
            return
        if self.pbr_slat_dec_path is not None:
            cfg = json.load(open(os.path.join(self.pbr_slat_dec_path, 'config.json'), 'r'))
            decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
            ckpt_path = os.path.join(self.pbr_slat_dec_path, 'ckpts', f'decoder_{self.pbr_slat_dec_ckpt}.pt')
            decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        else:
            decoder = models.from_pretrained(self.pretrained_pbr_slat_dec)
        self.pbr_slat_dec = decoder.cuda().eval()

        if self.shape_slat_dec_path is not None:
            cfg = json.load(open(os.path.join(self.shape_slat_dec_path, 'config.json'), 'r'))
            decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
            ckpt_path = os.path.join(self.shape_slat_dec_path, 'ckpts', f'decoder_{self.shape_slat_dec_ckpt}.pt')
            decoder.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
        else:
            decoder = models.from_pretrained(self.pretrained_shape_slat_dec)
        decoder.set_resolution(self.resolution)
        self.shape_slat_dec = decoder.cuda().eval()

    def _delete_slat_dec(self):
        del self.pbr_slat_dec
        self.pbr_slat_dec = None
        del self.shape_slat_dec
        self.shape_slat_dec = None
        
    @torch.no_grad()
    def decode_latent(self, z, shape_z, batch_size=4):
        self._loading_slat_dec()
        reps = []
        if self.shape_slat_normalization is not None:
            shape_z = shape_z * self.shape_slat_std.to(z.device) + self.shape_slat_mean.to(z.device)
        if self.pbr_slat_normalization is not None:
            z = z * self.pbr_slat_std.to(z.device) + self.pbr_slat_mean.to(z.device)
        for i in range(0, z.shape[0], batch_size):
            mesh, subs = self.shape_slat_dec(shape_z[i:i+batch_size], return_subs=True)
            vox = self.pbr_slat_dec(z[i:i+batch_size], guide_subs=subs) * 0.5 + 0.5
            reps.extend([
                MeshWithVoxel(
                    m.vertices, m.faces,
                    origin = [-0.5, -0.5, -0.5],
                    voxel_size = 1 / self.resolution,
                    coords = v.coords[:, 1:],
                    attrs = v.feats,
                    voxel_shape = torch.Size([*v.shape, *v.spatial_shape]),
                    layout = self.layout,
                )
                for m, v in zip(mesh, vox)
            ])
        self._delete_slat_dec()
        return reps
    
    @torch.no_grad()
    def visualize_sample(self, sample: dict):
        shape_z = sample['concat_cond'].cuda()
        z = sample['x_0'].cuda()
        reps = self.decode_latent(z, shape_z)
        
        # Extract camera parameters for GT view rendering (if available)
        camera_angle_x = sample.get('camera_angle_x')
        camera_distance = sample.get('camera_distance')
        mesh_scale = sample.get('mesh_scale')
        has_gt_camera = (
            camera_angle_x is not None and
            camera_distance is not None and
            mesh_scale is not None
        )
        
        # build camera
        yaw = [0, np.pi/2, np.pi, 3*np.pi/2]
        yaw_offset = -16 / 180 * np.pi
        yaw = [y + yaw_offset for y in yaw]
        pitch = [20 / 180 * np.pi for _ in range(4)]
        exts, ints = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, 2, 30)
        
        # render
        renderer = PbrMeshRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.near = 1
        renderer.rendering_options.far = 100
        renderer.rendering_options.ssaa = 2
        renderer.rendering_options.peel_layers = 8
        envmap = EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device='cuda'
        ))
        
        images = {}
        gt_view_images = {}
        for i, representation in enumerate(reps):
            # Validate mesh data before rasterization (same as shape training)
            verts = representation.vertices
            faces = representation.faces
            if verts.shape[0] == 0 or faces.shape[0] == 0:
                print(f"[visualize_sample] Warning: sample {i} has empty mesh, skipping")
                continue
            if faces.max() >= verts.shape[0]:
                print(f"[visualize_sample] Warning: sample {i} has out-of-bound face indices "
                      f"(max face idx={faces.max().item()}, num verts={verts.shape[0]}), skipping")
                continue
            if torch.isnan(verts).any() or torch.isinf(verts).any():
                print(f"[visualize_sample] Warning: sample {i} has NaN/Inf vertices, skipping")
                continue

            image = {}
            tile = [2, 2]
            try:
                for j, (ext, intr) in enumerate(zip(exts, ints)):
                    res = renderer.render(representation, ext, intr, envmap=envmap)
                    for k, v in res.items():
                        if k not in images:
                            images[k] = []
                        if k not in image:
                            image[k] = torch.zeros(3, 1024, 1024).cuda()  
                        image[k][:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = v
                for k in images.keys():
                    images[k].append(image[k])
            except RuntimeError as e:
                print(f"[visualize_sample] Warning: render failed for sample {i}: {e}")
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue
            
            # Render GT camera view
            # Must scale mesh vertices by / mesh_scale to match ProjGrid's projection space.
            # ProjGrid maps [-1,1]^3 -> / scale / 2 -> [-0.5/s, 0.5/s]^3
            # Mesh vertices in [-0.5, 0.5]^3 -> / scale -> [-0.5/s, 0.5/s]^3 (equivalent)
            if has_gt_camera:
                try:
                    scale = mesh_scale[i].item()
                    distance = camera_distance[i].item()
                    fov = camera_angle_x[i].item()
                    device = representation.vertices.device
                    
                    # Scale mesh and voxel to match ProjGrid's projection space
                    scaled_rep = MeshWithVoxel(
                        vertices=representation.vertices / scale,
                        faces=representation.faces,
                        origin=(representation.origin / scale).tolist(),
                        voxel_size=representation.voxel_size / scale,
                        coords=representation.coords,
                        attrs=representation.attrs,
                        voxel_shape=representation.voxel_shape,
                        layout=representation.layout,
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
                    
                    # Update near/far for the smaller scaled mesh
                    mesh_half_size = 0.5 / scale
                    renderer.rendering_options.near = max(0.01, distance - mesh_half_size - 0.5)
                    renderer.rendering_options.far = distance + mesh_half_size + 0.5
                    
                    gt_res = renderer.render(scaled_rep, gt_ext, gt_int, envmap=envmap)
                    for k, v in gt_res.items():
                        gt_key = f'gt_view_{k}'
                        if gt_key not in gt_view_images:
                            gt_view_images[gt_key] = []
                        gt_view_images[gt_key].append(v)
                except RuntimeError as e:
                    print(f"[visualize_sample] Warning: GT view render failed for sample {i}: {e}")
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
        
        for k in images.keys():
            images[k] = torch.stack(images[k], dim=0)
        
        for k, v in gt_view_images.items():
            images[k] = torch.stack(v)
        
        return images
    
    
class SLatPbr(SLatPbrVisMixin, StandardDatasetBase):
    """
    structured latent for sparse voxel pbr dataset
    
    Args:
        roots (str): path to the dataset
        latent_key (str): key of the latent to be used
        min_aesthetic_score (float): minimum aesthetic score
        normalization (dict): normalization stats
        resolution (int): resolution of decoded sparse voxel
        attrs (list): attributes to be decoded
        pretained_slat_dec (str): name of the pretrained slat decoder
        slat_dec_path (str): path to the slat decoder, if given, will override the pretrained_slat_dec
        slat_dec_ckpt (str): name of the slat decoder checkpoint
    """
    def __init__(self,
        roots: str,
        *,
        resolution: int,
        min_aesthetic_score: float = 5.0,
        max_tokens: int = 32768,
        full_pbr: bool = False,
        pbr_slat_normalization: Optional[dict] = None,
        shape_slat_normalization: Optional[dict] = None,
        attrs: list[str] = ['base_color', 'metallic', 'roughness', 'emissive', 'alpha'],
        pretrained_pbr_slat_dec: str = 'JeffreyXiang/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16',
        pbr_slat_dec_path: Optional[str] = None,
        pbr_slat_dec_ckpt: Optional[str] = None,
        pretrained_shape_slat_dec: str = 'JeffreyXiang/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16',
        shape_slat_dec_path: Optional[str] = None,
        shape_slat_dec_ckpt: Optional[str] = None,
        **kwargs
    ):  
        self.resolution = resolution
        self.pbr_slat_normalization = pbr_slat_normalization
        self.shape_slat_normalization = shape_slat_normalization
        self.min_aesthetic_score = min_aesthetic_score
        self.max_tokens = max_tokens
        self.full_pbr = full_pbr
        self.value_range = (0, 1)
        
        super().__init__(
            roots,
            pretrained_pbr_slat_dec=pretrained_pbr_slat_dec,
            pbr_slat_dec_path=pbr_slat_dec_path,
            pbr_slat_dec_ckpt=pbr_slat_dec_ckpt,
            pretrained_shape_slat_dec=pretrained_shape_slat_dec,
            shape_slat_dec_path=shape_slat_dec_path,
            shape_slat_dec_ckpt=shape_slat_dec_ckpt,
            **kwargs
        )
        
        self.loads = [self.metadata.loc[sha256, 'pbr_latent_tokens'] for _, sha256, _ in self.instances]
        
        if self.pbr_slat_normalization is not None:
            self.pbr_slat_mean = torch.tensor(self.pbr_slat_normalization['mean']).reshape(1, -1)
            self.pbr_slat_std = torch.tensor(self.pbr_slat_normalization['std']).reshape(1, -1)
        
        if self.shape_slat_normalization is not None:
            self.shape_slat_mean = torch.tensor(self.shape_slat_normalization['mean']).reshape(1, -1)
            self.shape_slat_std = torch.tensor(self.shape_slat_normalization['std']).reshape(1, -1)
        
        self.attrs = attrs
        self.channels = {
            'base_color': 3,
            'metallic': 1,
            'roughness': 1,
            'emissive': 3,
            'alpha': 1,
        }
        self.layout = {}
        start = 0
        for attr in attrs:
            self.layout[attr] = slice(start, start + self.channels[attr])
            start += self.channels[attr]
            
    def filter_metadata(self, metadata, dataset_name=None):
        stats = {}
        metadata = metadata[metadata['pbr_latent_encoded'] == True]
        stats['With PBR latent'] = len(metadata)
        metadata = metadata[metadata['shape_latent_encoded'] == True]
        stats['With shape latent'] = len(metadata)
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        metadata = metadata[metadata['pbr_latent_tokens'] <= self.max_tokens]
        stats[f'Num tokens <= {self.max_tokens}'] = len(metadata)
        if self.full_pbr:
            metadata = metadata[metadata['num_basecolor_tex'] > 0]
            metadata = metadata[metadata['num_metallic_tex'] > 0]
            metadata = metadata[metadata['num_roughness_tex'] > 0]
            stats['Full PBR'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        # PBR latent
        data = np.load(os.path.join(root['pbr_latent'], f'{instance}.npz'))
        coords = torch.tensor(data['coords']).int()
        coords = torch.cat([torch.zeros_like(coords)[:, :1], coords], dim=1)
        feats = torch.tensor(data['feats']).float()
        if self.pbr_slat_normalization is not None:
            feats = (feats - self.pbr_slat_mean) / self.pbr_slat_std
        pbr_z = SparseTensor(feats, coords)
        
        # Shape latent
        data = np.load(os.path.join(root['shape_latent'], f'{instance}.npz'))
        coords = torch.tensor(data['coords']).int()
        coords = torch.cat([torch.zeros_like(coords)[:, :1], coords], dim=1)
        feats = torch.tensor(data['feats']).float()
        if self.shape_slat_normalization is not None:
            feats = (feats - self.shape_slat_mean) / self.shape_slat_std
        shape_z = SparseTensor(feats, coords)
        
        assert torch.equal(shape_z.coords, pbr_z.coords), \
            f"Shape latent and PBR latent have different coordinates: {shape_z.coords.shape} vs {pbr_z.coords.shape}"
            
        return {
            'x_0': pbr_z,
            'concat_cond': shape_z,
        }
        
    @staticmethod
    def collate_fn(batch, split_size=None):
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            group_idx = load_balanced_group_indices([b['x_0'].feats.shape[0] for b in batch], split_size)
        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            pack = {}

            keys = [k for k in sub_batch[0].keys()]
            for k in keys:
                if isinstance(sub_batch[0][k], torch.Tensor):
                    pack[k] = torch.stack([b[k] for b in sub_batch])
                elif isinstance(sub_batch[0][k], SparseTensor):
                    pack[k] = sparse_cat([b[k] for b in sub_batch], dim=0)
                elif isinstance(sub_batch[0][k], list):
                    pack[k] = sum([b[k] for b in sub_batch], [])
                else:
                    pack[k] = [b[k] for b in sub_batch]
            
            packs.append(pack)
        
        if split_size is None:
            return packs[0]
        return packs


class ImageConditionedSLatPbr(ImageConditionedMixin, SLatPbr):
    """
    Image conditioned structured latent dataset
    """
    pass


class SLatPbrView(SLatPbrVisMixin, StandardDatasetBase):
    """
    View-based structured latent for PBR/texture generation with view-aligned projection.
    
    Data format: 
        PBR latent:   {sha256}/view{XX}.npz  (coords + feats)
        Shape latent: {sha256}/view{XX}.npz  (coords + feats, from shape_latent_view dir)
    
    Each view's PBR latent and Shape latent share the same sparse coordinates.
    
    Args:
        roots (str): path to the dataset
        resolution (int): resolution of decoded sparse voxel
        min_aesthetic_score (float): minimum aesthetic score
        max_tokens (int): maximum number of tokens
        num_views (int): Number of views to use (0 to num_views-1). Default is 2.
        full_pbr (bool): Whether to require full PBR textures
        pbr_slat_normalization (dict): normalization stats for PBR latent
        shape_slat_normalization (dict): normalization stats for shape latent
        attrs (list): PBR attributes to decode
        pretrained_pbr_slat_dec (str): pretrained PBR decoder name
        pretrained_shape_slat_dec (str): pretrained shape decoder name
        skip_list (str, optional): path to a file containing sha256 hashes to skip
        skip_aesthetic_score_datasets (list, optional): datasets to skip aesthetic score check
    """
    def __init__(self,
        roots: str,
        *,
        resolution: int,
        min_aesthetic_score: float = 5.0,
        max_tokens: int = 32768,
        num_views: int = 2,
        full_pbr: bool = False,
        pbr_slat_normalization: Optional[dict] = None,
        shape_slat_normalization: Optional[dict] = None,
        attrs: list[str] = ['base_color', 'metallic', 'roughness', 'emissive', 'alpha'],
        pretrained_pbr_slat_dec: str = 'microsoft/TRELLIS.2-4B/ckpts/tex_dec_next_dc_f16c32_fp16',
        pbr_slat_dec_path: Optional[str] = None,
        pbr_slat_dec_ckpt: Optional[str] = None,
        pretrained_shape_slat_dec: str = 'microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16',
        shape_slat_dec_path: Optional[str] = None,
        shape_slat_dec_ckpt: Optional[str] = None,
        skip_list: Optional[str] = None,
        skip_aesthetic_score_datasets: Optional[list] = None,
    ):
        self.resolution = resolution
        self.pbr_slat_normalization = pbr_slat_normalization
        self.shape_slat_normalization = shape_slat_normalization
        self.min_aesthetic_score = min_aesthetic_score
        self.max_tokens = max_tokens
        self.num_views = num_views
        self.full_pbr = full_pbr
        self.value_range = (0, 1)
        self.skip_aesthetic_score_datasets = set(skip_aesthetic_score_datasets or [])
        
        # Initialize visualization mixin
        SLatPbrVisMixin.__init__(
            self,
            roots,
            pretrained_pbr_slat_dec=pretrained_pbr_slat_dec,
            pbr_slat_dec_path=pbr_slat_dec_path,
            pbr_slat_dec_ckpt=pbr_slat_dec_ckpt,
            pretrained_shape_slat_dec=pretrained_shape_slat_dec,
            shape_slat_dec_path=shape_slat_dec_path,
            shape_slat_dec_ckpt=shape_slat_dec_ckpt,
        )
        StandardDatasetBase.__init__(
            self, roots,
            skip_list=skip_list,
            skip_aesthetic_score_datasets=skip_aesthetic_score_datasets,
        )
        
        # Calculate loads for load balancing
        self.loads = []
        for _, sha256, _ in self.instances:
            if 'pbr_latent_tokens' in self.metadata.columns:
                try:
                    self.loads.append(self.metadata.loc[sha256, 'pbr_latent_tokens'])
                except:
                    self.loads.append(self.max_tokens)
            else:
                self.loads.append(self.max_tokens)
        
        if self.pbr_slat_normalization is not None:
            self.pbr_slat_mean = torch.tensor(self.pbr_slat_normalization['mean']).reshape(1, -1)
            self.pbr_slat_std = torch.tensor(self.pbr_slat_normalization['std']).reshape(1, -1)
        
        if self.shape_slat_normalization is not None:
            self.shape_slat_mean = torch.tensor(self.shape_slat_normalization['mean']).reshape(1, -1)
            self.shape_slat_std = torch.tensor(self.shape_slat_normalization['std']).reshape(1, -1)
        
        self.attrs = attrs
        self.channels = {
            'base_color': 3,
            'metallic': 1,
            'roughness': 1,
            'emissive': 3,
            'alpha': 1,
        }
        self.layout = {}
        start = 0
        for attr in attrs:
            self.layout[attr] = slice(start, start + self.channels[attr])
            start += self.channels[attr]

    def filter_metadata(self, metadata, dataset_name=None):
        stats = {}
        # View-based PBR latent uses columns like pbr_latent_view00_encoded, etc.
        required_pbr_view_cols = [f'pbr_latent_view{i:02d}_encoded' for i in range(self.num_views)]
        existing_pbr_view_cols = [col for col in required_pbr_view_cols if col in metadata.columns]
        
        if existing_pbr_view_cols:
            has_all_pbr_views = (metadata[existing_pbr_view_cols] == True).all(axis=1)
            metadata = metadata[has_all_pbr_views]
            stats[f'With {self.num_views} PBR view latents'] = len(metadata)
        else:
            # Fallback: check pbr_latent_encoded
            if 'pbr_latent_encoded' in metadata.columns:
                metadata = metadata[metadata['pbr_latent_encoded'] == True]
                stats['With PBR latent'] = len(metadata)
        
        # Also require shape latent views
        required_shape_view_cols = [f'shape_latent_view{i:02d}_encoded' for i in range(self.num_views)]
        existing_shape_view_cols = [col for col in required_shape_view_cols if col in metadata.columns]
        
        if existing_shape_view_cols:
            has_all_shape_views = (metadata[existing_shape_view_cols] == True).all(axis=1)
            metadata = metadata[has_all_shape_views]
            stats[f'With {self.num_views} shape view latents'] = len(metadata)
        else:
            if 'shape_latent_encoded' in metadata.columns:
                metadata = metadata[metadata['shape_latent_encoded'] == True]
                stats['With shape latent'] = len(metadata)
        
        # Skip aesthetic score check for specified datasets
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
        if 'pbr_latent_tokens' in metadata.columns:
            metadata = metadata[metadata['pbr_latent_tokens'] <= self.max_tokens]
            stats[f'Num tokens <= {self.max_tokens}'] = len(metadata)
        
        if self.full_pbr:
            if 'num_basecolor_tex' in metadata.columns:
                metadata = metadata[metadata['num_basecolor_tex'] > 0]
            if 'num_metallic_tex' in metadata.columns:
                metadata = metadata[metadata['num_metallic_tex'] > 0]
            if 'num_roughness_tex' in metadata.columns:
                metadata = metadata[metadata['num_roughness_tex'] > 0]
            stats['Full PBR'] = len(metadata)
        
        return metadata, stats

    def get_instance(self, root, instance):
        # Randomly select a view from the configured range
        view_idx = np.random.randint(0, self.num_views)
        view_file = f'view{view_idx:02d}.npz'
        
        # Store view info for ViewImageConditionedMixin
        self._current_view_idx = view_idx
        
        # Load PBR latent for this view
        pbr_latent_dir = os.path.join(root['pbr_latent'], instance)
        self._current_latent_dir = pbr_latent_dir
        
        data = np.load(os.path.join(pbr_latent_dir, view_file))
        pbr_coords = torch.tensor(data['coords']).int()
        pbr_feats = torch.tensor(data['feats']).float()
        if self.pbr_slat_normalization is not None:
            pbr_feats = (pbr_feats - self.pbr_slat_mean) / self.pbr_slat_std
        
        # Load Shape latent for this view (as concat_cond)
        shape_latent_dir = os.path.join(root['shape_latent'], instance)
        data = np.load(os.path.join(shape_latent_dir, view_file))
        shape_coords = torch.tensor(data['coords']).int()
        shape_feats = torch.tensor(data['feats']).float()
        if self.shape_slat_normalization is not None:
            shape_feats = (shape_feats - self.shape_slat_mean) / self.shape_slat_std
        
        # Verify coordinates match
        assert torch.equal(pbr_coords, shape_coords), \
            f"PBR and shape latent coordinates mismatch for {instance}/view{view_idx:02d}"
        
        return {
            'coords': pbr_coords,
            'pbr_feats': pbr_feats,
            'shape_feats': shape_feats,
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
            
            # Build x_0 (PBR latent) and concat_cond (shape latent) as SparseTensors
            coords_list = []
            pbr_feats_list = []
            shape_feats_list = []
            layout = []
            start = 0
            for i, b in enumerate(sub_batch):
                batch_coords = torch.cat([
                    torch.full((b['coords'].shape[0], 1), i, dtype=torch.int32),
                    b['coords']
                ], dim=-1)
                coords_list.append(batch_coords)
                pbr_feats_list.append(b['pbr_feats'])
                shape_feats_list.append(b['shape_feats'])
                layout.append(slice(start, start + b['coords'].shape[0]))
                start += b['coords'].shape[0]
            
            all_coords = torch.cat(coords_list)
            
            # x_0: PBR latent
            pack['x_0'] = SparseTensor(
                coords=all_coords,
                feats=torch.cat(pbr_feats_list),
            )
            pack['x_0']._shape = torch.Size([len(group), *sub_batch[0]['pbr_feats'].shape[1:]])
            pack['x_0'].register_spatial_cache('layout', layout)
            
            # concat_cond: Shape latent (same coordinates)
            pack['concat_cond'] = SparseTensor(
                coords=all_coords.clone(),
                feats=torch.cat(shape_feats_list),
            )
            pack['concat_cond']._shape = torch.Size([len(group), *sub_batch[0]['shape_feats'].shape[1:]])
            pack['concat_cond'].register_spatial_cache('layout', layout)
            
            # collate other data (excluding already handled fields)
            skip_keys = {'coords', 'pbr_feats', 'shape_feats'}
            keys = [k for k in sub_batch[0].keys() if k not in skip_keys]
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


class ViewImageConditionedSLatPbrView(ViewImageConditionedMixin, SLatPbrView):
    """
    Image-conditioned view-based structured latent for PBR/texture generation
    with view-aligned projection.
    
    Loads PBR latent and shape latent from {sha256}/view{XX}.npz format and pairs
    with corresponding view from render_cond.
    
    Uses ViewImageConditionedMixin which reads mesh_scale from view{XX}_scale.json
    and provides camera parameters for 3D-to-2D projection.
    """
    pass
