from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def _rasterize_uv(vertices, faces, uvs, texture_size, device):
    import drtk

    size = int(texture_size)
    verts_uv = torch.stack(
        [
            uvs[:, 0] * size - 0.5,
            uvs[:, 1] * size - 0.5,
            torch.ones(uvs.shape[0], device=device),
        ],
        dim=-1,
    ).float().unsqueeze(0)

    rast_face_ids = torch.full((size, size), -1, dtype=torch.int32, device=device)
    chunk_size = 100_000
    for start in range(0, faces.shape[0], chunk_size):
        chunk_faces = faces[start:start + chunk_size].int()
        index_img = drtk.rasterize(verts_uv, chunk_faces, height=size, width=size)
        hit = index_img[0] >= 0
        rast_face_ids[hit] = (index_img[0][hit] + start).int()

    mask = rast_face_ids >= 0
    _, bary_img = drtk.render(verts_uv, faces.int(), rast_face_ids.unsqueeze(0))
    bary = bary_img[0].permute(1, 2, 0)
    face_ids = rast_face_ids[mask].long()
    face_verts = vertices[faces[face_ids].long()]
    valid_pos = (face_verts * bary[mask].unsqueeze(-1)).sum(dim=1)
    return mask, valid_pos


def _inpaint_channel(channel, mask_inv, radius=1):
    import cv2

    out = cv2.inpaint(channel, mask_inv, radius, cv2.INPAINT_TELEA)
    if out.ndim == 2:
        out = out[..., None]
    return out


def to_glb(
    vertices,
    faces,
    attr_volume,
    coords,
    attr_layout,
    grid_size=None,
    aabb=((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)),
    decimation_target=1000000,
    texture_size=4096,
    remesh=True,
    remesh_band=1,
    remesh_project=0.9,
    mesh_cluster_threshold_cone_half_angle_rad=np.radians(90.0),
    mesh_cluster_refine_iterations=0,
    mesh_cluster_global_iterations=1,
    mesh_cluster_smooth_strength=1,
    verbose=False,
    use_tqdm=True,
    **_,
):
    import cv2
    import trimesh
    try:
        import cumesh_vb as cumesh
        from cumesh_vb.bvh import cuBVH
    except Exception:
        import cumesh
        from cumesh.bvh import cuBVH
    try:
        from flex_gemm_ap.ops.grid_sample import grid_sample_3d
    except Exception:
        from flex_gemm.ops.grid_sample import grid_sample_3d

    device = torch.device("cuda")
    vertices = torch.as_tensor(vertices, dtype=torch.float32, device=device)
    faces = torch.as_tensor(faces, dtype=torch.int32, device=device)
    attr_volume = torch.as_tensor(attr_volume, dtype=torch.float32, device=device)
    coords = torch.as_tensor(coords, dtype=torch.float32, device=device)

    if isinstance(grid_size, int):
        grid_size = [grid_size, grid_size, grid_size]
    if grid_size is None:
        grid_size = [int(coords[:, i].max().item()) + 1 for i in range(3)]

    aabb = torch.tensor(aabb, dtype=torch.float32, device=device)
    voxel_size = (aabb[1] - aabb[0]) / torch.tensor(grid_size, dtype=torch.float32, device=device)

    progress_verbose = bool(use_tqdm or verbose)

    mesh = cumesh.CuMesh()
    mesh.init(vertices, faces)
    mesh.fill_holes(max_hole_perimeter=3e-2)
    orig_vertices, orig_faces = mesh.read()
    bvh = cuBVH(orig_vertices, orig_faces)

    target_faces = int(decimation_target) if decimation_target else 0
    if remesh:
        center = aabb.mean(dim=0)
        scale = (aabb[1] - aabb[0]).max().item()
        resolution = int(torch.as_tensor(grid_size, device=device).max().item())
        try:
            mesh.init(*cumesh.remeshing.remesh_narrow_band_dc(
                orig_vertices,
                orig_faces,
                center=center,
                scale=(resolution + 3 * remesh_band) / resolution * scale,
                resolution=resolution,
                band=remesh_band,
                project_back=remesh_project,
                verbose=progress_verbose,
                bvh=bvh,
            ))
        except Exception as exc:
            if verbose:
                print(f"Remesh failed ({type(exc).__name__}: {exc}); falling back to cleanup path.")
            remesh = False

    if target_faces > 0:
        if not remesh:
            if mesh.num_faces > target_faces * 3:
                mesh.simplify(target_faces * 3, verbose=progress_verbose)
            mesh.remove_duplicate_faces()
            mesh.repair_non_manifold_edges()
            mesh.remove_small_connected_components(1e-5)
            mesh.fill_holes(max_hole_perimeter=3e-2)
        if mesh.num_faces > target_faces:
            mesh.simplify(target_faces, verbose=progress_verbose)

    mesh.remove_duplicate_faces()
    mesh.repair_non_manifold_edges()
    mesh.remove_small_connected_components(1e-5)
    mesh.fill_holes(max_hole_perimeter=3e-2)
    if not remesh:
        mesh.unify_face_orientations()

    out_vertices, out_faces, out_uvs, out_vmaps = mesh.uv_unwrap(
        compute_charts_kwargs={
            "threshold_cone_half_angle_rad": mesh_cluster_threshold_cone_half_angle_rad,
            "refine_iterations": mesh_cluster_refine_iterations,
            "global_iterations": mesh_cluster_global_iterations,
            "smooth_strength": mesh_cluster_smooth_strength,
        },
        return_vmaps=True,
        verbose=progress_verbose,
    )
    out_vertices = out_vertices.to(device)
    out_faces = out_faces.to(device)
    out_uvs = out_uvs.to(device)
    out_vmaps = out_vmaps.to(device)
    mesh.compute_vertex_normals()
    out_normals = mesh.read_vertex_normals().to(device)[out_vmaps]

    mask, valid_pos = _rasterize_uv(out_vertices, out_faces, out_uvs, texture_size, device)
    _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
    orig_tri_verts = orig_vertices[orig_faces[face_id.long()]]
    valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)

    attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device=device)
    attrs[mask] = grid_sample_3d(
        attr_volume,
        torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
        shape=torch.Size([1, attr_volume.shape[1], *grid_size]),
        grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
        mode="trilinear",
    )

    mask_inv = (~mask.cpu().numpy()).astype(np.uint8)
    base_color = np.clip(attrs[..., attr_layout["base_color"]].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    metallic = np.clip(attrs[..., attr_layout["metallic"]].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    roughness = np.clip(attrs[..., attr_layout["roughness"]].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    alpha = np.clip(attrs[..., attr_layout["alpha"]].cpu().numpy() * 255, 0, 255).astype(np.uint8)

    base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
    metallic = _inpaint_channel(metallic, mask_inv)
    roughness = _inpaint_channel(roughness, mask_inv)
    alpha = _inpaint_channel(alpha, mask_inv)

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(
            np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)
        ),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode="OPAQUE",
        doubleSided=True if not remesh else False,
    )

    vertices_np = out_vertices.cpu().numpy()
    faces_np = out_faces.cpu().numpy()
    uvs_np = out_uvs.cpu().numpy()
    normals_np = out_normals.cpu().numpy()

    vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2].copy(), -vertices_np[:, 1].copy()
    normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2].copy(), -normals_np[:, 1].copy()
    uvs_np[:, 1] = 1 - uvs_np[:, 1]

    return trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np,
        vertex_normals=normals_np,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=uvs_np, material=material),
    )
