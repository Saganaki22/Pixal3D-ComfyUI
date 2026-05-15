# Related Repo Findings

This is the short version of what is useful from the nearby Pixal3D/Trellis work.

## Rizzlord/ComfyUI-Pixal3D-D

Repo: https://github.com/Rizzlord/ComfyUI-Pixal3D-D

This wraps **Pixal3D-D**, not the same TencentARC/Pixal3D image-to-3D pipeline used by this nodepack. The model layout, stages, and checkpoints are different.

Useful ideas copied into this nodepack:

| Idea | What Rizzlord does | What Pixal3D-ComfyUI does |
|---|---|---|
| Manual model folders | Expects `ComfyUI/models/pixal3d-d/dense`, `sparse512`, `sparse1024` | Expects `ComfyUI/models/Pixal3D/<owner_repo>/` clean snapshots |
| NAF low VRAM control | `low_vram` disables optional NAF and `upsample_res` lowers NAF resolution | `naf_mode=fallback_if_missing` works without NATTEN, and `naf_target_size` can lower real NAF resolution |
| Model retention | `keep_model_loaded` controls whether models stay resident | `vram_mode`, `force_offload`, and `Pixal3D Unload Model` control residency |
| Mesh cleanup | Uses `meshlib` for decimation/floater/interior cleanup | GLB export uses o_voxel/CuMesh cleanup; `remesh` is controlled by the export node |

The main warning from this repo is that “Pixal3D” and “Pixal3D-D” are not interchangeable. Do not mix their model folders or weights.

## visualbruno/ComfyUI-Trellis2 pixal3d branch

Repo: https://github.com/visualbruno/ComfyUI-Trellis2/tree/pixal3d

This is a TRELLIS.2 wrapper that added `TencentARC/Pixal3D-T` support. It is not the same node architecture, but its wheel folder is useful.

Useful findings:

| Area | Finding |
|---|---|
| Pixal3D-T | README says `TencentARC/Pixal3D-T` requires `natten`. |
| DINOv3 | README expects gated `facebook/dinov3-vitl16-pretrain-lvd1689m` under `ComfyUI/models/facebook/...`. This nodepack uses `camenduru/dinov3-vitl16-pretrain-lvd1689m` under `ComfyUI/models/Pixal3D/`. |
| Wheels | The branch includes Windows wheels for `cumesh`, `flex_gemm`, `o_voxel`, `nvdiffrast`, `nvdiffrec_render`, and some `natten` builds. |
| Names | Its wheel modules use generic names like `cumesh`, `flex_gemm`, and `o_voxel`; this nodepack also accepts the `_vb`/`_ap` names from Pozzetti wheels. |
| Blackwell | The branch includes a `Voxel to Mesh` node based on ThatButters' workaround. |

Useful wheel folders:

| Folder | Notes |
|---|---|
| `wheels/Windows/Torch2100` | Has `cumesh`, `flex_gemm`, `o_voxel`, `nvdiffrast` for cp311/cp312. No cp312 `natten` in this folder. |
| `wheels/Windows/Torch2100/CUDA 13.1` | Has cp311/cp312/cp313 CUDA wheels, but `natten-0.21.6` is cp313 only. |
| `wheels/Windows/Torch280` | Has `natten-0.21.6-cp312-cp312-win_amd64.whl`, useful only for Python 3.12 + Torch 2.8-era stacks. |

For the local stack `Python 3.12 + Torch 2.10.0+cu130`, visualbruno's branch is useful for several extension wheels, but it does not provide the missing native Windows `natten` wheel.

## ThatButters/trellis2-blackwell-fix

Repo: https://github.com/ThatButters/trellis2-blackwell-fix

This is the important remesh note. The repo documents that `o_voxel.postprocess.to_glb()` uses CuMesh remeshing and that `cumesh.remeshing.remesh_narrow_band_dc` can silently fail on some GPU/driver/wheel combinations. It does not necessarily crash; it can export a fragmented mesh that looks like a loose point cloud.

What was added here:

| Finding | Nodepack response |
|---|---|
| Some GPU/driver/wheel combinations can break CuMesh remesh | The export node exposes `remesh` so users can turn the o_voxel remesh path on or off directly. |
| Over-aggressive decimation can shred detailed assets | Export defaults now match upstream more closely: `decimation_target=1000000`, `texture_size=4096`. |
| CPU voxel fallback works for TRELLIS.2 mesh objects with voxel coords | Not added as a full Pixal3D GLB replacement yet; Pixal3D's textured GLB path still needs attr-volume texture baking. |

If you see shredded GLBs or thousands of disconnected pieces, try `remesh=false` and avoid low decimation targets like `200000`.
