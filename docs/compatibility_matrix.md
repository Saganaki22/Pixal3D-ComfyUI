# Pixal3D-ComfyUI Compatibility Matrix

Pixal3D-ComfyUI does not pin one global Python, PyTorch, or CUDA version. It works when the compiled CUDA extension wheels match the Python, PyTorch, CUDA, and platform used by ComfyUI.

Use **Pixal3D Environment Check** inside ComfyUI after installing wheels.

## Verified Example Stack

| OS | Python | PyTorch | CUDA runtime | GPU | Attention | Status |
|---|---:|---:|---:|---|---|---|
| Windows | 3.12.x | 2.10.0+cu130 | 13.0 | NVIDIA CUDA GPU | FlashAttention 2.8.3 cu130 torch2.10 | Required imports pass when matching CUDA wheels are installed; use `naf_mode=fallback_if_missing` unless `natten.HAS_LIBNATTEN=True` |

Required Pixal3D CUDA module APIs for this stack:

```text
flex_gemm_ap or flex_gemm
cumesh_vb or cumesh
o_voxel_vb_ap or o_voxel
drtk
```

Attention module requirement:

```text
flash_attn or flash_attn_interface
```

## Compatibility Rules

| Component | Requirement |
|---|---|
| Python | Must match wheel tag, for example `cp312` for Python 3.12 |
| PyTorch | Must match wheel build tag, for example `torch2.10` |
| CUDA runtime | Must match wheel build tag, for example `cu130` |
| Platform | Windows needs `win_amd64`; Linux needs matching Linux wheels or source builds |
| GPU | CUDA-capable NVIDIA GPU with matching compiled wheels |

## Recommended Settings By Platform

Pixal3D-ComfyUI is not tied to a specific folder. It uses whichever Python environment launches ComfyUI. The important part is that every compiled wheel imports inside that same environment.

For Windows Python 3.12 + PyTorch 2.10 + CUDA 13.0, use this practical setup:

| Area | Setting |
|---|---|
| Attention | `flash_attn` 2.8.3 cu130 torch2.10, or another matching FlashAttention wheel |
| NAF | `naf_mode=fallback_if_missing` |
| NAF target | `upstream` unless strict NAF is installed; lower values only matter for real NAF |
| Export | `decimation_target=1000000`, `texture_size=4096`, `remesh=true` by default |

For Linux or WSL, install official NATTEN/libnatten wheels when your PyTorch/CUDA stack is listed by NATTEN. That is the recommended path for exact upstream NAF behavior.

## Helper Models

These are model files, not Python packages:

| Helper | Default location | Notes |
|---|---|---|
| Pixal3D | `ComfyUI/models/Pixal3D/TencentARC_Pixal3D/` | Main model, includes `ckpts/*.safetensors` |
| DINOv3 | `ComfyUI/models/Pixal3D/camenduru_dinov3-vitl16-pretrain-lvd1689m/` | Needs `model.safetensors` |
| MoGe | `ComfyUI/models/geometry_estimation/` | Native ComfyUI MoGe from `https://huggingface.co/Comfy-Org/MoGe`; Pixal3D-ComfyUI uses `moge_2_vitl_normal_fp16.safetensors` |
| RMBG-2.0 | `ComfyUI/models/Pixal3D/briaai_RMBG-2.0/` | Gated Hugging Face model; needed for `background_mode=auto_remove` |

Preferred clean folder names are `owner_repo`. The Pixal3D/RMBG/DINO helpers also check common manual-download names like `Pixal3D`, `RMBG-2.0`, and Hugging Face cache-style folders like `models--owner--repo/snapshots/<hash>/`.

Model folders may be normal directories, Windows junctions, or symlinks. The linked target must contain the normal model files; broken links or blob-only Hugging Face cache folders are treated as missing/incomplete models.

Native ComfyUI MoGe files belong directly under `ComfyUI/models/geometry_estimation/`:

```text
moge_1_vitl_fp16.safetensors
moge_2_vitl_normal_fp16.safetensors
```

If `download_if_missing=true`, Pixal3D-ComfyUI downloads missing files from `https://huggingface.co/Comfy-Org/MoGe` into `ComfyUI/models/geometry_estimation/`. It uses `moge_2_vitl_normal_fp16.safetensors` for `camera_mode=moge` and does not load a `Ruicheng/moge-2-vitl` snapshot folder.

Pixal3D-ComfyUI removes Hugging Face `.cache` metadata and `.git` folders after node downloads. Native MoGe files are placed directly in `ComfyUI/models/geometry_estimation/`; Pixal3D/DINO/RMBG helper folders should contain normal files like `.safetensors`, `.json`, and `.py`, not blob-only cache paths.

Torch Hub helper code for Pixal3D's NAF upsampler is redirected to `ComfyUI/models/Pixal3D/torch_hub/`. If a user pre-downloaded `valeoai/NAF` before ComfyUI redirected Torch Hub, Pixal3D-ComfyUI also checks the normal user cache at `~/.cache/torch/hub`.

## NAF And NATTEN

Official TencentARC Pixal3D uses NAF for the shape and texture stages. The released shape/texture weights expect 2048-channel projected features, which normally come from DINOv3 features concatenated with NAF-upsampled features.

On Windows stacks without a matching CUDA NATTEN build, use `naf_mode=fallback_if_missing`. Pixal3D-ComfyUI keeps the 2048-channel shape by duplicating DINO projection features and avoids downloading NAF when `download_if_missing=false`.

For exact upstream NAF behavior, install a CUDA-enabled NATTEN wheel matching Python, PyTorch, CUDA, and GPU architecture, then use `naf_mode=strict`.

For Windows Python 3.12 + PyTorch 2.10 + CUDA 13.0, the official `natten==0.21.6+torch2100cu130` index currently exposes Linux wheels, not `win_amd64` wheels. Community Windows wheels exist for some GPU architectures — see [NeilsMabet/Natten-0.21.6-Amphere-wheel-windows](https://github.com/NeilsMabet/Natten-0.21.6-Amphere-wheel-windows) for Ampere sm86 CUDA 13.0 builds for Python 3.12.10 and 3.13.12, [drbaph/NATTEN-0.21.6-torch2100cu130-cp312-cp312-win_amd64](https://huggingface.co/drbaph/NATTEN-0.21.6-torch2100cu130-cp312-cp312-win_amd64) for a Blackwell sm120 CUDA 13.0 build, or [naxneri/natten-0.21.6-blackwell-cu128-cp312-cp312-win_amd64](https://huggingface.co/naxneri/natten-0.21.6-blackwell-cu128-cp312-cp312-win_amd64) for a Blackwell CUDA 12.8 build. visualbruno's Pixal3D branch has a Windows `natten-0.21.6` wheel for Python 3.12 + Torch 2.8 and a Torch 2.10 CUDA 13.1 NATTEN wheel for Python 3.13, but not Python 3.12 + Torch 2.10. Pixal3D-ComfyUI installs plain `natten==0.21.6` as a baseline dependency, but if `natten.HAS_LIBNATTEN` is `False`, strict NAF is still unavailable.

`naf_target_size` only affects strict/real NAF:

| Value | Effect |
|---|---|
| `upstream` | Shape stages use 512, texture stage uses 1024 |
| `1024` | Forces all NAF stages to 1024; highest VRAM |
| `512` | Keeps texture NAF lower than upstream; useful if strict NAF works but VRAM is tight |
| `256` / `128` | More aggressive VRAM reduction, more likely to reduce detail |

## Remesh Export Notes

Pixal3D-ComfyUI uses `o_voxel.postprocess.to_glb()` for textured GLB export. The `remesh` setting is passed through exactly as requested by the workflow.

| Export setting | Recommendation |
|---|---|
| `remesh=false` | Skips the o_voxel remesh pass |
| `remesh=true` | Default node value; uses the full o_voxel remesh path |

If an exported model looks like loose shards or a point cloud, use `decimation_target=1000000` or higher and try `remesh=true`.

## Conditional Helpers

MoGe is required only when `load_moge=true` and `camera_mode=moge`. If you disable MoGe, use `camera_mode=manual`.

RMBG-2.0 is required only when `load_rembg=true` and `background_mode=auto_remove`. If you disable RMBG, use `background_mode=none` or pass an RGBA image and use `background_mode=keep_alpha`.
