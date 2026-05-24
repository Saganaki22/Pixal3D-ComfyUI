# Windows Wheel Guide

This page is only for choosing and installing Windows wheels. NATTEN source build steps live in [Build_Natten_windows.md](Build_Natten_windows.md).

Install everything into the Python environment that launches ComfyUI:

```bat
cd C:\path\to\ComfyUI
venv\Scripts\python.exe -m pip install --no-deps "<wheel-url>"
```

For portable ComfyUI, use `python_embeded\python.exe` instead of `venv\Scripts\python.exe`.

## Match Your Stack

Compiled wheels must match all of these:

```text
Python ABI: cp310, cp311, cp312, ...
PyTorch build: torch2.8, torch2.9, torch2.10, ...
CUDA build: cu128, cu130, ...
OS tag: win_amd64
GPU architecture when the wheel is architecture-specific
```

Check your stack:

```bat
venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))"
```

Use `--no-deps` for manual CUDA wheels so pip does not replace a working PyTorch install.

## Required Pixal3D CUDA Wheels

These are required for Pixal3D generation/export. NATTEN does not replace them.

| Module family | Acceptable import |
|---|---|
| Sparse GEMM | `flex_gemm_ap` or `flex_gemm` |
| Mesh ops | `cumesh_vb` or `cumesh` |
| Voxel/export ops | `o_voxel_vb_ap` or `o_voxel` |
| DRTK helper | `drtk` |

Pozzetti wheels provide common Windows builds:

- https://github.com/PozzettiAndrea/cuda-wheels/releases

Example for Python 3.12, PyTorch 2.10, CUDA 13.0 on Blackwell sm120. This installs the required Pixal3D CUDA wheels plus the prebuilt NATTEN/libnatten wheel:

```bat
venv\Scripts\python.exe -m pip install --no-deps ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/flex_gemm_ap-latest/flex_gemm_ap-1.0.0%2Bcu130torch2.10-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/cumesh_vb-latest/cumesh_vb-1.0%2Bcu130torch2.10-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/o_voxel_vb_ap-latest/o_voxel_vb_ap-0.0.1%2Bcu130torch2.10-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/drtk-latest/drtk-0.1.0%2Bcu130torch2.10-cp312-cp312-win_amd64.whl" ^
  "https://huggingface.co/drbaph/NATTEN-0.21.6-torch2100cu130-cp312-cp312-win_amd64/resolve/main/natten-0.21.6+torch2100cu130-cp312-cp312-win_amd64.whl"
```

If your Python, PyTorch, CUDA, or GPU architecture does not match that NATTEN wheel, omit the final NATTEN URL and use fallback NAF mode.

Example for Python 3.12, PyTorch 2.8, CUDA 12.8 on Blackwell sm100/sm120. This installs matching Pixal3D CUDA wheels plus the `naxneri` NATTEN/libnatten wheel:

```bat
venv\Scripts\python.exe -m pip install --no-deps ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/flex_gemm_ap-latest/flex_gemm_ap-1.0.0%2Bcu128torch2.8-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/cumesh_vb-latest/cumesh_vb-1.0%2Bcu128torch2.8-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/o_voxel_vb_ap-latest/o_voxel_vb_ap-0.0.1%2Bcu128torch2.8-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/drtk-latest/drtk-0.1.0%2Bcu128torch2.8-cp312-cp312-win_amd64.whl" ^
  "https://huggingface.co/naxneri/natten-0.21.6-blackwell-cu128-cp312-cp312-win_amd64/resolve/main/natten-0.21.6-blackwell-cu128-cp312-cp312-win_amd64.whl"
```

For PyTorch 2.9 or another CUDA 12.8 stack, change the four Pozzetti URLs to wheels built for that exact Torch version. Keep the NATTEN URL only when it matches your Python, CUDA, and GPU.

Other Windows wheel sources:

- https://huggingface.co/Wildminder/AI-windows-whl/tree/main
- https://github.com/visualbruno/ComfyUI-Trellis2/tree/main/wheels

Do not install a wheel just because the Python tag matches. The Torch and CUDA tags must match too.

## Attention Wheel

Pixal3D needs one attention backend:

| Backend | Import |
|---|---|
| FlashAttention 2 | `flash_attn` |
| FlashAttention 3 | `flash_attn_interface` |

Example FlashAttention 2 wheel for Python 3.12, PyTorch 2.10, CUDA 13.0:

```bat
venv\Scripts\python.exe -m pip install --no-deps "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/flash_attn-latest/flash_attn-2.8.3%2Bcu130torch2.10-cp312-cp312-win_amd64.whl"
```

Use `attention_backend=auto` unless you need to force a specific backend.

## Triton

Some Windows stacks need Triton for related CUDA kernels:

```bat
venv\Scripts\python.exe -m pip install -U "triton-windows<3.7"
```

Install it only if your stack needs it or **Pixal3D Environment Check** reports it missing.

## NATTEN / NAF

NAF is Pixal3D's feature refinement step for shape and texture generation. NAF uses NATTEN.

Strict upstream NAF requires NATTEN with CUDA `libnatten`:

```bat
venv\Scripts\python.exe -c "import natten; print(natten.__version__, natten.HAS_LIBNATTEN)"
```

`HAS_LIBNATTEN` must be `True` for strict NAF.

If pip installs normal `natten-0.21.6-py3-none-any.whl`, NATTEN may import but CUDA `libnatten` is not available. In that case set:

```text
Pixal3D Model Loader naf_mode=fallback_if_missing
Pixal3D Model Loader preload_naf=false
```

Fallback mode is expected on Windows when no matching NATTEN/libnatten wheel exists. It can be slower, use more RAM/VRAM, and produce lower quality than strict NAF, but it lets Pixal3D run.

## Community Windows NATTEN Wheels

Use these only when they match your stack exactly:

| Python | PyTorch | CUDA | GPU | Wheel |
|---|---|---|---|---|
| 3.12.10 / 3.13.12 | 2.10 | 13.0 | Ampere sm86, RTX 3050-3090 Ti | [NeilsMabet/Natten-0.21.6-Amphere-wheel-windows](https://github.com/NeilsMabet/Natten-0.21.6-Amphere-wheel-windows) |
| 3.12 | 2.10 | 13.0 | Blackwell sm120 | [drbaph/NATTEN-0.21.6-torch2100cu130-cp312-cp312-win_amd64](https://huggingface.co/drbaph/NATTEN-0.21.6-torch2100cu130-cp312-cp312-win_amd64) |
| 3.12 | 2.8+ | 12.8 | Blackwell sm100/sm120 | [naxneri/natten-0.21.6-blackwell-cu128-cp312-cp312-win_amd64](https://huggingface.co/naxneri/natten-0.21.6-blackwell-cu128-cp312-cp312-win_amd64) |

Install example for the Blackwell sm120 Python 3.12 wheel:

```bat
venv\Scripts\python.exe -m pip install --no-deps "https://huggingface.co/drbaph/NATTEN-0.21.6-torch2100cu130-cp312-cp312-win_amd64/resolve/main/natten-0.21.6+torch2100cu130-cp312-cp312-win_amd64.whl"
```

Then verify `HAS_LIBNATTEN=True`.

If no Windows NATTEN wheel matches your exact Python, PyTorch, CUDA, and GPU architecture, choose one path:

```text
Recommended: use naf_mode=fallback_if_missing and preload_naf=false
Advanced: build NATTEN yourself using Build_Natten_windows.md
```

## Verify

Run these in ComfyUI's Python:

```bat
venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda)"
venv\Scripts\python.exe -c "import importlib.util as u; print(u.find_spec('flash_attn') or u.find_spec('flash_attn_interface'))"
venv\Scripts\python.exe -c "import importlib.util as u; print(u.find_spec('flex_gemm_ap') or u.find_spec('flex_gemm')); print(u.find_spec('cumesh_vb') or u.find_spec('cumesh')); print(u.find_spec('o_voxel_vb_ap') or u.find_spec('o_voxel')); print(u.find_spec('drtk'))"
venv\Scripts\python.exe -c "import natten; print(natten.__version__, natten.HAS_LIBNATTEN)"
```

Then run **Pixal3D Environment Check** in ComfyUI.
