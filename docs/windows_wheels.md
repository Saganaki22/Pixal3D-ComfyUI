# Windows Wheel Guide

Install these into ComfyUI's Python environment only. Do not install them into system Python.

Generic venv example:

```bat
cd C:\path\to\ComfyUI
venv\Scripts\python.exe -m pip install --no-deps "<wheel-url>"
```

## Required Pixal3D CUDA Wheels

Install all four matching your environment. This nodepack accepts either the Pozzetti-style module names or the generic module names used by visualbruno where the API matches:

```text
flex_gemm_ap or flex_gemm
cumesh_vb or cumesh
o_voxel_vb_ap or o_voxel
drtk
```

### Python 3.12, PyTorch 2.10, CUDA 13.0

```bat
venv\Scripts\python.exe -m pip install --no-deps ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/flex_gemm_ap-latest/flex_gemm_ap-1.0.0%2Bcu130torch2.10-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/cumesh_vb-latest/cumesh_vb-1.0%2Bcu130torch2.10-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/o_voxel_vb_ap-latest/o_voxel_vb_ap-0.0.1%2Bcu130torch2.10-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/drtk-latest/drtk-0.1.0%2Bcu130torch2.10-cp312-cp312-win_amd64.whl"
```

### Python 3.12, PyTorch 2.9, CUDA 13.0

```bat
venv\Scripts\python.exe -m pip install --no-deps ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/flex_gemm_ap-latest/flex_gemm_ap-1.0.0%2Bcu130torch2.9-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/cumesh_vb-latest/cumesh_vb-1.0%2Bcu130torch2.9-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/o_voxel_vb_ap-latest/o_voxel_vb_ap-0.0.1%2Bcu130torch2.9-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/drtk-latest/drtk-0.1.0%2Bcu130torch2.9-cp312-cp312-win_amd64.whl"
```

### Python 3.12, PyTorch 2.9, CUDA 12.8

```bat
venv\Scripts\python.exe -m pip install --no-deps ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/flex_gemm_ap-latest/flex_gemm_ap-1.0.0%2Bcu128torch2.9-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/cumesh_vb-latest/cumesh_vb-1.0%2Bcu128torch2.9-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/o_voxel_vb_ap-latest/o_voxel_vb_ap-0.0.1%2Bcu128torch2.9-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/drtk-latest/drtk-0.1.0%2Bcu128torch2.9-cp312-cp312-win_amd64.whl"
```

### Python 3.12, PyTorch 2.8, CUDA 12.8

```bat
venv\Scripts\python.exe -m pip install --no-deps ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/flex_gemm_ap-latest/flex_gemm_ap-1.0.0%2Bcu128torch2.8-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/cumesh_vb-latest/cumesh_vb-1.0%2Bcu128torch2.8-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/o_voxel_vb_ap-latest/o_voxel_vb_ap-0.0.1%2Bcu128torch2.8-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/drtk-latest/drtk-0.1.0%2Bcu128torch2.8-cp312-cp312-win_amd64.whl"
```

## Attention Wheels

Pixal3D needs FlashAttention 2 or FlashAttention 3 before the model can load. Treat this as a prerequisite; the guarded installer does not install FlashAttention.

### FlashAttention 2

Pozzetti wheels:

```bat
venv\Scripts\python.exe -m pip install --no-deps "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/flash_attn-latest/flash_attn-2.8.3%2Bcu130torch2.10-cp312-cp312-win_amd64.whl"
```

Wildminder wheels are also available for many stacks. Example for Python 3.12, PyTorch 2.10.0, CUDA 13.0:

```bat
venv\Scripts\python.exe -m pip install --no-deps "https://huggingface.co/Wildminder/AI-windows-whl/resolve/main/flash_attn-2.8.3+cu130torch2.10.0cxx11abiTRUE-cp312-cp312-win_amd64.whl"
```

### FlashAttention 3

Use only if the wheel provides `flash_attn_interface`.

Example for PyTorch 2.10, CUDA 13.0, Python 3.9+ ABI3:

```bat
venv\Scripts\python.exe -m pip install --no-deps "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.3/flash_attn_3-3.0.0+cu130torch2.10gite2743ab-cp39-abi3-win_amd64.whl"
```

Then set `attention_backend=flash_attn_3`, or leave `attention_backend=auto`.

## Triton

Triton is optional for Pixal3D-ComfyUI, but many Windows AI environments already use it.

If your Windows stack needs Triton:

```bat
venv\Scripts\python.exe -m pip install -U "triton-windows<3.7"
```

## visualbruno Wheel Folder

visualbruno's Pixal3D branch has a useful Windows wheel folder:

```text
https://github.com/visualbruno/ComfyUI-Trellis2/tree/pixal3d/wheels/Windows
```

Important folders:

| Folder | Contains | Notes for Python 3.12 + Torch 2.10.0+cu130 |
|---|---|---|
| `Torch2100` | `cumesh`, `custom_rasterizer`, `flex_gemm`, `nvdiffrast`, `o_voxel` cp311/cp312 | Useful fallback source for generic module-name wheels; no cp312 `natten` here |
| `Torch2100/CUDA 13.1` | cp311/cp312/cp313 CUDA wheels plus `natten-0.21.6-cp313` | Extension wheels may be useful if compatible; NATTEN is Python 3.13 only |
| `Torch280` | Torch 2.8-era cp311/cp312 wheels plus `natten-0.21.6-cp312` | NATTEN wheel is not for Torch 2.10 |
| `Torch270` | Torch 2.7-era cp311/cp312 wheels | Not for the locked Torch 2.10 stack |

Do not install a wheel just because the Python tag matches. The package also has to match the Torch/CUDA ABI it was built against.

## NATTEN / NAF

Official TencentARC Pixal3D uses NAF for the shape and texture stages, and NAF imports NATTEN. The released Pixal3D shape/texture weights expect 2048-channel projected features.

For Windows Python 3.12, PyTorch 2.10, CUDA 13.0, there is currently no known `win_amd64` NATTEN wheel in the official NATTEN wheel index. The tag people look for is:

```bat
natten==0.21.6+torch2100cu130
```

At the time of writing, `https://whl.natten.org` provides that tag for Linux, not Windows. `requirements.txt` installs plain `natten==0.21.6` as a baseline dependency, but if pip installs `natten-0.21.6-py3-none-any.whl`, that is not the CUDA extension build and will not satisfy strict NAF.

Pixal3D-ComfyUI defaults to `naf_mode=fallback_if_missing`, which keeps the 2048-channel tensor shape by duplicating DINO projection features when CUDA NATTEN/NAF is unavailable. For exact upstream NAF behavior, install a matching CUDA-enabled NATTEN wheel and set `naf_mode=strict`.

Official NATTEN wheel commands from `https://whl.natten.org`:

| PyTorch | CUDA | Install command |
|---|---:|---|
| `2.11.0+cu130` | 13.0 | `pip install natten==0.21.6+torch2110cu130 -f https://whl.natten.org` |
| `2.11.0+cu128` | 12.8 | `pip install natten==0.21.6+torch2110cu128 -f https://whl.natten.org` |
| `2.11.0+cu126` | 12.6 | `pip install natten==0.21.6+torch2110cu126 -f https://whl.natten.org` |
| `2.10.0+cu130` | 13.0 | `pip install natten==0.21.6+torch2100cu130 -f https://whl.natten.org` |
| `2.10.0+cu128` | 12.8 | `pip install natten==0.21.6+torch2100cu128 -f https://whl.natten.org` |
| `2.10.0+cu126` | 12.6 | `pip install natten==0.21.6+torch2100cu126 -f https://whl.natten.org` |

Notes:

- The official index describes these as x86-64/aarch64 builds, but the current wheel files for PyTorch 2.10/2.11 are Linux wheels. On Windows, pip will reject them because they are not `win_amd64`.
- For CUDA 12.6 builds, Blackwell FNA/FMHA kernels are not available. Blackwell support starts with CUDA Toolkit 12.8.
- For Windows Python 3.12 + Torch 2.10 + CUDA 13.0, do not change Python or Torch just to chase NATTEN. Use `naf_mode=fallback_if_missing` unless a real matching Windows NATTEN wheel is found or a local source build succeeds.
- Use `--no-deps` when testing a NATTEN wheel inside an existing ComfyUI environment so pip does not change Torch:

```bat
venv\Scripts\python.exe -m pip install --no-deps "natten==0.21.6+torch2100cu130" -f https://whl.natten.org
```

Check whether NATTEN is actually usable:

```bat
venv\Scripts\python.exe -c "import natten; print(natten.__version__, natten.HAS_LIBNATTEN)"
```

`HAS_LIBNATTEN` must be `True` for strict NAF. If it is `False`, set `naf_mode=fallback_if_missing`.

`comfy-sparse-attn==0.0.9` from ComfyUI-TRELLIS2 is not a replacement for NATTEN or NAF. It provides sparse/variable-length attention dispatch and helper namespace links for ComfyUI sparse primitives. It does not provide NATTEN's neighborhood attention API or CUDA `libnatten`, so it cannot enable `naf_mode=strict`.

## Native Windows NATTEN Build Attempt

Native Windows builds are not regularly tested by NATTEN upstream. This is the right shape of the build environment if you want to try anyway.

Open a normal Command Prompt and launch the MSVC developer environment:

```bat
"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
```

Then run:

```bat
cd /d C:\path\to\ComfyUI
set PATH=%CD%\venv\Scripts;%PATH%
set CMAKE_GENERATOR=Ninja
set CMAKE_MAKE_PROGRAM=%CD%\venv\Scripts\ninja.exe
set NATTEN_CUDA_ARCH=12.0
set NATTEN_N_WORKERS=8
set NATTEN_VERBOSE=1

where cl
where nvcc
where cmake
where ninja

venv\Scripts\python.exe -m pip uninstall -y natten libnatten
venv\Scripts\python.exe -m pip install --upgrade "setuptools>=80" wheel packaging
venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation --no-binary=:all: --no-cache-dir -v natten==0.21.6
venv\Scripts\python.exe -c "import natten; print(natten.__version__, natten.HAS_LIBNATTEN)"
```

For GPUs where PyTorch reports compute capability `12.0`, use `NATTEN_CUDA_ARCH=12.0`. If the build succeeds and another ComfyUI package needs old setuptools, restore it afterward:

```bat
venv\Scripts\python.exe -m pip install "setuptools==65.0.0"
```

If CMake still says Ninja or compilers are missing, the problem is the active shell, not the Python package. Confirm all four commands resolve in the same terminal before building:

```bat
where cl
where nvcc
where cmake
where ninja
```

If `cl` is missing, you are not inside the Visual Studio developer environment. If `ninja` is missing, make sure `%CD%\venv\Scripts` is first on `PATH`.

## Remesh Export Note

Pixal3D-ComfyUI accepts `Pixal3D Export GLB remesh=true` to match upstream workflows and passes that value through to `o_voxel`. If remesh creates fragmented output on your wheel/driver/GPU combination, turn it off in the export node.

Use:

```text
Pixal3D Export GLB -> decimation_target=1000000
Pixal3D Export GLB -> texture_size=4096
```

If you see tiny GLBs, loose shards, or point-cloud-looking output, do not lower decimation to 200000. Use the upstream default `1000000` or higher.

## Verification

Run:

```bat
venv\Scripts\python.exe -c "import torch, flash_attn, drtk; print(torch.__version__, torch.version.cuda); print(flash_attn.__version__)"
venv\Scripts\python.exe -c "import importlib; print(importlib.import_module('flex_gemm_ap' if importlib.util.find_spec('flex_gemm_ap') else 'flex_gemm')); print(importlib.import_module('cumesh_vb' if importlib.util.find_spec('cumesh_vb') else 'cumesh')); print(importlib.import_module('o_voxel_vb_ap' if importlib.util.find_spec('o_voxel_vb_ap') else 'o_voxel'))"
```

Then run **Pixal3D Environment Check** in ComfyUI.
