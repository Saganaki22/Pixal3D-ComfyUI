# Linux / WSL CUDA Requirements

Use this when `requirements.txt`, `pip install .`, `uv pip install .`, or `install.py` finished but **Pixal3D Environment Check** still reports missing CUDA packages.

The guarded installer does not change PyTorch and does not auto-build these packages. They must match your active ComfyUI Python, PyTorch, CUDA runtime, and GPU.

## Start From ComfyUI Python

From the ComfyUI root:

```bash
./venv/bin/python -m pip install -r custom_nodes/Pixal3D-ComfyUI/requirements.txt
./venv/bin/python custom_nodes/Pixal3D-ComfyUI/install.py --check
```

Check the stack:

```bash
./venv/bin/python - <<'PY'
import sys, torch
print("python", sys.version)
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

## System Build Tools

Ubuntu / WSL:

```bash
sudo apt update
sudo apt install -y git build-essential cmake ninja-build python3-dev libjpeg-dev
```

You also need a working NVIDIA driver and a CUDA toolkit/new enough `nvcc` if a package has to compile locally.

## FlashAttention

Pixal3D needs either `flash_attn` or `flash_attn_interface`.

Try the normal Linux path first:

```bash
./venv/bin/python -m pip install --no-build-isolation flash-attn
```

If your stack needs a specific wheel/version, install that exact wheel instead. Verify:

```bash
./venv/bin/python -c "import flash_attn; print(flash_attn.__version__)"
```

FlashAttention 3 is also valid if it provides `flash_attn_interface`:

```bash
./venv/bin/python -c "import flash_attn_interface; print('flash_attn_interface ok')"
```

## Required Pixal3D CUDA Extensions

Required imports:

```text
flex_gemm_ap or flex_gemm
cumesh_vb or cumesh
o_voxel_vb_ap or o_voxel
drtk
```

If you do not have prebuilt Linux wheels for your exact stack, build from source inside the ComfyUI venv:

```bash
mkdir -p /tmp/pixal3d_extensions

git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/pixal3d_extensions/FlexGEMM
./venv/bin/python -m pip install --no-deps --no-build-isolation /tmp/pixal3d_extensions/FlexGEMM

git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/pixal3d_extensions/CuMesh
./venv/bin/python -m pip install --no-deps --no-build-isolation /tmp/pixal3d_extensions/CuMesh

git clone --depth=1 https://github.com/microsoft/TRELLIS.2.git /tmp/pixal3d_extensions/TRELLIS.2
./venv/bin/python -m pip install --no-deps --no-build-isolation /tmp/pixal3d_extensions/TRELLIS.2/o-voxel

./venv/bin/python -m pip install --no-deps --no-build-isolation "git+https://github.com/facebookresearch/DRTK.git@stable"
```

If a build fails, do not let pip replace Torch. Fix the compiler/CUDA error, then retry with `--no-deps`.

## Optional Renderer Packages

These are not required for the basic GLB export path, but some renderer/texturing paths may use them:

```bash
git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/pixal3d_extensions/nvdiffrast
./venv/bin/python -m pip install --no-deps --no-build-isolation /tmp/pixal3d_extensions/nvdiffrast

git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/pixal3d_extensions/nvdiffrec
./venv/bin/python -m pip install --no-deps --no-build-isolation /tmp/pixal3d_extensions/nvdiffrec
```

## NATTEN / Strict NAF

Linux/WSL is the recommended path for full NAF because upstream NATTEN publishes CUDA `libnatten` wheels for recent official PyTorch builds.

Use the guarded installer when your stack is supported:

```bash
PIXAL3D_INSTALL_NATTEN=1 ./venv/bin/python custom_nodes/Pixal3D-ComfyUI/install.py
```

or install the matching upstream tag manually:

```bash
./venv/bin/python -m pip install --no-deps "natten==0.21.6+torch2100cu130" -f https://whl.natten.org
```

Pick the tag that matches your PyTorch/CUDA:

| PyTorch | CUDA | Package |
|---|---:|---|
| `2.11.0+cu130` | 13.0 | `natten==0.21.6+torch2110cu130` |
| `2.11.0+cu128` | 12.8 | `natten==0.21.6+torch2110cu128` |
| `2.11.0+cu126` | 12.6 | `natten==0.21.6+torch2110cu126` |
| `2.10.0+cu130` | 13.0 | `natten==0.21.6+torch2100cu130` |
| `2.10.0+cu128` | 12.8 | `natten==0.21.6+torch2100cu128` |
| `2.10.0+cu126` | 12.6 | `natten==0.21.6+torch2100cu126` |

Verify:

```bash
./venv/bin/python -c "import natten; print(natten.__version__, natten.HAS_LIBNATTEN)"
```

`HAS_LIBNATTEN` must be `True` for `naf_mode=strict`. If it is `False`, keep `naf_mode=fallback_if_missing`.

## Final Verification

```bash
./venv/bin/python - <<'PY'
import importlib.util
mods = ["flash_attn", "flex_gemm", "cumesh", "o_voxel", "drtk"]
for m in mods:
    print(m, "OK" if importlib.util.find_spec(m) else "MISSING")
try:
    import natten
    print("natten", natten.__version__, "HAS_LIBNATTEN", natten.HAS_LIBNATTEN)
except Exception as e:
    print("natten MISSING", e)
PY
```

Then restart ComfyUI and run **Pixal3D Environment Check**.

