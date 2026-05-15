# Portable And Standalone Install

Pixal3D-ComfyUI now has a guarded installer for ComfyUI Manager, portable Windows builds, normal venv installs, and Linux installs.

The installer does three things:

1. Installs `requirements.txt` into the Python that is running ComfyUI.
2. Detects OS, Python ABI, PyTorch, CUDA, GPU, and import status.
3. Optionally installs exact known Pixal3D CUDA extension wheels if explicitly enabled.

It does not install CUDA `libnatten` by default, and it does not change PyTorch.
`requirements.txt` installs plain `natten==0.21.6` as the baseline package. That is not the same as a CUDA `libnatten` build; strict NAF still requires `natten.HAS_LIBNATTEN == True`.

If you prefer package-style installs, `pip install .` and `uv pip install .` read `pyproject.toml`, which mirrors the baseline runtime dependencies from `requirements.txt`. CUDA wheels are still manual or opt-in because they must match the exact Python/PyTorch/CUDA/platform stack.

## ComfyUI Manager

Manager should run `requirements.txt` and `install.py` during node installation/update.

After install, restart ComfyUI and run **Pixal3D Environment Check**.

## Windows Portable

From the portable root:

```bat
python_embeded\python.exe -s ComfyUI\custom_nodes\Pixal3D-ComfyUI\install.py
```

Check only:

```bat
python_embeded\python.exe -s ComfyUI\custom_nodes\Pixal3D-ComfyUI\install.py --check
```

Install known exact CUDA wheels when the detected stack is in the bundled wheel map:

```bat
set PIXAL3D_INSTALL_KNOWN_CUDA=1
python_embeded\python.exe -s ComfyUI\custom_nodes\Pixal3D-ComfyUI\install.py
```

The known wheel map currently covers selected Windows `cp312` stacks from the documented Pozzetti wheel URLs. If your stack is not listed, the installer prints the detected key and links you back to the wheel docs.

FlashAttention 2 or 3 remains a prerequisite. `--install-known-cuda` installs Pixal3D extension groups such as `flex_gemm`, `cumesh`, `o_voxel`, and `drtk`; it does not install FlashAttention.

Native Windows `libnatten` is not installed automatically. Upstream NATTEN points Windows users to MSVC source builds. Use `naf_mode=fallback_if_missing` unless you build NATTEN/libnatten yourself.

## Windows Standalone Venv

From the ComfyUI root:

```bat
venv\Scripts\python.exe custom_nodes\Pixal3D-ComfyUI\install.py
```

Check only:

```bat
venv\Scripts\python.exe custom_nodes\Pixal3D-ComfyUI\install.py --check
```

Install exact known Pixal3D CUDA extension wheels:

```bat
set PIXAL3D_INSTALL_KNOWN_CUDA=1
venv\Scripts\python.exe custom_nodes\Pixal3D-ComfyUI\install.py
```

Package-style dependency install from the node folder:

```bat
venv\Scripts\python.exe -m pip install .
uv pip install --python venv\Scripts\python.exe .
```

## Linux Venv

From the ComfyUI root:

```bash
./venv/bin/python custom_nodes/Pixal3D-ComfyUI/install.py
```

Check only:

```bash
./venv/bin/python custom_nodes/Pixal3D-ComfyUI/install.py --check
```

Package-style dependency install from the node folder:

```bash
./venv/bin/python -m pip install .
uv pip install --python ./venv/bin/python .
```

Linux generally has an easier path for NATTEN/libnatten because upstream NATTEN publishes wheel commands for Linux/WSL NVIDIA users. You still need matching PyTorch/CUDA wheels or local source builds for FlashAttention and the Pixal3D CUDA extensions.

For the full manual Linux/WSL install path, including `flash-attn`, `flex_gemm`, `cumesh`, `o_voxel`, `drtk`, optional `nvdiffrast`/`nvdiffrec_render`, and verification commands, see [Linux / WSL CUDA Requirements](linux_wsl_cuda.md).

Install official NATTEN+libnatten when the detected Linux/WSL stack is in the upstream wheel map:

```bash
PIXAL3D_INSTALL_NATTEN=1 ./venv/bin/python custom_nodes/Pixal3D-ComfyUI/install.py
```

or:

```bash
./venv/bin/python custom_nodes/Pixal3D-ComfyUI/install.py --install-natten
```

Known NATTEN tags from upstream:

| PyTorch | CUDA | Package |
|---------|------|---------|
| `2.11` | `cu130` | `natten==0.21.6+torch2110cu130` |
| `2.11` | `cu128` | `natten==0.21.6+torch2110cu128` |
| `2.11` | `cu126` | `natten==0.21.6+torch2110cu126` |
| `2.10` | `cu130` | `natten==0.21.6+torch2100cu130` |
| `2.10` | `cu128` | `natten==0.21.6+torch2100cu128` |
| `2.10` | `cu126` | `natten==0.21.6+torch2100cu126` |

## Why Not Full comfy-env Yet?

`comfy-env` is good when a node has a reliable environment map for every supported stack. TRELLIS2 uses it to run `install()` and `setup_env()` from tiny `install.py` and `prestartup_script.py` files.

Pixal3D-ComfyUI does not use full `comfy-env` automation yet because the difficult part is not pure dependency installation. It is exact CUDA wheel availability:

```text
Python ABI
PyTorch ABI
CUDA runtime
OS wheel tag
GPU architecture
NATTEN/libnatten availability
```

The current installer is deliberately safer:

- installs pure Python deps everywhere,
- never changes Torch,
- installs plain `natten==0.21.6` only as a baseline import/runtime package,
- can install known exact CUDA wheels only when explicitly enabled,
- tells users exactly what is missing.

If a broader verified wheel index becomes available, `comfy-env` can be added later as a second install mode.

## Supported Install Modes

| Mode | Pure deps | Known CUDA wheels | NATTEN strict NAF |
|------|-----------|-------------------|-------------------|
| ComfyUI Manager | Yes | Pixal3D extensions explicit only | Manual/source build |
| Windows portable | Yes | Pixal3D extensions explicit exact-match only | Manual/source build |
| Windows venv | Yes | Pixal3D extensions explicit exact-match only | Manual/source build |
| Linux venv | Yes | Manual for now | Easier via upstream NATTEN wheels if stack matches |

## Safety Rules

- Do not run `pip install torch` from this node.
- Do not treat plain `natten` as strict NAF support; check `natten.HAS_LIBNATTEN`.
- Use `--no-deps` for CUDA wheel URLs inside an existing ComfyUI environment.
- Restart ComfyUI after changing wheels.
- Confirm with **Pixal3D Environment Check** before loading the model.
