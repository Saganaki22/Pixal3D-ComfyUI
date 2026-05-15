<div align="center">

# Pixal3D: Pixel-Aligned 3D Generation from Images

<h3>SIGGRAPH 2026</h3>

[Dong-Yang Li](https://ldyang694.github.io/)¹ · [Wang Zhao](https://thuzhaowang.github.io/)²* · [Yuxin Chen](https://orcid.org/0000-0002-7854-1072)² · [Wenbo Hu](https://wbhu.github.io/)² · [Meng-Hao Guo](https://menghaoguo.github.io/)¹ · [Fang-Lue Zhang](https://fanglue.github.io/)³ · [Ying Shan](https://www.linkedin.com/in/YingShanProfile)² · [Shi-Min Hu](https://cg.cs.tsinghua.edu.cn/shimin.htm)¹✉

¹Tsinghua University (BNRist) &nbsp;&nbsp; ²Tencent ARC Lab &nbsp;&nbsp; ³Victoria University of Wellington

*Project lead &nbsp;&nbsp; ✉Corresponding author

</div>

<div align="center">
  <a href="https://ldyang694.github.io/projects/pixal3d/"><img src=https://img.shields.io/badge/Project%20Page-333399.svg?logo=googlehome height=22px></a>
  <a href="https://huggingface.co/spaces/TencentARC/Pixal3D"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Demo-276cb4.svg height=22px></a>
  <a href="https://huggingface.co/TencentARC/Pixal3D"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg height=22px></a>
  <a href="https://arxiv.org/abs/2605.10922"><img src=https://img.shields.io/badge/Arxiv-b5212f.svg?logo=arxiv height=22px></a>
</div>

<div align="center">
   <img width="3840" height="2160" alt="teaser-jpeg" src="https://github.com/user-attachments/assets/80c31413-e51c-437f-9c5f-1c7fd7ee77f3" />

</div>

**Pixal3D** generates high-fidelity 3D assets from a single image. Unlike previous methods that loosely inject image features via attention, Pixal3D explicitly lifts pixel features into 3D through back-projection, establishing direct pixel-to-3D correspondences. This enables near-reconstruction-level fidelity with detailed geometry and PBR textures.

---

# Pixal3D-ComfyUI

**Pixal3D image-to-3D nodes for ComfyUI** - local TencentARC Pixal3D generation with textured GLB export, FlashAttention 2/3 backend selection, and ComfyUI DynamicVRAM/Aimdo support.

[![ComfyUI](https://img.shields.io/badge/ComfyUI-custom%20node-2f80ed)](https://github.com/comfyanonymous/ComfyUI)
[![Windows CUDA](https://img.shields.io/badge/Windows-CUDA%20required-76b900)](docs/windows_wheels.md)
[![Python](https://img.shields.io/badge/Python-3.10--3.13-3776ab)](docs/compatibility_matrix.md)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8%2B-ee4c2c)](docs/compatibility_matrix.md)
[![Pixal3D Model](https://img.shields.io/badge/HuggingFace-TencentARC%2FPixal3D-blue)](https://huggingface.co/TencentARC/Pixal3D)
[![MoGe Weights](https://img.shields.io/badge/HuggingFace-Comfy--Org%2FMoGe-blue)](https://huggingface.co/Comfy-Org/MoGe)
[![RMBG-2.0](https://img.shields.io/badge/RMBG--2.0-gated-orange)](https://huggingface.co/briaai/RMBG-2.0)
[![License](https://img.shields.io/badge/License-see%20LICENSE-lightgrey)](LICENSE)

[中文说明](README_ZH.md) | [Compatibility](docs/compatibility_matrix.md) | [Portable Install](docs/portable_standalone_install.md) | [Windows Wheels](docs/windows_wheels.md) | [Troubleshooting](docs/troubleshooting.md) | [Related Repos](docs/related_repos.md)

<p align="center">
  <img src="https://github.com/user-attachments/assets/45d596b4-9070-44d2-8e4f-1019169d3daa" width="1200"><br><br>

  <img src="https://github.com/user-attachments/assets/a2ef8b6e-ff68-4a81-a595-1e84eab2062c" width="800">
</p>



## Features

- Pixal3D image-to-3D generation directly inside ComfyUI
- Textured `.glb` export from Pixal3D voxel attributes
- FlashAttention 2 and FlashAttention 3 runtime selection
- `auto` attention mode picks FlashAttention 3 when installed, otherwise FlashAttention 2
- ComfyUI model management, unload, DynamicVRAM, and Aimdo/MemoryVisualization visibility
- Native low-VRAM Pixal3D mode for staged CPU/GPU movement
- GLB path output connects directly to ComfyUI's native **Preview 3D & Animation**

## Installation

### Method 1: ComfyUI Manager

Search for `Pixal3D` or `Pixal3D-ComfyUI` in ComfyUI Manager and install it.

### Method 2: Manual Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Saganaki22/Pixal3D-ComfyUI.git
cd Pixal3D-ComfyUI
python -m pip install -r requirements.txt
python install.py --check
```

### Method 3: uv

```bash
cd ComfyUI/custom_nodes/Pixal3D-ComfyUI
uv pip install -r requirements.txt
```

Package-style installs also use the same dependency set:

```bash
python -m pip install .
uv pip install .
```

Restart ComfyUI after installing or updating.

Pixal3D-ComfyUI includes a guarded `install.py` for ComfyUI Manager, portable ComfyUI, standalone venv installs, and Linux venv installs. By default it installs only the safe runtime requirements and prints an environment report. It does **not** change PyTorch and does **not** install CUDA wheels unless you explicitly enable an exact-match wheel path.

`requirements.txt` is the safe runtime list. It intentionally does not include `torch`, `torchvision`, `flash-attn`, `triton`, `flex_gemm`, `cumesh`, `o_voxel`, `drtk`, or `nvdiffrast`. Those are binary/CUDA stack packages and must be installed from matching wheels. See [requirements-cuda-manual.txt](requirements-cuda-manual.txt), [portable install guide](docs/portable_standalone_install.md), and [Windows wheel guide](docs/windows_wheels.md).

Plain `natten==0.21.6` is included as a baseline dependency because similar Pixal3D wrappers import it. Do not mistake that for strict NAF support. Pixal3D's strict NAF path requires `natten.HAS_LIBNATTEN == True`; a generic `natten-0.21.6-py3-none-any.whl` imports but does not provide CUDA libnatten.

FlashAttention 2 or 3 is a prerequisite. Install a matching FlashAttention wheel for your Python, PyTorch, CUDA, and OS before loading Pixal3D.

If **Pixal3D Environment Check** reports missing `flex_gemm`, `cumesh`, `o_voxel`, or `drtk`, the normal install did not fail. Those are Pixal3D compiled CUDA wheels and are opt-in. On a known Windows stack, run:

```bash
python install.py --install-known-cuda
```

If your stack is not in the bundled wheel map, install matching wheels manually from [Windows wheel guide](docs/windows_wheels.md).

### Platform Reality Check

For the smoothest full upstream Pixal3D experience, Linux or WSL is recommended because upstream NATTEN publishes prebuilt NATTEN/libnatten wheels for recent official PyTorch CUDA stacks there.

Native Windows is supported and can generate/export GLBs, but it may need fallback settings unless exact Windows CUDA wheels exist for your stack. In particular, for Python 3.12 + PyTorch 2.10 + CUDA 13.0, there is currently no known official `win_amd64` NATTEN/libnatten wheel for `natten==0.21.6+torch2100cu130`. Plain `natten==0.21.6` is installed for baseline imports, but if `natten.HAS_LIBNATTEN` is `False`, use `naf_mode=fallback_if_missing` instead of `strict`.

Recommended default:

| Environment | Recommendation |
|---|---|
| Linux/WSL NVIDIA | Best path for full upstream NAF if official NATTEN/libnatten wheels match your Torch/CUDA |
| Native Windows NVIDIA | Works, but use exact CUDA extension wheels and `naf_mode=fallback_if_missing` unless `natten.HAS_LIBNATTEN` is `True` |

<details>
<summary>Guarded Installer Policy</summary>

Some 3D ComfyUI nodes use `comfy-env` with `install.py`, `prestartup_script.py`, and `comfy-env.toml` to build isolated CUDA environments automatically. That can be convenient, but it depends on the wheel map matching the user's exact stack.

Pixal3D-ComfyUI keeps the risky parts explicit:

- No `prestartup_script.py`
- No automatic Torch changes
- No automatic CUDA wheel install unless an exact known wheel map is explicitly enabled
- No automatic model download unless `download_if_missing` is enabled
- Normal `pip` and `uv` installs for runtime requirements

This makes it less automatic, but safer for custom ComfyUI installs, portable ComfyUI, newer PyTorch/CUDA stacks, and users who already have working FlashAttention/Triton wheels.

</details>

## Compatibility

This nodepack is written to import cleanly on normal ComfyUI, portable ComfyUI, venv installs, and uv-managed installs. Actual generation/export requires CUDA because Pixal3D depends on sparse attention and mesh/voxel extension wheels.

The node is **not pinned to one tiny stack**. It should work on any Python/PyTorch/CUDA combo where the required extension modules import successfully inside the same ComfyUI Python environment.

| Component | Supported range | Notes |
|-----------|-----------------|-------|
| VRAM | **20–32 GB required** | `1536_cascade` needs ~32 GB; `1024_cascade` may fit in ~20 GB with `native_low_vram` |
| System RAM | **40 GB minimum** | Pixal3D stages large CPU-side tensors before GPU transfer |
| OS | Windows and Linux CUDA supported; macOS import-only | macOS should not break ComfyUI import, but CUDA generation/export is not supported unless compatible deps exist |
| Python | `3.10`-`3.13` expected if wheels exist, `3.12.1` target-friendly | Wheels must match the Python ABI, for example `cp312` for Python 3.12.x |
| PyTorch | `2.8+` expected if wheels exist, including `2.10` | The extension wheels must match the installed Torch ABI/build |
| CUDA | `12.8` tested, `13.x` allowed if wheels exist | Use wheels matching `torch.version.cuda`, not just the system CUDA toolkit |
| FlashAttention 2 | `flash-attn 2.8.3`, Torch `2.8.x`, CUDA `12.8`, Python `3.12` tested | Provides the `flash_attn` module |
| FlashAttention 2 newer stacks | Torch `2.10` / CUDA `13.x` should work if your wheel imports | Select `flash_attn_2` or use `auto` |
| FlashAttention 3 | Torch `>=2.9` and matching wheel expected | Must provide the `flash_attn_interface` module |
| Triton | Windows: matching `triton-windows`; Linux: matching `triton` | Required by several modern CUDA wheel stacks |
| Required Pixal3D CUDA wheels | `flex_gemm_ap`/`flex_gemm`, `o_voxel_vb_ap`/`o_voxel`, `cumesh_vb`/`cumesh`, `drtk` | These must match Python, Torch, CUDA, and OS |
| Optional CUDA wheels | `nvdiffrast`, `nvdiffrec_render` | Useful for renderer paths, not the basic GLB export path |
| ComfyUI | Current ComfyUI with `CoreModelPatcher` and `load_models_gpu` | Needed for DynamicVRAM/Aimdo/MemoryVisualization visibility |
| GPU | CUDA-capable NVIDIA GPU | Newer GPU architectures need extension wheels built for that Torch/CUDA stack |
| CPU only | Not supported | Pixal3D sparse/mesh ops need CUDA |

Example verified stack, not a required install path:

```text
Windows
Python 3.12
PyTorch 2.8.0+cu128
CUDA wheel target cu128
flash-attn 2.8.3+cu128torch2.8.0
triton-windows 3.5+
```

Another valid stack shape:

```text
Windows
Python 3.12.1
PyTorch 2.10.x
CUDA 13.x
FlashAttention 2 or 3 matching Torch/CUDA/Python
Triton matching Torch/CUDA/Python
Pixal3D CUDA extension wheels matching Torch/CUDA/Python
```

Use **Pixal3D Environment Check** inside ComfyUI before loading the model. It checks `torch`, CUDA, FlashAttention 2/3, Triton, `flex_gemm_ap`, `cumesh_vb`, `o_voxel_vb_ap`, and `drtk` without downloading the model.

More setup detail:

- [Compatibility matrix](docs/compatibility_matrix.md)
- [Portable and standalone install](docs/portable_standalone_install.md)
- [Windows wheel guide](docs/windows_wheels.md)
- [Related repo findings](docs/related_repos.md)
- [Troubleshooting](docs/troubleshooting.md)

<details>
<summary>Production Readiness</summary>

This nodepack is close to production for Windows CUDA users who already have matching extension wheels installed, but it is not a one-click package for every environment. Release readiness depends on these checks:

| Check | Status |
|-------|--------|
| ComfyUI import without model download | Ready |
| Native ComfyUI MoGe folder support | Ready |
| RMBG-2.0 gated-model note | Ready |
| GLB export with Windows-viewer-friendly PNG textures | Ready |
| DynamicVRAM/Aimdo model wrapper | Ready |
| Automatic CUDA wheel installation | Opt-in exact-match installer only |
| Exact upstream NAF on Windows | Requires a real CUDA NATTEN/libnatten wheel; fallback mode works without it |
| Fresh-machine smoke test | Recommended before tagging a production release |

Run **Pixal3D Environment Check** first. A production-capable Windows install must show the required CUDA modules importing in the same Python environment that launches ComfyUI.

</details>

## Model Setup

The loader looks for the Pixal3D model in:

```text
ComfyUI/models/Pixal3D/TencentARC_Pixal3D/
```

Set `download_if_missing` to `true` on **Pixal3D Model Loader** to download `TencentARC/Pixal3D` there. The default is `false`, so the node will not download anything unless you ask it to.

Helper models also live under `ComfyUI/models/Pixal3D/`, except native ComfyUI MoGe:

```text
ComfyUI/models/Pixal3D/briaai_RMBG-2.0/
ComfyUI/models/Pixal3D/camenduru_dinov3-vitl16-pretrain-lvd1689m/
```

Preferred clean folder names are `owner_repo`, because Windows folders cannot use the Hugging Face slash. The Pixal3D/RMBG/DINO helpers also check common manual-download names such as `RMBG-2.0`, `Pixal3D`, and Hugging Face cache-style folders like `models--owner--repo/snapshots/<hash>/`.

The model folders may be normal directories, Windows junctions, or symlinks. Broken links will be treated as missing models. Linked folders should still expose normal files such as `pipeline.json`, `.json`, `.safetensors`, and `ckpts/*.safetensors`; blob-only Hugging Face cache folders are not enough.

Native ComfyUI MoGe uses [Comfy-Org/MoGe](https://huggingface.co/Comfy-Org/MoGe). Place the files directly in `ComfyUI/models/moge/`:

```text
ComfyUI/
└── models/
    └── moge/
        ├── moge_1_vitl_fp16.safetensors
        └── moge_2_vitl_normal_fp16.safetensors
```

Pixal3D-ComfyUI uses `moge_2_vitl_normal_fp16.safetensors` from that native ComfyUI folder for `camera_mode=moge`. It does not use a `Ruicheng/moge-2-vitl` snapshot folder.

If `download_if_missing` is `true`, Pixal3D-ComfyUI downloads missing Comfy-Org/MoGe files into `ComfyUI/models/moge/` and missing Pixal3D helper snapshots into `ComfyUI/models/Pixal3D/`. If `download_if_missing` is `false`, Pixal3D-ComfyUI will not download these helper models. `hf_endpoint` defaults to `https://huggingface.co`; set it to a mirror such as `https://hf-mirror.com` when downloading from regions where the normal Hugging Face domain is blocked.

`briaai/RMBG-2.0` is gated on Hugging Face. To use `background_mode=auto_remove`, accept the model terms and either log in with Hugging Face before launching ComfyUI, set `HF_TOKEN`, or place the downloaded snapshot in `ComfyUI/models/Pixal3D/briaai_RMBG-2.0/`.

Node downloads remove Hugging Face `.cache` metadata and `.git` folders after use. MoGe files are placed directly in `ComfyUI/models/moge/`; other helper snapshots are normalized under `ComfyUI/models/Pixal3D/`. The runtime expects normal files such as `.safetensors`, `pipeline.json`, and `ckpts/*.safetensors`, not blob-only cache directories.

Torch Hub helper code for Pixal3D's NAF upsampler is redirected to:

```text
ComfyUI/models/Pixal3D/torch_hub/
```

Official TencentARC Pixal3D uses NAF for the shape and texture stages, and those released weights expect 2048-channel projected features. On Windows stacks without a matching CUDA NATTEN build, leave `naf_mode=fallback_if_missing`: the node keeps the expected tensor shape by duplicating DINO projection features and will not download NAF when `download_if_missing=false`. For exact upstream NAF behavior, install a CUDA-enabled NATTEN build for your Python/PyTorch/CUDA stack and set `naf_mode=strict`.

`naf_target_size` controls real NAF only. Leave it on `upstream` for normal behavior. Lower values such as `512`, `256`, or `128` reduce NAF VRAM if strict NAF works, and are ignored by fallback mode.

## Windows CUDA Wheels

This node uses ComfyUI's Python environment only. Do not install these packages into system Python.

For Windows, FlashAttention 2 is enough if the wheel matches your exact Python, PyTorch, and CUDA build:

```bash
cd C:\path\to\ComfyUI
.\venv\Scripts\python.exe -m pip show flash-attn
```

Portable ComfyUI example:

```bash
cd ComfyUI_windows_portable
.\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\Pixal3D-ComfyUI\requirements.txt
```

uv example:

```bash
uv pip install --python C:\path\to\ComfyUI\venv\Scripts\python.exe -r C:\path\to\ComfyUI\custom_nodes\Pixal3D-ComfyUI\requirements.txt
```

If you use package-style install instead of requirements files, `pip install .` and `uv pip install .` install the same baseline runtime dependencies from `pyproject.toml`, including plain `natten==0.21.6`. CUDA wheels still remain manual or opt-in because they must match the exact stack.

FlashAttention 3 also works if you install a wheel that provides `flash_attn_interface` for your exact Python, CUDA, and PyTorch build. Use the loader's `attention_backend` dropdown:

| Option | What it uses |
|--------|--------------|
| `auto` | FlashAttention 3 if present, otherwise FlashAttention 2 |
| `flash_attn_2` | `flash_attn` |
| `flash_attn_3` | `flash_attn_interface` |

Pixal3D sparse attention needs FlashAttention 2 or 3; plain PyTorch SDPA is not enough for the sparse stages.

## Nodes

### Pixal3D Model Loader

Loads the Pixal3D pipeline and returns a Comfy-managed model handle.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_repo` | `TencentARC/Pixal3D` | Hugging Face repo or local Pixal3D model folder |
| `hf_endpoint` | `https://huggingface.co` | Hugging Face endpoint or mirror used when `download_if_missing` is enabled |
| `attention_backend` | `auto` | `auto`, `flash_attn_2`, or `flash_attn_3` |
| `vram_mode` | `dynamic_vram` | Best-effort Comfy/Aimdo staging with Comfy-aware torch ops |
| `download_if_missing` | `false` | Downloads Pixal3D/helper models into `ComfyUI/models/Pixal3D/` and native MoGe into `ComfyUI/models/moge/` only when enabled |
| `load_moge` | `true` | Load MoGe for automatic camera estimation |
| `load_rembg` | `true` | Load gated RMBG-2.0 for built-in background removal |
| `naf_mode` | `fallback_if_missing` | Use duplicated DINO features if CUDA NATTEN/NAF is unavailable; `strict` requires real NAF |
| `naf_target_size` | `upstream` | Real NAF upsample size; lower values reduce VRAM and are ignored by fallback mode |
| `preload_naf` | `false` | Preload the NAF upsampler only when `naf_mode=strict` and CUDA NATTEN/libnatten is available |
| `force_reload` | `false` | Rebuild the cached model handle |

Advanced source override: set `PIXAL3D_REPO_PATH` before launching ComfyUI if you want to use a different Pixal3D source checkout instead of the vendored source.

### Pixal3D Image To 3D

Runs Pixal3D from a ComfyUI `IMAGE`.

Background handling is built in:

| Mode | Description |
|------|-------------|
| `auto_remove` | Default. Uses Pixal3D/rembg to remove the background when the image has no alpha |
| `keep_alpha` | Uses the input alpha mask when present; falls back to auto remove if there is no alpha |
| `none` | Sends the RGB image through without background removal |

Outputs:

| Output | Description |
|--------|-------------|
| `pixal3d_result` | Pixal3D mesh, voxel attributes, latents, and camera metadata |

### Pixal3D Export GLB

Exports a textured GLB from `pixal3d_result`.

Output files are written to `ComfyUI/output/`.

Exports use embedded PNG textures for broad Windows/glTF viewer compatibility. Older Pixal3D-ComfyUI builds wrote WebP textures with `EXT_texture_webp`; those GLBs could open in some tools but fail in Windows' built-in 3D viewer.

`remesh` defaults to `true` to match the Pixal3D demo path and is passed through exactly as set in the node. If cleanup creates fragmented meshes, try `remesh=false` and keep the upstream-style export defaults: `decimation_target=1000000` and `texture_size=4096`.

Connect:

```text
Pixal3D Export GLB glb_path
  -> Preview 3D & Animation model_file
```

### Pixal3D Unload Model

Fully removes the loaded Pixal3D model handle from ComfyUI model management and Pixal3D-ComfyUI's Python cache. Use this when you want CPU RAM released, not only VRAM offloaded.

## Recommended Workflow

```text
Load Image
  -> Pixal3D Model Loader
  -> Pixal3D Image To 3D
  -> Pixal3D Export GLB
```

Optional preview:

```text
Pixal3D Export GLB glb_path
  -> Preview 3D & Animation model_file
```

<details>
<summary>VRAM Modes</summary>

`dynamic_vram` is the default loader mode. Pixal3D-ComfyUI builds Pixal3D with Comfy/Aimdo-aware `Linear`, `Conv`, `LayerNorm`, `GroupNorm`, and `Embedding` ops where possible, then wraps the pipeline in ComfyUI's model-management path.

Pixal3D-ComfyUI keeps one active pipeline cache. Changing Model Loader settings such as `vram_mode`, `attention_backend`, helper-model toggles, or NAF settings unloads and destroys the previous handle before loading the new one. The **Pixal3D Unload Model** node also removes the active handle from the Python cache. Pixal3D-ComfyUI also hooks ComfyUI's global unload button so native unloads clear the Pixal3D cache too.

Task Manager may still show some RAM held after unload because Python, PyTorch, memory-mapped safetensors, Hugging Face/Transformers imports, and Windows allocators can keep reserved pages for reuse. That is different from the Pixal3D model object still being referenced. A full ComfyUI restart is the only guaranteed way to return every reserved page to the OS immediately.

Pixal3D is still not fully Comfy-native: it has custom sparse kernel modules and large temporary tensors that Aimdo cannot virtualize like a normal Comfy UNet. If Comfy still reports a large `Force pre-loaded` value or a 1536 run OOMs, use `native_low_vram` as the fallback. That mode bypasses Comfy's bulk model load and lets Pixal3D move stages to GPU only when needed. In that mode the background remover is moved to GPU only for preprocessing and then returned to CPU, MoGe is moved to GPU only for camera estimation and then returned to CPU, and the upstream Pixal3D pipeline stages its flow/decoder modules one at a time.

Use `full_gpu` only when you want the whole model resident on the GPU and your card has enough free VRAM.
</details>

<details>
<summary>Safety Notes</summary>

- No packages are installed into system Python.
- `install.py` uses the Python that launches it; use ComfyUI's Python or portable `python_embeded`.
- There is no `prestartup_script.py`.
- CUDA wheels are not installed automatically unless an exact known wheel path is explicitly enabled.
- The model is not downloaded during ComfyUI startup.
- `download_if_missing` is off by default.
- Pixal3D source imports are lazy; the heavy model code loads only when the loader node runs.
- CUDA module aliases for `cumesh`, `flex_gemm`, and `o_voxel` are created only when Pixal3D loads, and point at the installed wheel modules.

</details>

## Windows CUDA Wheel Resources

- [PozzettiAndrea/cuda-wheels](https://github.com/PozzettiAndrea/cuda-wheels/releases) — direct Windows wheels for `flex_gemm_ap`, `cumesh_vb`, `o_voxel_vb_ap`, `drtk`, and some `flash_attn` builds.
- [visualbruno/ComfyUI-Trellis2 wheels](https://github.com/visualbruno/ComfyUI-Trellis2/tree/main/wheels) — alternate Windows wheels for `flex_gemm`, `cumesh`, `o_voxel`, `nvdiffrast`, `nvdiffrec_render`, and some NATTEN builds.
- [Wildminder/AI-windows-whl](https://huggingface.co/Wildminder/AI-windows-whl/tree/main) — Windows AI wheel index, especially useful for FlashAttention and related AI packages.
- [lldacing/NATTEN-windows](https://huggingface.co/lldacing/NATTEN-windows/tree/main) — Windows NATTEN wheels where available; strict NAF still requires `natten.HAS_LIBNATTEN == True`.

## 🤗 Acknowledgements

This project is heavily built upon [Trellis.2](https://github.com/microsoft/TRELLIS.2) and [Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2). We sincerely thank the authors for their outstanding work on scalable 3D generation, which serves as the foundation of our codebase and model architecture.

We also thank the following repos for their great contributions:

- [Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2)
- [Trellis](https://github.com/microsoft/TRELLIS)
- [Trellis.2](https://github.com/microsoft/TRELLIS.2)

## 📄 Citation

If you find this work useful, please consider citing:

```bibtex
@article{li2026pixal3d,
    title={Pixal3D: Pixel-Aligned 3D Generation from Images},
    author={Li, Dong-Yang and Zhao, Wang and Chen, Yuxin and Hu, Wenbo and Guo, Meng-Hao and Zhang, Fang-Lue and Shan, Ying and Hu, Shi-Min},
    journal={arXiv preprint arXiv:2605.10922},
    year={2026}
}
```
