# Pixal3D-ComfyUI

ComfyUI custom nodes for [TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D): image-to-3D generation, textured GLB export, FlashAttention 2/3 selection, manual camera control, and ComfyUI model unload support.

[Compatibility](docs/compatibility_matrix.md) | [Windows Wheels](docs/windows_wheels.md) | [Build NATTEN On Windows](docs/Build_Natten_windows.md) | [Troubleshooting](docs/troubleshooting.md) | [Chinese README](README_ZH.md)

![Pixal3D preview](https://github.com/user-attachments/assets/45d596b4-9070-44d2-8e4f-1019169d3daa)

## Quick Start

ComfyUI Manager:

1. Open **ComfyUI Manager**.
2. Search for **Pixal3D** by **Saganaki22**.
3. Install it, restart ComfyUI, then run **Pixal3D Environment Check**.

Manual install:

```bat
cd ComfyUI\custom_nodes
git clone https://github.com/Saganaki22/Pixal3D-ComfyUI.git
cd Pixal3D-ComfyUI
python -m pip install -r requirements.txt
python install.py --check
```

Restart ComfyUI, then run **Pixal3D Environment Check** before loading the model.

`requirements.txt` installs only safe Python packages. It deliberately does not install Torch, FlashAttention, Triton, Pixal3D CUDA kernels, or renderer kernels because those wheels must match your exact Python, PyTorch, CUDA, OS, and GPU.

## Required Pieces

A working generation environment needs these imports inside the same Python that launches ComfyUI:

| Piece | Required import or file | Notes |
|---|---|---|
| PyTorch CUDA | `torch.cuda.is_available() == True` | CPU-only is not supported |
| Attention | `flash_attn` or `flash_attn_interface` | FlashAttention 2 or 3 |
| Sparse GEMM | `flex_gemm_ap` or `flex_gemm` | Pixal3D CUDA kernel |
| Mesh ops | `cumesh_vb` or `cumesh` | Pixal3D CUDA kernel |
| Voxel/export ops | `o_voxel_vb_ap` or `o_voxel` | Pixal3D CUDA kernel |
| DRTK | `drtk` | UV/export helper |
| Pixal3D model | `ComfyUI/models/Pixal3D/TencentARC_Pixal3D/pipeline.json` | Download manually or use `download_if_missing=true` |
| DINOv3 helper | `ComfyUI/models/Pixal3D/camenduru_dinov3-vitl16-pretrain-lvd1689m/` | Needed by the image encoder |
| MoGe | `ComfyUI/models/geometry_estimation/moge_2_vitl_normal_fp16.safetensors` | Only needed for `camera_mode=moge` |
| RMBG-2.0 | `ComfyUI/models/Pixal3D/briaai_RMBG-2.0/` | Gated model; only needed for `background_mode=auto_remove` |
| NATTEN/libnatten | `natten.HAS_LIBNATTEN == True` | Only needed for strict NAF |

If Environment Check says a CUDA package is missing, install a wheel that exactly matches your stack. Do not let pip replace a working Torch install while testing random wheels; use `--no-deps` for manual CUDA wheels.

## Windows Wheel Order

On Windows, install wheels in this order:

1. PyTorch with CUDA for your GPU/driver.
2. Required Pixal3D CUDA wheels: `flex_gemm_ap`, `cumesh_vb`, `o_voxel_vb_ap`, and `drtk`.
3. One attention wheel: FlashAttention 2 (`flash_attn`) or FlashAttention 3 (`flash_attn_interface`).
4. Optional strict NAF wheel: NATTEN with CUDA `libnatten`.

The required Pixal3D CUDA wheels are separate from NATTEN. A working NATTEN install does not mean `flex_gemm`, `cumesh`, `o_voxel`, or `drtk` are installed.

For Python 3.12, PyTorch 2.10, CUDA 13.0 on Blackwell sm120, install the required Pixal3D CUDA wheels plus the prebuilt NATTEN/libnatten wheel with:

```bat
venv\Scripts\python.exe -m pip install --no-deps ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/flex_gemm_ap-latest/flex_gemm_ap-1.0.0%2Bcu130torch2.10-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/cumesh_vb-latest/cumesh_vb-1.0%2Bcu130torch2.10-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/o_voxel_vb_ap-latest/o_voxel_vb_ap-0.0.1%2Bcu130torch2.10-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/drtk-latest/drtk-0.1.0%2Bcu130torch2.10-cp312-cp312-win_amd64.whl" ^
  "https://huggingface.co/drbaph/NATTEN-0.21.6-torch2100cu130-cp312-cp312-win_amd64/resolve/main/natten-0.21.6+torch2100cu130-cp312-cp312-win_amd64.whl"
```

If your Python, PyTorch, CUDA, or GPU architecture does not match that NATTEN wheel, omit the final NATTEN URL and use `naf_mode=fallback_if_missing`, `preload_naf=false`.

For Python 3.12, PyTorch 2.8, CUDA 12.8 on Blackwell sm100/sm120, use the matching Pixal3D CUDA wheels plus the `naxneri` NATTEN/libnatten wheel:

```bat
venv\Scripts\python.exe -m pip install --no-deps ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/flex_gemm_ap-latest/flex_gemm_ap-1.0.0%2Bcu128torch2.8-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/cumesh_vb-latest/cumesh_vb-1.0%2Bcu128torch2.8-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/o_voxel_vb_ap-latest/o_voxel_vb_ap-0.0.1%2Bcu128torch2.8-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/drtk-latest/drtk-0.1.0%2Bcu128torch2.8-cp312-cp312-win_amd64.whl" ^
  "https://huggingface.co/naxneri/natten-0.21.6-blackwell-cu128-cp312-cp312-win_amd64/resolve/main/natten-0.21.6-blackwell-cu128-cp312-cp312-win_amd64.whl"
```

For PyTorch 2.9 or another CUDA 12.8 stack, change the four Pozzetti URLs to wheels built for that exact Torch version. Keep the NATTEN URL only when it matches your Python, CUDA, and GPU.

More detail: [Windows wheel guide](docs/windows_wheels.md).

## Windows NATTEN / NAF

Pixal3D uses **NAF** as a feature refinement step for the shape and texture stages. NAF uses NATTEN. Strict upstream NAF only works when NATTEN includes CUDA `libnatten`:

```bat
python -c "import natten; print(natten.__version__, natten.HAS_LIBNATTEN)"
```

If that prints `False`, you have normal NATTEN without CUDA libnatten. The node can still run, but you must use:

```text
Pixal3D Model Loader naf_mode=fallback_if_missing
Pixal3D Model Loader preload_naf=false
```

Fallback mode avoids loading NAF and keeps the expected tensor shape by using DINO projection features. It is usually slower and may use more RAM/VRAM than a proper CUDA NATTEN/libnatten build, and quality can be lower than strict upstream NAF.

On Windows, a NATTEN wheel must match all of these:

```text
Python ABI, for example cp312
PyTorch build, for example torch2.10
CUDA build, for example cu130
GPU architecture, for example sm120
OS tag, win_amd64
```

If you cannot find a matching Windows wheel, use fallback mode or build NATTEN from source.

Known community Windows NATTEN wheels:

| Python | PyTorch | CUDA | GPU | Wheel |
|---|---|---|---|---|
| 3.12 | 2.10 | 13.0 | Blackwell sm120 | [drbaph/NATTEN-0.21.6-torch2100cu130-cp312-cp312-win_amd64](https://huggingface.co/drbaph/NATTEN-0.21.6-torch2100cu130-cp312-cp312-win_amd64) |
| 3.12 | 2.8+ | 12.8 | Blackwell sm100/sm120 | [naxneri/natten-0.21.6-blackwell-cu128-cp312-cp312-win_amd64](https://huggingface.co/naxneri/natten-0.21.6-blackwell-cu128-cp312-cp312-win_amd64) |

More detail: [Windows wheel guide](docs/windows_wheels.md) and [Build NATTEN on Windows](docs/Build_Natten_windows.md).

## Model Folders

Default model layout:

```text
ComfyUI/models/
├── Pixal3D/
│   ├── TencentARC_Pixal3D/
│   │   ├── pipeline.json
│   │   └── ckpts/*.safetensors
│   ├── camenduru_dinov3-vitl16-pretrain-lvd1689m/
│   └── briaai_RMBG-2.0/
└── geometry_estimation/
    ├── moge_1_vitl_fp16.safetensors
    └── moge_2_vitl_normal_fp16.safetensors
```

`download_if_missing=false` is the default. Turn it on only if you want the node to download helper models. `hf_endpoint` can be changed to a Hugging Face mirror if needed.

RMBG-2.0 is gated on Hugging Face. Accept the model terms and log in, set `HF_TOKEN`, or provide a transparent PNG/WebP and use `background_mode=keep_alpha` so RMBG is not needed.

## Recommended Loader Settings

General Windows baseline:

| Node | Setting |
|---|---|
| Pixal3D Model Loader | `attention_backend=auto` |
| Pixal3D Model Loader | `vram_mode=dynamic_vram` |
| Pixal3D Model Loader | `naf_mode=fallback_if_missing` unless `natten.HAS_LIBNATTEN=True` |
| Pixal3D Model Loader | `preload_naf=false` unless strict NAF works |
| Pixal3D Image To 3D | `pipeline_type=1024_cascade` for lower VRAM, `1536_cascade` for quality |
| Pixal3D Export GLB | `decimation_target=1000000`, `texture_size=4096` |

Lowest-VRAM/manual path:

| Node | Setting |
|---|---|
| Pixal3D Model Loader | `vram_mode=native_low_vram` |
| Pixal3D Model Loader | `load_moge=false` |
| Pixal3D Model Loader | `load_rembg=false` |
| Pixal3D Image To 3D | `camera_mode=manual` |
| Pixal3D Image To 3D | `background_mode=keep_alpha` with transparent PNG/WebP |
| Pixal3D Camera Control | Connect `manual_fov` to `Pixal3D Image To 3D.manual_fov` |

`native_low_vram` moves Pixal3D stages between CPU and GPU as needed. It can reduce VRAM pressure, but it is slower and needs a lot of system RAM, often 20-40 GB.

## Nodes

| Node | Purpose |
|---|---|
| Pixal3D Environment Check | Prints installed/missing dependencies |
| Pixal3D Model Loader | Loads Pixal3D and helper models |
| Pixal3D Camera Control | Manual FOV, distance, and mesh scale with Scene/POV preview |
| Pixal3D Image To 3D | Runs image-to-3D generation |
| Pixal3D Export GLB | Exports the result to textured `.glb` |
| Pixal3D Unload Model | Clears the Pixal3D pipeline cache and releases the model handle |

Basic workflow:

```text
Load Image -> Pixal3D Image To 3D image
Pixal3D Model Loader -> Pixal3D Image To 3D model
Pixal3D Image To 3D -> Pixal3D Export GLB
Pixal3D Export GLB glb_path -> Preview 3D & Animation model_file
```

Manual camera workflow:

```text
Load Image -> Pixal3D Camera Control image
Pixal3D Camera Control manual_fov -> Pixal3D Image To 3D manual_fov
Pixal3D Image To 3D camera_mode=manual
```

## Troubleshooting Shortcuts

| Symptom | Fix |
|---|---|
| `No module named flash_attn` | Install a matching FlashAttention 2 wheel, or FlashAttention 3 with `flash_attn_interface` |
| `flex_gemm`, `cumesh`, `o_voxel`, or `drtk` missing | Install matching Pixal3D CUDA wheels for your Python/PyTorch/CUDA/OS |
| `natten.HAS_LIBNATTEN=False` | Use `naf_mode=fallback_if_missing`, `preload_naf=false`, or install/build CUDA NATTEN |
| RMBG download fails | Accept gated model terms, log in, set `HF_TOKEN`, or use transparent input with `keep_alpha` |
| MoGe missing | Download Comfy-Org/MoGe files to `ComfyUI/models/geometry_estimation/` or use manual camera mode |
| GLB looks fragmented | Try `remesh=false`; keep `decimation_target=1000000` or higher |
| RAM stays high after unload | Use Pixal3D Unload Model; restart ComfyUI to return all reserved Python/PyTorch memory to the OS |

See [Troubleshooting](docs/troubleshooting.md) for longer explanations.

## Useful Links

- [Windows wheel guide](docs/windows_wheels.md)
- [Build NATTEN on Windows](docs/Build_Natten_windows.md)
- [Linux/WSL CUDA guide](docs/linux_wsl_cuda.md)
- [Portable/standalone install](docs/portable_standalone_install.md)
- [Compatibility matrix](docs/compatibility_matrix.md)
- [Related repositories](docs/related_repos.md)

## Acknowledgements

This nodepack builds on [TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D), [Trellis.2](https://github.com/microsoft/TRELLIS.2), [Trellis](https://github.com/microsoft/TRELLIS), and [Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2).

If Pixal3D is useful in your work, please cite the upstream project:

```bibtex
@article{li2026pixal3d,
    title={Pixal3D: Pixel-Aligned 3D Generation from Images},
    author={Li, Dong-Yang and Zhao, Wang and Chen, Yuxin and Hu, Wenbo and Guo, Meng-Hao and Zhang, Fang-Lue and Shan, Ying and Hu, Shi-Min},
    journal={arXiv preprint arXiv:2605.10922},
    year={2026}
}
```
