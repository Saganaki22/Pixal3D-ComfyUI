# Troubleshooting

Run **Pixal3D Environment Check** first. It tells you which dependency is missing from the same Python environment that launches ComfyUI.

## Fast Fix Table

| Error or symptom | Meaning | Fix |
|---|---|---|
| `flash_attn` and `flash_attn_interface` missing | No attention backend | Install a matching FlashAttention 2 or 3 wheel |
| `flex_gemm` missing | Sparse GEMM kernel missing | Install a matching `flex_gemm_ap` or `flex_gemm` wheel |
| `cumesh` missing | Mesh kernel missing | Install a matching `cumesh_vb` or `cumesh` wheel |
| `o_voxel` missing | Voxel/export kernel missing | Install a matching `o_voxel_vb_ap` or `o_voxel` wheel |
| `drtk` missing | UV/export helper missing | Install a matching `drtk` wheel |
| `No module named natten` | NATTEN is not installed | Install `natten==0.21.6` or use fallback NAF mode |
| `natten.HAS_LIBNATTEN=False` | NATTEN imports but has no CUDA libnatten | Use `naf_mode=fallback_if_missing`, `preload_naf=false`, or install/build CUDA NATTEN |
| RMBG fails to download | `briaai/RMBG-2.0` is gated | Accept model terms, log in/set `HF_TOKEN`, or use transparent input with `keep_alpha` |
| MoGe missing | Auto camera model missing | Put Comfy-Org/MoGe files in `ComfyUI/models/geometry_estimation/`, or use manual camera mode |
| GLB opens in Blender but not Windows 3D Viewer | Old WebP texture export | Re-export with current Pixal3D-ComfyUI PNG texture export |
| Mesh is shards/point cloud | Remesh/decimation issue or bad cutout | Try `remesh=true`, `decimation_target=1000000`, clean transparent input |
| RAM stays high after unload | Python/PyTorch reserved memory | Use Pixal3D Unload Model; restart ComfyUI to return all reserved memory |

## NATTEN And NAF

NAF is Pixal3D's refinement step for shape and texture features. NAF needs NATTEN. Strict upstream NAF needs CUDA `libnatten`.

Check:

```bat
venv\Scripts\python.exe -c "import natten; print(natten.__version__, natten.HAS_LIBNATTEN)"
```

If `HAS_LIBNATTEN` is `False`, normal NATTEN is installed but strict NAF is not available. Use:

```text
Pixal3D Model Loader naf_mode=fallback_if_missing
Pixal3D Model Loader preload_naf=false
```

Do not keep reinstalling plain `natten==0.21.6` expecting strict NAF to become available. On Windows, strict NAF needs a `win_amd64` NATTEN wheel built for your exact Python, PyTorch, CUDA, and GPU architecture, or a local source build.

Fallback NAF mode lets generation continue, but it can be slower, use more RAM/VRAM, and produce lower quality than true CUDA NAF.

## Windows Wheel Rules

When fixing missing CUDA packages, match all of these:

```text
Python ABI, for example cp312
PyTorch build, for example torch2.10
CUDA build, for example cu130
OS, for example win_amd64
GPU architecture if the wheel is architecture-specific
```

Install manual CUDA wheels with `--no-deps` so pip does not replace Torch.

See [Windows wheel guide](windows_wheels.md).

## Lowest VRAM Setup

Use this when the model loads but VRAM is tight:

```text
Pixal3D Model Loader vram_mode=hybrid_low_vram
Pixal3D Model Loader load_moge=false
Pixal3D Model Loader load_rembg=false
Pixal3D Model Loader naf_mode=fallback_if_missing
Pixal3D Model Loader preload_naf=false
Pixal3D Image To 3D camera_mode=manual
Pixal3D Image To 3D background_mode=keep_alpha
Pixal3D Camera Control manual_fov -> Pixal3D Image To 3D manual_fov
```

Use a transparent PNG/WebP so RMBG is not needed. `hybrid_low_vram` combines Comfy/Aimdo-aware modules with native Pixal3D stage offload. If it behaves badly on a stack, fall back to `native_low_vram`. Plan for 20-40 GB RAM.

## Model Folder Problems

Pixal3D expects normal model folders, not raw Hugging Face blob cache folders.

Main model:

```text
ComfyUI/models/Pixal3D/TencentARC_Pixal3D/
```

MoGe:

```text
ComfyUI/models/geometry_estimation/
├── moge_1_vitl_fp16.safetensors
└── moge_2_vitl_normal_fp16.safetensors
```

RMBG:

```text
ComfyUI/models/Pixal3D/briaai_RMBG-2.0/
```

If `download_if_missing=true`, the node downloads into these folders. If it is `false`, it will not download helper models.

## Camera Issues

MoGe estimates camera/FOV only. It does not improve mesh detail or texture quality.

Non-square images are padded to square before Pixal3D's image encoder so the subject is not stretched. Padding is automatic and happens after background handling:

```text
auto_remove: input -> RMBG/alpha crop -> pad to square -> RGB image sent to Pixal3D
keep_alpha: transparent input -> alpha crop -> pad to square -> RGB image sent to Pixal3D
none: input -> convert to RGB -> pad to square -> RGB image sent to Pixal3D
```

For transparent images without RMBG, use `background_mode=keep_alpha`. `background_mode=none` ignores alpha by design.

If framing looks wrong, try manual camera mode:

```text
camera_mode=manual
manual_camera_angle_x=0.858
manual_distance=2.0
mesh_scale=1.0
```

Or connect **Pixal3D Camera Control** to `manual_fov`.

## Export Issues

Use these defaults first:

```text
Pixal3D Export GLB decimation_target=1000000
Pixal3D Export GLB texture_size=4096
```

For lower-poly exports, reduce `decimation_target`. Values around `5000` are allowed, but expect visible detail loss on complex geometry.

If the exported mesh looks fragmented, set:

```text
Pixal3D Export GLB remesh=true
```

## Before Filing A Bug

Include:

```bat
venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))"
venv\Scripts\python.exe -c "import natten; print(getattr(natten, '__version__', 'missing'), getattr(natten, 'HAS_LIBNATTEN', None))"
```

Also include:

- Pixal3D Environment Check output
- Exact node settings
- Full first traceback
- Whether you are using fallback NAF or strict NAF
- Whether the input image has a clean alpha channel
