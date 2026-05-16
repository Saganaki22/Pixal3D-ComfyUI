# Troubleshooting

This document separates two things that often get mixed together:

- What Pixal3D-ComfyUI can safely install from `requirements.txt`.
- What must come from exact CUDA wheels or a local source build.

The short version: Pixal3D-ComfyUI can run on Windows, including current CUDA/PyTorch stacks, but not every upstream setting is available on every Windows environment. The exact upstream NAF path requires CUDA NATTEN/libnatten. If a matching Windows wheel is not available, use `naf_mode=fallback_if_missing` or build NATTEN from source.

## Windows Reality Check

Upstream NATTEN strongly recommends NVIDIA users install NATTEN with the CUDA kernel library `libnatten`. Their install docs point Linux or WSL NVIDIA users to prebuilt PyPI wheels as the fastest path, while Windows users are directed to MSVC source builds and warned that Windows builds are experimental and not regularly tested.

Relevant upstream docs:

- NATTEN install guide: https://natten.org/install/
- NATTEN GitHub: https://github.com/SHI-Labs/NATTEN

What that means here:

| Setting/path | Windows status |
|--------------|----------------|
| `naf_mode=fallback_if_missing` | Supported and recommended when CUDA NATTEN is missing |
| `naf_mode=strict` | Requires `natten.HAS_LIBNATTEN == True` |
| Plain `pip install natten` | Usually not enough for NAF if it installs `py3-none-any` |
| Official `whl.natten.org` command | Good to try, but may not provide `win_amd64` for your exact stack |
| Building NATTEN from source | Possible, but requires MSVC, CUDA Toolkit, CMake, Ninja, and patience |

## Required Pieces

`requirements.txt` covers runtime packages only. It deliberately does not include binary CUDA packages. It does include plain `natten==0.21.6` as a baseline import/runtime dependency, but that package may not include CUDA `libnatten`.

Required from the ComfyUI environment:

```text
torch
torchvision
```

Required CUDA extension imports:

```text
flash_attn or flash_attn_interface
triton or triton-windows
flex_gemm_ap or flex_gemm
cumesh_vb or cumesh
o_voxel_vb_ap or o_voxel
drtk
```

Optional CUDA extension imports:

```text
nvdiffrast
nvdiffrec_render
natten with libnatten
```

Run **Pixal3D Environment Check** in ComfyUI first. A production-ready local install means the required CUDA imports pass in the same Python environment that starts ComfyUI.

Manual wheel notes live in:

- [requirements-cuda-manual.txt](../requirements-cuda-manual.txt)
- [Linux / WSL CUDA Requirements](linux_wsl_cuda.md)
- [Windows wheel guide](windows_wheels.md)
- [Compatibility matrix](compatibility_matrix.md)

## Recommended Windows Settings

These settings are the stable Windows baseline when a real CUDA NATTEN/libnatten wheel is not installed:

| Node | Setting | Value |
|------|---------|-------|
| Pixal3D Model Loader | `attention_backend` | `auto` |
| Pixal3D Model Loader | `vram_mode` | `dynamic_vram` |
| Pixal3D Model Loader | `download_if_missing` | `false` unless you want node downloads |
| Pixal3D Model Loader | `load_moge` | `true` |
| Pixal3D Model Loader | `load_rembg` | `true` only if using `background_mode=auto_remove` |
| Pixal3D Model Loader | `naf_mode` | `fallback_if_missing` |
| Pixal3D Model Loader | `naf_target_size` | `upstream` |
| Pixal3D Image To 3D | `pipeline_type` | `1536_cascade` for quality, `1024_cascade` for lower VRAM |
| Pixal3D Image To 3D | `background_mode` | `keep_alpha` for clean transparent PNGs, otherwise `auto_remove` |
| Pixal3D Export GLB | `decimation_target` | `1000000` |
| Pixal3D Export GLB | `texture_size` | `4096` |
| Pixal3D Export GLB | `remesh` | Default `true`; if cleanup fragments the mesh, try `false` |

## Common Errors

### `No module named 'natten'`

NAF tried to import NATTEN and it is not installed. You have three choices:

1. Use `naf_mode=fallback_if_missing`.
2. Install a matching CUDA NATTEN/libnatten wheel.
3. Build NATTEN from source.

For Windows, option 1 is the expected default unless you already know you have a working NATTEN CUDA build.

Some other Pixal3D wrappers put plain `natten` in `requirements.txt`. Pixal3D-ComfyUI does that too now, because it is useful as a basic package and can support NATTEN's non-libnatten paths where available. It still does not guarantee the CUDA `libnatten` kernels that Pixal3D's strict NAF path needs.

### `natten is installed, but HAS_LIBNATTEN is false`

This means Python can import `natten`, but it does not have the CUDA kernel library needed by Pixal3D's real NAF path.

Check it directly:

```bat
venv\Scripts\python.exe -c "import natten; print(natten.__version__, natten.HAS_LIBNATTEN)"
```

If it prints `False`, use:

```text
naf_mode=fallback_if_missing
```

Do not keep reinstalling plain `natten==0.21.6` and expect strict NAF to start working. If pip installs a generic `py3-none-any` package, it is not the CUDA build.

### NATTEN wheel tag for Torch 2.10 + CUDA 13.0

The upstream command for that Linux/WSL stack is:

```bat
venv\Scripts\python.exe -m pip install --no-deps "natten==0.21.6+torch2100cu130" -f https://whl.natten.org
```

After installing, verify:

```bat
venv\Scripts\python.exe -c "import natten; print(natten.__version__, natten.HAS_LIBNATTEN)"
```

`HAS_LIBNATTEN` must be `True` for `naf_mode=strict`.

The guarded installer can apply these official Linux/WSL NATTEN tags when requested:

```bash
PIXAL3D_INSTALL_NATTEN=1 ./venv/bin/python custom_nodes/Pixal3D-ComfyUI/install.py
```

On native Windows it will not do this automatically because upstream points Windows users to MSVC source builds.

### Can `comfy-sparse-attn` replace NATTEN or NAF?

No. `comfy-sparse-attn` is useful for ComfyUI sparse/variable-length attention dispatch and for staging `comfy.sparse`, `comfy.ops_sparse`, and `comfy.attention_sparse` namespace files. It is not a NATTEN replacement and it is not a NAF upsampler.

Pixal3D's NAF path imports NATTEN-style neighborhood attention APIs. `comfy-sparse-attn` dispatches varlen attention to available attention backends; it does not provide `natten.functional.na2d_qk`, `natten.functional.na2d_av`, `natten.na2d`, or CUDA `libnatten`.

So:

| Package | Can replace NATTEN for Pixal3D NAF? | Why |
|---------|-------------------------------------|-----|
| `natten` py3-none-any | No | Imports may exist, but `HAS_LIBNATTEN` is usually `False` |
| `natten + libnatten` matching your stack | Yes | Provides CUDA kernels needed for strict NAF |
| `comfy-sparse-attn` | No | Different API and purpose; sparse/varlen attention dispatch, not NAF |

It may become useful in a future deeper refactor if Pixal3D's sparse attention is rewritten around Comfy's sparse primitives, but it is not a drop-in fix for NAF quality today.

### NATTEN source build fails with CMake, Ninja, CXX, or CUDA compiler errors

Those errors usually mean the build tools are not visible to the build subprocess.

You generally need:

```text
Visual Studio Build Tools with Desktop development with C++
x64 Native Tools Command Prompt for Visual Studio
CUDA Toolkit with nvcc
CMake
Ninja
Git with recursive submodules
ComfyUI venv active or on PATH
```

Common checks:

```bat
where cl
where nvcc
where cmake
where ninja
venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda)"
```

Use the architecture format expected by your NATTEN version. Current upstream examples use dotted forms such as `10.0;10.3` for Blackwell-class builds, and the setup may print an internal multiplied form during compilation. Do not guess blindly from random snippets. Follow the NATTEN docs for the exact release you are building.

On Windows, upstream's MSVC path is:

```bat
git clone --recursive https://github.com/SHI-Labs/NATTEN
cd NATTEN
WindowsBuilder.bat install
```

If this succeeds, run the `HAS_LIBNATTEN` check above before using `naf_mode=strict`.

### RMBG-2.0 will not download

`briaai/RMBG-2.0` is a gated Hugging Face model. You must accept its terms on Hugging Face and either:

- log in before launching ComfyUI,
- set `HF_TOKEN`, or
- manually place the model snapshot in `ComfyUI/models/Pixal3D/briaai_RMBG-2.0/`.

If you do not want RMBG, set:

```text
load_rembg=false
background_mode=none
```

or provide a transparent PNG and use:

```text
background_mode=keep_alpha
```

### MoGe says `Ruicheng/moge-2-vitl` or asks for a snapshot folder

Newer Pixal3D-ComfyUI uses native ComfyUI MoGe files only:

```text
ComfyUI/models/moge/moge_1_vitl_fp16.safetensors
ComfyUI/models/moge/moge_2_vitl_normal_fp16.safetensors
```

Download source:

```text
https://huggingface.co/Comfy-Org/MoGe
```

If the old `moge_repo` field still appears in the node UI, restart ComfyUI and hard refresh the browser. The node schema is cached in the running ComfyUI frontend.

### MoGe camera fit looks slightly off

MoGe is used for camera/FOV estimation only. It does not improve mesh detail, texture quality, or NAF behavior.

Pixal3D-ComfyUI uses the native ComfyUI `moge_2_vitl_normal_fp16.safetensors` file. If that file did not match the expected MoGe v2 ViT-L architecture, loading would fail with missing or unexpected keys instead of silently producing a different camera.

If the subject looks framed wrong, too close, too far, or slightly distorted, compare against manual camera mode:

```text
camera_mode=manual
manual_camera_angle_x=0.858
manual_distance=2.0
mesh_scale=1.0
extend_pixel=0
```

If manual mode looks better, MoGe is not broken; the automatic camera fit is just not matching that input image. Keep manual mode for that image or adjust `extend_pixel`, `manual_camera_angle_x`, and `manual_distance`.

### Model download created `blobs` or cache-style folders

Hugging Face cache folders are not the same thing as clean model folders. A cache path can contain `blobs`, refs, and snapshot hashes. Pixal3D-ComfyUI expects normal model files, for example:

```text
pipeline.json
model_index.json
*.safetensors
ckpts/*.safetensors
```

For MoGe, the correct flat target is:

```text
ComfyUI/models/moge/
```

For Pixal3D helpers, the target is:

```text
ComfyUI/models/Pixal3D/
```

### `hf_endpoint` downloads from a mirror

`hf_endpoint` is the base host passed to Hugging Face download helpers. Use:

```text
https://huggingface.co
```

unless you intentionally want a mirror such as:

```text
https://hf-mirror.com
```

If you put only an owner/repo string where a full endpoint URL is expected, Hugging Face helpers can combine paths in surprising ways. Keep the endpoint as a full URL.

### GLB opens in Blender but not Windows 3D Viewer

Older exports used WebP textures and `EXT_texture_webp`. Some Windows viewers reject GLBs that require that extension.

Current Pixal3D-ComfyUI exports embedded PNG textures for better Windows viewer compatibility. Re-export the GLB after updating the node.

### GeometryPack preview creates `.vtp`

That is from GeometryPack's preview worker, not Pixal3D Export GLB. VTP is a VTK mesh format used by that preview path. For normal ComfyUI 3D preview, use:

```text
Pixal3D Export GLB glb_path
  -> Preview 3D & Animation model_file
```

Pixal3D Image To 3D now returns only `pixal3d_result`; it does not build a separate `TRIMESH` preview object.

### Mesh has holes or strange artifacts

Likely causes:

- input image has unclear silhouette, shadows, or background leakage,
- RMBG cutout is damaging the subject,
- `naf_mode=fallback_if_missing` is being used because strict NAF is unavailable,
- decimation/remesh settings are too aggressive,
- the source image is stylized or has occluded parts that the model has to guess.

Try:

```text
pipeline_type=1536_cascade
steps=18 to 24
max_num_tokens=98304 if VRAM allows
background_mode=keep_alpha with a clean transparent PNG
decimation_target=1000000
texture_size=4096
remesh=false if you suspect mesh cleanup is causing damage
```

If strict NAF is important for quality, you need a working CUDA NATTEN/libnatten build. Without that, Windows can still run the node, but it is using the fallback feature path.

### MemoryVisualization squares are not fully colored

Pixal3D uses upstream `torch.nn.Module` models and a Comfy wrapper. The wrapper lets ComfyUI manage the model as one loaded object, but it does not turn every internal tensor into Comfy's native per-layer VBAR objects.

That means MemoryVisualization may show the Pixal3D wrapper and staged memory, but not the same colored per-block view as a nodepack built entirely through Comfy native ops. This is expected unless the upstream model is refactored deeper into Comfy's ops system.

### CPU RAM stays high after ComfyUI native unload

ComfyUI's native unload buttons primarily target Comfy-managed VRAM/model entries. Pixal3D-ComfyUI also keeps a Python-side pipeline cache so the 18GB+ Pixal3D stack does not reload from disk every prompt.

Current behavior:

- Changing **Pixal3D Model Loader** settings unloads and destroys stale cached handles.
- **Pixal3D Unload Model** removes the active handle from both Comfy model management and the Pixal3D Python cache.
- Successful runs drop the huge shape/texture latents from `pixal3d_result`; export only needs the decoded mesh.
- CUDA OOMs offload the active handle before re-raising the error.

Pixal3D-ComfyUI hooks ComfyUI's global unload button, so native unloads also clear the active Pixal3D Python cache. The **Pixal3D Unload Model** node does the same thing from inside a workflow.

Task Manager can still show high RAM after this because Python, PyTorch, memory-mapped safetensors, Hugging Face/Transformers imports, and Windows memory allocators may keep reserved pages for reuse. That does not necessarily mean the Pixal3D pipeline is still referenced. Restart ComfyUI when you need every reserved page returned to the OS immediately.

### OOM at decode after long 1536 runs

Pixal3D loads checkpoints through CPU RAM first, then moves modules to GPU when they run. That is normal PyTorch/Hugging Face behavior. The heavy part is that the Pixal3D pipeline contains several large stages at once: sparse structure, shape 512, shape 1024/1536, texture, decoders, plus four DINOv3 projection feature extractors.

Use `vram_mode=dynamic_vram` first on a 32GB card. Pixal3D-ComfyUI builds standard Pixal3D layers with Comfy/Aimdo-aware ops where possible, then wraps the pipeline in ComfyUI model management. Pixal3D is still not a fully Comfy-native model, so custom sparse modules and temporary tensors can still force-load or spike VRAM.

If Comfy logs a large `Force pre-loaded` value or a 1536 run OOMs late in decode, switch to `vram_mode=native_low_vram`. That mode can run smaller workflows in low VRAM ranges such as 4-8 GB VRAM, but it trades that for much higher host memory use: plan for 20-40 GB system RAM and slower runs. It bypasses Comfy's bulk model load and lets Pixal3D move stages to GPU one at a time and back to CPU afterwards. RMBG is staged only for background removal, MoGe is staged only for camera estimation, and Pixal3D's flow/decoder modules are staged by the upstream low-VRAM pipeline.

Lowest-VRAM recipe:

```text
Pixal3D Model Loader vram_mode=native_low_vram
Pixal3D Model Loader load_moge=false
Pixal3D Model Loader load_rembg=false
Pixal3D Image To 3D camera_mode=manual
Pixal3D Image To 3D background_mode=keep_alpha
Pixal3D Camera Control manual_fov -> Pixal3D Image To 3D manual_fov
```

Use a transparent-background PNG or WebP for this path. That avoids the RMBG helper model, and Camera Control avoids loading MoGe. The Camera Control node only applies in manual camera mode; when its `manual_fov` output is connected, it overrides the scalar `manual_camera_angle_x`, `manual_distance`, and `mesh_scale` inputs on `Pixal3D Image To 3D`.

A `1536_cascade` run with high token counts can still exceed 32GB VRAM during decode because temporary tensors, sparse latents, decoded voxels, xatlas data, and GLB/preview data are separate from model weights.

For the local high-VRAM Windows stack, if a 1536 run OOMs after HR shape/texture sampling, try:

```text
pipeline_type=1024_cascade
steps=12 to 18
max_num_tokens=32768 or 49152
force_offload=true after generation
Pixal3D Export GLB decimation_target=1000000
Pixal3D Export GLB remesh=false
```

If 1536 is required, close other GPU-heavy nodes/apps first and use **Pixal3D Unload Model** between experiments that change loader settings.

## Before Filing A Bug

Attach:

```bat
venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))"
venv\Scripts\python.exe -c "import natten; print(getattr(natten, '__version__', 'missing'), getattr(natten, 'HAS_LIBNATTEN', None))"
venv\Scripts\python.exe -c "import flash_attn; print(flash_attn.__version__)"
```

Also include:

- Pixal3D Environment Check output.
- The exact node settings.
- The first real error traceback, not only the later cascade errors.
- Whether the GLB was exported before or after the PNG texture compatibility fix.
