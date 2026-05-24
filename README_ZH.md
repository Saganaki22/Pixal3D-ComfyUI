# Pixal3D-ComfyUI 中文说明

TencentARC [Pixal3D](https://github.com/TencentARC/Pixal3D) 的 ComfyUI 节点：单图生成 3D、导出带贴图的 `.glb`、支持 FlashAttention 2/3、手动相机、DynamicVRAM 和模型卸载。

[English README](README.md) | [Windows 轮子](docs/windows_wheels.md) | [Windows 构建 NATTEN](docs/Build_Natten_windows.md) | [故障排查](docs/troubleshooting.md) | [兼容性矩阵](docs/compatibility_matrix.md)

![Pixal3D preview](https://github.com/user-attachments/assets/45d596b4-9070-44d2-8e4f-1019169d3daa)

## 快速安装

ComfyUI Manager：

1. 打开 **ComfyUI Manager**。
2. 搜索 **Pixal3D**，作者选择/确认 **Saganaki22**。
3. 安装普通/稳定版本。不要安装 Nightly 节点。
4. 重启 ComfyUI，然后先运行 **Pixal3D Environment Check**。

手动安装，在启动 ComfyUI 的同一个 Python 环境里执行：

```bat
cd ComfyUI\custom_nodes
git clone https://github.com/Saganaki22/Pixal3D-ComfyUI.git
cd Pixal3D-ComfyUI
python -m pip install -r requirements.txt
python install.py --check
```

重启 ComfyUI，然后先运行 **Pixal3D Environment Check**。

`requirements.txt` 只安装安全的 Python 依赖。它不会自动安装 Torch、FlashAttention、Triton、Pixal3D CUDA kernels、renderer kernels 或 NATTEN/libnatten，因为这些轮子必须匹配你的 Python、PyTorch、CUDA、Windows 和 GPU。

## 必需组件

同一个 ComfyUI Python 环境里需要这些内容：

| 组件 | 需要能导入/找到 | 说明 |
|---|---|---|
| PyTorch CUDA | `torch.cuda.is_available() == True` | 不支持纯 CPU |
| Attention | `flash_attn` 或 `flash_attn_interface` | FlashAttention 2 或 3 |
| Sparse GEMM | `flex_gemm_ap` 或 `flex_gemm` | Pixal3D CUDA kernel |
| Mesh ops | `cumesh_vb` 或 `cumesh` | Pixal3D CUDA kernel |
| Voxel/export ops | `o_voxel_vb_ap` 或 `o_voxel` | Pixal3D CUDA kernel |
| DRTK | `drtk` | UV/export helper |
| Pixal3D 模型 | `ComfyUI/models/Pixal3D/TencentARC_Pixal3D/pipeline.json` | 可手动下载或开启 `download_if_missing=true` |
| DINOv3 helper | `ComfyUI/models/Pixal3D/camenduru_dinov3-vitl16-pretrain-lvd1689m/` | image encoder 需要 |
| MoGe | `ComfyUI/models/geometry_estimation/moge_2_vitl_normal_fp16.safetensors` | 只在 `camera_mode=moge` 时需要 |
| RMBG-2.0 | `ComfyUI/models/Pixal3D/briaai_RMBG-2.0/` | gated model；只在 `background_mode=auto_remove` 时需要 |
| NATTEN/libnatten | `natten.HAS_LIBNATTEN == True` | 只在 strict NAF 时需要 |

手动安装 CUDA 轮子时使用 `--no-deps`，避免 pip 替换已经可用的 Torch。

## Windows 轮子顺序

Windows 上建议按这个顺序处理：

1. 先安装匹配的 PyTorch CUDA。
2. 安装 Pixal3D 必需 CUDA 轮子：`flex_gemm_ap`、`cumesh_vb`、`o_voxel_vb_ap`、`drtk`。
3. 安装一个 attention 轮子：FlashAttention 2 (`flash_attn`) 或 FlashAttention 3 (`flash_attn_interface`)。
4. 如果要 strict NAF，再安装带 CUDA `libnatten` 的 NATTEN。

Pixal3D CUDA 轮子和 NATTEN 是两件事。NATTEN 装好了不代表 `flex_gemm`、`cumesh`、`o_voxel`、`drtk` 已经装好。

Python 3.12、PyTorch 2.10、CUDA 13.0、Blackwell sm120 示例，包含 Pixal3D 必需 CUDA 轮子和预编译 NATTEN/libnatten 轮子：

```bat
venv\Scripts\python.exe -m pip install --no-deps ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/flex_gemm_ap-latest/flex_gemm_ap-1.0.0%2Bcu130torch2.10-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/cumesh_vb-latest/cumesh_vb-1.0%2Bcu130torch2.10-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/o_voxel_vb_ap-latest/o_voxel_vb_ap-0.0.1%2Bcu130torch2.10-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/drtk-latest/drtk-0.1.0%2Bcu130torch2.10-cp312-cp312-win_amd64.whl" ^
  "https://huggingface.co/drbaph/NATTEN-0.21.6-torch2100cu130-cp312-cp312-win_amd64/resolve/main/natten-0.21.6+torch2100cu130-cp312-cp312-win_amd64.whl"
```

如果你的 Python、PyTorch、CUDA 或 GPU 架构不匹配这个 NATTEN 轮子，删掉最后一个 NATTEN URL，并使用 `naf_mode=fallback_if_missing`、`preload_naf=false`。

Python 3.12、PyTorch 2.8、CUDA 12.8、Blackwell sm100/sm120 示例，包含匹配的 Pixal3D CUDA 轮子和 `naxneri` NATTEN/libnatten 轮子：

```bat
venv\Scripts\python.exe -m pip install --no-deps ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/flex_gemm_ap-latest/flex_gemm_ap-1.0.0%2Bcu128torch2.8-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/cumesh_vb-latest/cumesh_vb-1.0%2Bcu128torch2.8-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/o_voxel_vb_ap-latest/o_voxel_vb_ap-0.0.1%2Bcu128torch2.8-cp312-cp312-win_amd64.whl" ^
  "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/drtk-latest/drtk-0.1.0%2Bcu128torch2.8-cp312-cp312-win_amd64.whl" ^
  "https://huggingface.co/naxneri/natten-0.21.6-blackwell-cu128-cp312-cp312-win_amd64/resolve/main/natten-0.21.6-blackwell-cu128-cp312-cp312-win_amd64.whl"
```

如果使用 PyTorch 2.9 或其他 CUDA 12.8 栈，四个 Pozzetti URL 必须换成对应 Torch 版本的轮子。只有 Python、CUDA、GPU 匹配时才保留这个 NATTEN URL。

更多说明见 [Windows 轮子指南](docs/windows_wheels.md)。

## NATTEN / NAF

NAF 是 Pixal3D 的 shape/texture refinement 步骤，NAF 使用 NATTEN。strict NAF 需要 NATTEN 带 CUDA `libnatten`：

```bat
python -c "import natten; print(natten.__version__, natten.HAS_LIBNATTEN)"
```

如果结果是 `False`，说明你装的是普通 NATTEN，没有 CUDA libnatten。Windows 上必须使用：

```text
Pixal3D Model Loader naf_mode=fallback_if_missing
Pixal3D Model Loader preload_naf=false
```

fallback 模式可以运行，但通常更慢，RAM/VRAM 占用可能更高，质量也可能低于 strict NAF。

Windows NATTEN/libnatten 轮子必须精确匹配：

```text
Python ABI，例如 cp312
PyTorch 版本，例如 torch2.10
CUDA 版本，例如 cu130
GPU 架构，例如 sm120
Windows 标签，win_amd64
```

已知社区 Windows NATTEN 轮子：

| Python | PyTorch | CUDA | GPU | Wheel |
|---|---|---|---|---|
| 3.12.10 / 3.13.12 | 2.10 | 13.0 | Ampere sm86，RTX 3050-3090 Ti | [NeilsMabet/Natten-0.21.6-Amphere-wheel-windows](https://github.com/NeilsMabet/Natten-0.21.6-Amphere-wheel-windows) |
| 3.12 | 2.10 | 13.0 | Blackwell sm120 | [drbaph/NATTEN-0.21.6-torch2100cu130-cp312-cp312-win_amd64](https://huggingface.co/drbaph/NATTEN-0.21.6-torch2100cu130-cp312-cp312-win_amd64) |
| 3.12 | 2.8+ | 12.8 | Blackwell sm100/sm120 | [naxneri/natten-0.21.6-blackwell-cu128-cp312-cp312-win_amd64](https://huggingface.co/naxneri/natten-0.21.6-blackwell-cu128-cp312-cp312-win_amd64) |

如果没有匹配轮子，就使用 fallback 模式，或者按 [Build_Natten_windows.md](docs/Build_Natten_windows.md) 自己构建。

## 手动下载模型

如果 `download_if_missing=false`，请手动下载模型并放到下面这些目录。请下载完整 snapshot，不要只随便下载单个文件。

| 模型 | 下载链接 | 本地目录 | 什么时候需要 |
|---|---|---|---|
| Pixal3D | [TencentARC/Pixal3D](https://huggingface.co/TencentARC/Pixal3D) | `ComfyUI/models/Pixal3D/TencentARC_Pixal3D/` | 必需 |
| DINOv3 helper | [camenduru/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/camenduru/dinov3-vitl16-pretrain-lvd1689m) | `ComfyUI/models/Pixal3D/camenduru_dinov3-vitl16-pretrain-lvd1689m/` | 必需 |
| MoGe | [Comfy-Org/MoGe](https://huggingface.co/Comfy-Org/MoGe) | `ComfyUI/models/geometry_estimation/` | `camera_mode=moge` |
| RMBG-2.0 | [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) | `ComfyUI/models/Pixal3D/briaai_RMBG-2.0/` | `background_mode=auto_remove` |
| NAF upsampler | [valeoai/NAF](https://github.com/valeoai/NAF) | `ComfyUI/models/Pixal3D/torch_hub/` cache | 只在 strict NAF 时需要 |

RMBG-2.0 是 Hugging Face gated model。先申请权限并登录再下载。如果不想用 RMBG，请使用透明 PNG/WebP，并设置 `background_mode=keep_alpha`，或者使用 `background_mode=none`。

期望目录结构：

```text
ComfyUI/models/
├── Pixal3D/
│   ├── TencentARC_Pixal3D/
│   │   ├── pipeline.json
│   │   └── ckpts/
│   │       ├── *.json
│   │       └── *.safetensors
│   ├── camenduru_dinov3-vitl16-pretrain-lvd1689m/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── preprocessor_config.json
│   └── briaai_RMBG-2.0/
│       ├── config.json
│       ├── BiRefNet_config.py
│       ├── birefnet.py
│       ├── model.safetensors
│       └── preprocessor_config.json
└── geometry_estimation/
    ├── moge_1_vitl_fp16.safetensors
    └── moge_2_vitl_normal_fp16.safetensors
```

`Comfy-Org/MoGe` 里的 MoGe 文件要直接放在 `ComfyUI/models/geometry_estimation/`，不要再套一层 `Comfy-Org/MoGe` 文件夹。`hf_endpoint` 可以改成 Hugging Face 镜像。

## 推荐设置

Windows 通用设置：

| 节点 | 设置 |
|---|---|
| Pixal3D Model Loader | `attention_backend=auto` |
| Pixal3D Model Loader | `vram_mode=dynamic_vram` |
| Pixal3D Model Loader | `naf_mode=fallback_if_missing`，除非 `natten.HAS_LIBNATTEN=True` |
| Pixal3D Model Loader | `preload_naf=false`，除非 strict NAF 可用 |
| Pixal3D Image To 3D | `pipeline_type=1024_cascade` 省显存，`1536_cascade` 提高质量 |

最低显存/手动相机：

```text
Pixal3D Model Loader vram_mode=hybrid_low_vram
Pixal3D Model Loader load_moge=false
Pixal3D Model Loader load_rembg=false
Pixal3D Image To 3D camera_mode=manual
Pixal3D Image To 3D background_mode=keep_alpha
Pixal3D Camera Control manual_fov -> Pixal3D Image To 3D manual_fov
```

`hybrid_low_vram` 会保留 Pixal3D 分阶段 CPU/GPU offload，同时使用 Comfy/Aimdo-aware ops；如果某个环境不稳定，可退回 `native_low_vram`。

如需检查背景处理后的输入图，可将 `Pixal3D Image To 3D rembg_image` 连接到 `Preview Image`。

非 1:1 输入会自动 padding 成正方形，再进入 Pixal3D 的正方形 image encoder，所以 9:16 或 16:9 不会被拉伸。padding 发生在背景处理之后：

```text
auto_remove: 输入 -> RMBG/alpha 裁剪 -> padding 成正方形 -> RGB 图像送入 Pixal3D
keep_alpha: 透明输入 -> alpha 裁剪 -> padding 成正方形 -> RGB 图像送入 Pixal3D
none: 输入 -> 转成 RGB -> padding 成正方形 -> RGB 图像送入 Pixal3D
```

如果输入是透明图，并且不想使用 RMBG，请用 `background_mode=keep_alpha`。`background_mode=none` 会按设计忽略 alpha。

如果需要低面数导出，可以降低 **Pixal3D Export GLB** 的 `decimation_target`。默认是 `1000000`；现在可以设置到 `5000` 左右，但复杂模型会丢细节。

## 常见问题

| 问题 | 处理 |
|---|---|
| `flash_attn` 缺失 | 安装匹配的 FlashAttention 2 或 3 轮子 |
| `flex_gemm`、`cumesh`、`o_voxel`、`drtk` 缺失 | 安装匹配 Python/PyTorch/CUDA/Windows/GPU 的 Pixal3D CUDA 轮子 |
| `natten.HAS_LIBNATTEN=False` | 使用 `naf_mode=fallback_if_missing`、`preload_naf=false`，或安装/构建 CUDA NATTEN |
| strict NAF 在 12 GB 显存 OOM | 尝试 `vram_mode=hybrid_low_vram`，把 `naf_target_size` 降到 `256` 或 `128`，或使用 `naf_mode=fallback_if_missing` |
| RMBG 下载失败 | 接受模型条款、登录并设置 `HF_TOKEN`，或用透明输入和 `keep_alpha` |
| MoGe 缺失 | 下载 Comfy-Org/MoGe 到 `ComfyUI/models/geometry_estimation/`，或使用手动相机 |
| GLB 碎裂 | 尝试 `remesh=true`，保持 `decimation_target=1000000` 或更高 |

更多见 [故障排查](docs/troubleshooting.md)。
