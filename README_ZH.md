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

# Pixal3D-ComfyUI 中文说明

[English README](README.md) | [兼容性矩阵](docs/compatibility_matrix.md) | [便携版/独立版安装](docs/portable_standalone_install.md) | [Linux/WSL CUDA 指南](docs/linux_wsl_cuda.md) | [Windows 轮子指南](docs/windows_wheels.md) | [故障排查](docs/troubleshooting.md)

Pixal3D-ComfyUI 是 TencentARC Pixal3D 的 ComfyUI 节点封装，用于从单张图片生成带贴图的 3D 模型，并导出 `.glb` 文件。节点支持 FlashAttention 2/3 选择、ComfyUI DynamicVRAM/Aimdo 管理、原生 ComfyUI MoGe 权重路径，以及 Windows CUDA 扩展轮子的手动安装流程。

<p align="center">
  <img src="https://github.com/user-attachments/assets/45d596b4-9070-44d2-8e4f-1019169d3daa" width="1200"><br><br>

  <img src="https://github.com/user-attachments/assets/a2ef8b6e-ff68-4a81-a595-1e84eab2062c" width="800">
</p>

## 功能

- 在 ComfyUI 里直接运行 Pixal3D image-to-3D
- 从 Pixal3D voxel attributes 导出带贴图的 `.glb`
- FlashAttention 2 / FlashAttention 3 后端选择，`auto` 会自动选择可用后端
- ComfyUI 模型管理、卸载、DynamicVRAM、Aimdo/MemoryVisualization 可见性
- `native_low_vram` 模式，按阶段在 CPU/GPU 之间移动模型
- **Pixal3D Camera Control** 节点：手动控制 FOV、distance、mesh scale，并提供 Scene/POV 预览

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/8b53ffe6-115c-4ab2-9170-dc7e8f0e69aa" width="350"><br>
      <b>scene</b>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/ba6e7a92-89c8-449c-a81a-79a6d67c6011" width="350"><br>
      <b>POV</b>
    </td>
  </tr>
</table>


- `glb_path` 可直接连接到 ComfyUI 原生 **Preview 3D & Animation**

## 安装

在 ComfyUI 的 Python 环境里安装依赖，不要装到系统 Python。

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Saganaki22/Pixal3D-ComfyUI.git
cd Pixal3D-ComfyUI
python -m pip install -r requirements.txt
python install.py --check
```

也可以用包安装方式安装基础依赖：

```bash
python -m pip install .
uv pip install .
```

安装或更新后重启 ComfyUI。

本节点包含一个保守的 `install.py`，用于 ComfyUI Manager、Windows portable、普通 venv 和 Linux venv。默认只安装 runtime 依赖并打印环境检查，不会更改 PyTorch，也不会自动安装 CUDA 轮子，除非你显式开启已知精确匹配的轮子安装。

`requirements.txt` 只包含安全的 runtime 依赖。它故意不包含 `torch`、`torchvision`、`flash-attn`、`triton`、`flex_gemm`、`cumesh`、`o_voxel`、`drtk`、`nvdiffrast`。这些是二进制/CUDA 依赖，必须根据你的环境手动选择轮子或源码构建。参考 [requirements-cuda-manual.txt](requirements-cuda-manual.txt)、[便携版/独立版安装](docs/portable_standalone_install.md)、[Linux/WSL CUDA 指南](docs/linux_wsl_cuda.md) 和 [Windows 轮子指南](docs/windows_wheels.md)。

普通 `natten==0.21.6` 会作为基础依赖安装，和一些 Pixal3D-D 封装保持一致。但这不等于 strict NAF 可用。只有 `natten.HAS_LIBNATTEN == True` 时，才是真正带 CUDA libnatten 的 NAF 路径。

## 必需轮子和模型文件

`requirements.txt` 不是完整 Pixal3D 安装。真正能生成需要下面这些：

| 类型 | 必需项 | 需要能找到/导入什么 | 去哪里装 |
|------|--------|----------------------|----------|
| Attention 轮子 | FlashAttention 2 或 3 | 能导入 `flash_attn` 或 `flash_attn_interface` | [Windows 轮子指南](docs/windows_wheels.md#attention-wheels) 或 [Linux/WSL CUDA 指南](docs/linux_wsl_cuda.md#flashattention) |
| CUDA 轮子 | Sparse GEMM | 能导入 `flex_gemm_ap` 或 `flex_gemm` | [Windows 轮子指南](docs/windows_wheels.md#required-pixal3d-cuda-wheels) 或 [Linux/WSL CUDA 指南](docs/linux_wsl_cuda.md#required-pixal3d-cuda-extensions) |
| CUDA 轮子 | Mesh ops | 能导入 `cumesh_vb` 或 `cumesh` | 参考对应系统的轮子/编译指南 |
| CUDA 轮子 | Voxel/remesh ops | 能导入 `o_voxel_vb_ap` 或 `o_voxel` | 参考对应系统的轮子/编译指南 |
| CUDA 轮子 | DRTK | 能导入 `drtk` | 参考对应系统的轮子/编译指南 |
| CUDA/runtime 轮子 | Triton | 如果你的 attention/sparse 栈需要它，应能导入 `triton` | Windows 通常用 `triton-windows`；Linux 通常用匹配的 `triton` |
| 可选 renderer 轮子 | NVIDIA raster/render helper | 能导入 `nvdiffrast` / `nvdiffrec_render` | 仅在你的 renderer 路径需要时安装 |
| 可选 strict NAF 轮子 | CUDA NATTEN/libnatten | `natten.HAS_LIBNATTEN == True` | Linux/WSL 通常有官方轮子；Windows 多数情况用 fallback 或自行编译 |
| 主模型文件 | TencentARC Pixal3D | `ComfyUI/models/Pixal3D/TencentARC_Pixal3D/pipeline.json` 和 `ckpts/*.safetensors` | [TencentARC/Pixal3D](https://huggingface.co/TencentARC/Pixal3D) 或 `download_if_missing=true` |
| DINOv3 helper 文件 | Pixal3D image encoder | `ComfyUI/models/Pixal3D/camenduru_dinov3-vitl16-pretrain-lvd1689m/model.safetensors` | 开启 helper 下载，或手动放完整 snapshot |
| MoGe 相机文件 | 自动相机模式 | `ComfyUI/models/moge/moge_2_vitl_normal_fp16.safetensors` | [Comfy-Org/MoGe](https://huggingface.co/Comfy-Org/MoGe)，或改用 `camera_mode=manual` |
| RMBG 文件 | 内置背景移除 | `ComfyUI/models/Pixal3D/briaai_RMBG-2.0/` 完整 snapshot | [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) 是 gated model，先申请权限；也可用透明 PNG/WebP 跳过 |

最低显存/手动相机路线可以跳过 MoGe 和 RMBG：设置 `load_moge=false`、`load_rembg=false`，使用透明 PNG/WebP，`background_mode=keep_alpha`，并把 **Pixal3D Camera Control** 连接到 `manual_fov`。

## 我到底缺什么？

先运行 **Pixal3D Environment Check**，然后按第一条缺失信息查表：

| Environment Check 显示 | 代表什么 | 该怎么做 |
|------------------------|----------|----------|
| `flash_attn: MISSING` 且 `flash_attn_interface: MISSING` | 缺 attention 后端 | 安装匹配 Python/PyTorch/CUDA/系统的 FlashAttention 2 或 3 轮子 |
| `flex_gemm_ap: MISSING` 且 `flex_gemm: MISSING` | 缺 Pixal3D sparse GEMM 扩展 | 安装/编译匹配的 `flex_gemm_ap` 或 `flex_gemm` |
| `cumesh_vb: MISSING` 且 `cumesh: MISSING` | 缺 Pixal3D mesh 扩展 | 安装/编译匹配的 `cumesh_vb` 或 `cumesh` |
| `o_voxel_vb_ap: MISSING` 且 `o_voxel: MISSING` | 缺 Pixal3D voxel/remesh 扩展 | 安装/编译匹配的 `o_voxel_vb_ap` 或 `o_voxel` |
| `drtk: MISSING` | 缺 DRTK renderer 依赖 | 安装/编译匹配的 `drtk` |
| `triton: MISSING` | 缺 Triton runtime | 如果你的轮子栈需要 Triton，Windows 安装匹配的 `triton-windows`，Linux 安装匹配的 `triton` |
| `nvdiffrast: MISSING` 或 `nvdiffrec_render: MISSING` | 缺可选 renderer 包 | 基础 GLB 导出通常不需要；只有 renderer 路径需要时再装 |
| `natten: MISSING` | 缺基础 NATTEN 导入 | 重新安装 `requirements.txt`，或在 ComfyUI Python 里运行 `pip install natten==0.21.6` |
| `natten.HAS_LIBNATTEN: False` | NATTEN 能导入，但 strict CUDA NAF 不可用 | 使用 `naf_mode=fallback_if_missing`，或安装/编译匹配 CUDA NATTEN/libnatten |
| `RMBG-2.0` 缺失或 gated | 背景移除模型不可用 | 先在 Hugging Face 申请 [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) 权限并下载，或用透明 PNG/WebP + `background_mode=keep_alpha` |
| MoGe 缺失 | 自动相机模型不可用 | 下载 [Comfy-Org/MoGe](https://huggingface.co/Comfy-Org/MoGe)，或用 `camera_mode=manual` + **Pixal3D Camera Control** |

Windows 用户先看 [Windows 轮子指南](docs/windows_wheels.md)。Linux/WSL 用户先看 [Linux/WSL CUDA 指南](docs/linux_wsl_cuda.md)。修 CUDA 包时不要让 pip 替换已经工作的 Torch；按指南使用精确轮子或 `--no-deps`。

## 硬件要求

| 项目 | 要求 |
|------|------|
| 显存 (VRAM) | **推荐 20–32 GB**（`1536_cascade` 约需 32 GB；`native_low_vram` 在较小流程中可用到约 4–8 GB 显存） |
| 内存 (RAM) | **native low-VRAM 推荐 20–40 GB**（Pixal3D 在传到 GPU 前会在 CPU 侧暂存大量张量） |

## 平台现实情况

如果想尽量接近上游 Pixal3D 的完整体验，推荐 Linux 或 WSL，因为上游 NATTEN 为这些环境提供了近期官方 PyTorch CUDA 栈的 NATTEN/libnatten 预编译轮子。

原生 Windows 也受支持，可以生成和导出 GLB，但如果缺少完全匹配当前环境的 Windows CUDA 轮子，就需要使用 fallback 设置。社区已有部分 Windows NATTEN 轮子（例如 [drbaph/NATTEN-0.21.6-torch2100cu130-cp312-cp312-win_amd64](https://huggingface.co/drbaph/NATTEN-0.21.6-torch2100cu130-cp312-cp312-win_amd64)，适用于 Python 3.12、PyTorch 2.10、CUDA 13.0、sm120/RTX 5090），但官方 NATTEN 轮子索引 `whl.natten.org` 目前只为 PyTorch 2.10/2.11 提供 Linux 构建。普通 `natten==0.21.6` 只保证基础导入；如果 `natten.HAS_LIBNATTEN` 是 `False`，请使用 `naf_mode=fallback_if_missing`，不要使用 `strict`。

## GPU 计算能力

| 层级 | 架构 | 状态 |
|------|------|------|
| `sm80`-`sm89` | Ampere (A100, RTX 3090)、Ada Lovelace (RTX 4090) | 完全支持，FlashAttention 2/3 和 NATTEN/libnatten 轮子可用 |
| `sm90` | Hopper (H100) | 完全支持，轮子可用性与 Ampere/Ada 相同 |
| `sm100` | Blackwell (B200) | 预期支持，FlashAttention 2/3 轮子应可用，请验证 NATTEN/libnatten |
| `sm120` | Blackwell 消费级 (RTX 5090) | **可用，社区 NATTEN 轮子已有部分构建。** FlashAttention 2/3 轮子可用。[drbaph/NATTEN-0.21.6-torch2100cu130-cp312-cp312-win_amd64](https://huggingface.co/drbaph/NATTEN-0.21.6-torch2100cu130-cp312-cp312-win_amd64) 提供 Python 3.12 + PyTorch 2.10 + CUDA 13.0 的 sm120 轮子。如果轮子不匹配你的环境，仍可使用 `naf_mode=fallback_if_missing`。 |

Linux 或 WSL 是获得完整 NAF 支持的最佳方式，因为上游 NATTEN 在这些平台上发布了预编译轮子。原生 Windows 上，即使 `sm80`-`sm100` 也可能需要 fallback 设置，除非存在精确匹配当前 Python/PyTorch/CUDA 栈的 `win_amd64` NATTEN 轮子。

## 必需模型

主模型默认放在：

```text
ComfyUI/models/Pixal3D/TencentARC_Pixal3D/
```

辅助模型放在：

```text
ComfyUI/models/Pixal3D/briaai_RMBG-2.0/
ComfyUI/models/Pixal3D/camenduru_dinov3-vitl16-pretrain-lvd1689m/
```

MoGe 使用 ComfyUI 原生路径，不使用 `Ruicheng/moge-2-vitl` snapshot 文件夹：

```text
ComfyUI/
└── models/
    └── moge/
        ├── moge_1_vitl_fp16.safetensors
        └── moge_2_vitl_normal_fp16.safetensors
```

MoGe 下载地址：[Comfy-Org/MoGe](https://huggingface.co/Comfy-Org/MoGe)。

`briaai/RMBG-2.0` 是 Hugging Face gated model。要使用 `background_mode=auto_remove`，需要先在 Hugging Face 接受模型条款并登录，或手动把模型文件放到 `ComfyUI/models/Pixal3D/briaai_RMBG-2.0/`。

这些模型目录可以是普通文件夹、Windows junction，或 symlink。坏掉的链接会被当成缺失模型。链接目标仍然要是正常文件结构，例如 `pipeline.json`、`.json`、`.safetensors`、`ckpts/*.safetensors`，不要只链接 Hugging Face 的 blob cache。

## 推荐设置

一般 Windows / NVIDIA GPU 用户建议：

| 节点 | 参数 | 推荐值 |
|------|------|--------|
| Pixal3D Model Loader | `attention_backend` | `auto` |
| Pixal3D Model Loader | `vram_mode` | `dynamic_vram` |
| Pixal3D Model Loader | `download_if_missing` | `false`，除非你明确想让节点下载 |
| Pixal3D Model Loader | `load_moge` | `true` |
| Pixal3D Model Loader | `load_rembg` | `true`，如果使用 `auto_remove` |
| Pixal3D Model Loader | `naf_mode` | `fallback_if_missing` |
| Pixal3D Model Loader | `naf_target_size` | `upstream` |
| Pixal3D Image To 3D | `pipeline_type` | `1536_cascade` 质量更好，`1024_cascade` 更省显存 |
| Pixal3D Image To 3D | `background_mode` | 透明 PNG 用 `keep_alpha`，普通图用 `auto_remove` |
| Pixal3D Export GLB | `decimation_target` | `1000000` |
| Pixal3D Export GLB | `texture_size` | `4096` |
| Pixal3D Export GLB | `remesh` | 默认 `true`，节点会按设置传给 o_voxel；如果网格碎裂可改为 `false` |

### Pixal3D Camera Control

这个节点用于手动相机模式，只输出一个打包后的 ComfyUI 相机值：

| 输出 | 连接到 |
|------|--------|
| `manual_fov` | 单线连接到 `Pixal3D Image To 3D.manual_fov` |

可选的 `image` 输入只用于相机控件预览，不会输出图片。请把同一个 `Load Image` 直接连接到 `Pixal3D Image To 3D.image`。

这个节点只在 `Pixal3D Image To 3D.camera_mode=manual` 时生效。把 `manual_fov` 连接到 `Pixal3D Image To 3D.manual_fov` 后，在 manual 模式下 `Pixal3D Image To 3D` 上的 `manual_camera_angle_x`、`manual_distance`、`mesh_scale` 三个普通输入会被忽略，改用 Camera Control 里的值。如果 `camera_mode=moge`，连接的 `manual_fov` 会被忽略，仍然使用 MoGe 自动估计相机。

相机控件有 Scene 视图和 POV 视图，POV 使用的就是传给 Pixal3D 的同一组水平 FOV、distance 和 mesh scale。水平 FOV 会转换成弧度给 `manual_camera_angle_x`，distance/scale 会原样传入。

<details>
<summary>NAF 和 NATTEN</summary>

上游 Pixal3D 使用 NAF 来提升 shape/texture 阶段的特征质量。NAF 需要 CUDA 版 NATTEN/libnatten。只有安装了匹配当前 Python/PyTorch/CUDA 的 NATTEN，并且：

```bash
python -c "import natten; print(natten.HAS_LIBNATTEN)"
```

输出 `True`，才是真正可用的 NAF。

如果输出 `False`，说明安装的是纯 Python NATTEN 或不匹配的构建。此时请使用 `naf_mode=fallback_if_missing`。fallback 会保持 Pixal3D 权重需要的 2048 通道形状，但质量可能不如完整 NAF。

官方 NATTEN 命令示例：

```bash
pip install natten==0.21.6+torch2100cu130 -f https://whl.natten.org
```

注意：官方 `whl.natten.org` 轮子主要面向 Linux/WSL。Windows 上如果 pip 装到 `natten-0.21.6-py3-none-any.whl`，它不包含 libnatten，不能启用 strict NAF。

</details>

<details>
<summary>GLB 导出</summary>

导出的 GLB 写入：

```text
ComfyUI/output/
```

当前导出使用 PNG 贴图以兼容 Windows 3D Viewer。旧版如果使用 WebP 贴图并写入 `EXT_texture_webp`，部分 Windows 查看器会打不开。

连接方式：

```text
Pixal3D Export GLB glb_path
  -> Preview 3D & Animation model_file
```

**Pixal3D Image To 3D** 现在只输出 `pixal3d_result`，再接到 **Pixal3D Export GLB**。原生 3D 预览请使用导出的 `glb_path`。

</details>

<details>
<summary>常见问题</summary>

如果模型文件下载成 `blobs` 或 Hugging Face cache 结构，不要直接把 blob 文件夹当模型路径。Pixal3D-ComfyUI 期望正常文件夹和正常文件名，例如 `.safetensors`、`pipeline.json`、`ckpts/*.safetensors`。

如果 `moge_repo` 或旧字段还显示在节点 UI，重启 ComfyUI 并强制刷新浏览器。新版节点固定使用 `ComfyUI/models/moge/moge_2_vitl_normal_fp16.safetensors`。

如果看到 `NAF unavailable (natten is installed, but HAS_LIBNATTEN is false)`，说明 NATTEN 没有 CUDA libnatten。生成可以继续，但会使用 fallback。

如果 Windows 3D Viewer 打不开 GLB，请确认使用的是新版导出的 PNG texture GLB，而不是旧的 WebP texture GLB。

如果切换 `vram_mode`、`attention_backend` 或其他 Model Loader 设置，新版节点会卸载旧的 Pixal3D cache handle。ComfyUI 原生 unload 主要释放 VRAM；如果想立刻释放 Pixal3D 的 CPU RAM，请运行 **Pixal3D Unload Model** 节点，或重启 ComfyUI。

`native_low_vram` 模式会尽量分阶段移动模型：小流程可能只需要约 4–8 GB 显存，但需要大量系统内存，建议准备 20–40 GB RAM，速度也会更慢。RMBG 只在背景预处理时上 GPU，MoGe 只在相机估计时上 GPU，之后都会回到 CPU；Pixal3D 主流程则按上游 low-vram 逻辑逐个移动 flow/decoder 模块。

低显存推荐设置：

| 节点 | 设置 |
|------|------|
| Pixal3D Model Loader | `vram_mode=native_low_vram` |
| Pixal3D Model Loader | `load_moge=false` |
| Pixal3D Model Loader | `load_rembg=false` |
| Pixal3D Image To 3D | `camera_mode=manual` |
| Pixal3D Image To 3D | 透明 PNG/WebP 输入用 `background_mode=keep_alpha` |
| Pixal3D Camera Control | 把 `manual_fov` 连接到 `Pixal3D Image To 3D.manual_fov` |

这条路径建议使用带透明背景的 PNG 或 WebP，这样不用加载 RMBG；相机则用 **Pixal3D Camera Control**，不要加载 MoGe。

</details>

## Windows CUDA 轮子资源

- [Wildminder/AI-windows-whl](https://huggingface.co/Wildminder/AI-windows-whl/tree/main) — 预编译 Windows CUDA 轮子（FlashAttention、flex_gemm、cumesh、o_voxel、drtk 等）
- [lldacing/NATTEN-windows](https://huggingface.co/lldacing/NATTEN-windows/tree/main) — 预编译 Windows CUDA NATTEN/libnatten 轮子，用于 strict NAF 支持
- [drbaph/NATTEN-0.21.6-torch2100cu130-cp312-cp312-win_amd64](https://huggingface.co/drbaph/NATTEN-0.21.6-torch2100cu130-cp312-cp312-win_amd64) — Windows NATTEN 0.21.6 轮子，Python 3.12、PyTorch 2.10、CUDA 13.0、sm120 (RTX 5090)
- [naxneri/natten-0.21.6-blackwell-cu128-cp312-cp312-win_amd64](https://huggingface.co/naxneri/natten-0.21.6-blackwell-cu128-cp312-cp312-win_amd64) — Windows NATTEN 0.21.6 轮子，Python 3.12、CUDA 12.8、Blackwell (sm100/sm120)
- [在 Windows 上编译 NATTEN](docs/Build_Natten_windows.md) — 从源码编译 NATTEN 的完整指南，包括 MSVC 和 CUDA 修复

## 🤗 致谢

本项目大量基于 [Trellis.2](https://github.com/microsoft/TRELLIS.2) 和 [Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2) 构建。我们衷心感谢作者在可扩展 3D 生成方面的杰出工作，这是我们代码库和模型架构的基础。

我们也感谢以下仓库的贡献：

- [Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2)
- [Trellis](https://github.com/microsoft/TRELLIS)
- [Trellis.2](https://github.com/microsoft/TRELLIS.2)

## 📄 引用

如果觉得本工作有用，请考虑引用：

```bibtex
@article{li2026pixal3d,
    title={Pixal3D: Pixel-Aligned 3D Generation from Images},
    author={Li, Dong-Yang and Zhao, Wang and Chen, Yuxin and Hu, Wenbo and Guo, Meng-Hao and Zhang, Fang-Lue and Shan, Ying and Hu, Shi-Min},
    journal={arXiv preprint arXiv:2605.10922},
    year={2026}
}
```
