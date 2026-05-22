from __future__ import annotations

import gc
import importlib
import importlib.util
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

import comfy.model_management as model_management
import comfy.model_patcher
import folder_paths

LOGGER = logging.getLogger("Pixal3D_ComfyUI")


def _install_safe_model_patcher_del() -> None:
    """Avoid noisy ignored exceptions from already-detached Comfy patchers."""
    model_patcher_cls = getattr(comfy.model_patcher, "ModelPatcher", None)
    if model_patcher_cls is None or getattr(model_patcher_cls, "_pixal3d_safe_del", False):
        return

    original_del = getattr(model_patcher_cls, "__del__", None)
    if not callable(original_del):
        return

    def safe_del(self):
        if getattr(self, "model", None) is None:
            return
        try:
            return original_del(self)
        except AttributeError as exc:
            if "NoneType" not in str(exc):
                raise

    model_patcher_cls.__del__ = safe_del
    model_patcher_cls._pixal3d_safe_del = True
    model_patcher_cls._pixal3d_original_del = original_del


_install_safe_model_patcher_del()

DEFAULT_MODEL_REPO = "TencentARC/Pixal3D"
DEFAULT_DINO_REPO = "camenduru/dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_REMBG_REPO = "briaai/RMBG-2.0"
DEFAULT_MOGE_REPO = "Comfy-Org/MoGe"
NATIVE_COMFY_MOGE_REPO = "Comfy-Org/MoGe"
NATIVE_COMFY_MOGE_SUBDIR = "geometry_estimation"
NATIVE_COMFY_MOGE_FILES = {
    "moge_1_vitl_fp16.safetensors": "geometry_estimation/moge_1_vitl_fp16.safetensors",
    "moge_2_vitl_normal_fp16.safetensors": "geometry_estimation/moge_2_vitl_normal_fp16.safetensors",
}
NATIVE_COMFY_MOGE_MODEL = "moge_2_vitl_normal_fp16.safetensors"
ATTENTION_CHOICES = ["auto", "flash_attn_2", "flash_attn_3"]
VRAM_MODE_CHOICES = ["dynamic_vram", "hybrid_low_vram", "native_low_vram", "full_gpu"]
NAF_MODE_CHOICES = ["fallback_if_missing", "strict"]
NAF_TARGET_SIZE_CHOICES = ["upstream", "1024", "512", "256", "128"]
COMFY_OPS_VRAM_MODES = {"dynamic_vram", "hybrid_low_vram"}
LOW_VRAM_MODE_CHOICES = {"native_low_vram", "hybrid_low_vram"}
REQUIRED_CUDA_MODULE_GROUPS = (
    ("flex_gemm", ("flex_gemm_ap", "flex_gemm")),
    ("cumesh", ("cumesh_vb", "cumesh")),
    ("o_voxel", ("o_voxel_vb_ap", "o_voxel")),
    ("drtk", ("drtk",)),
)
OPTIONAL_CUDA_MODULES = ("triton", "nvdiffrast.torch", "nvdiffrec_render")

IMAGE_COND_CONFIGS = {
    "ss": {
        "model_name": DEFAULT_DINO_REPO,
        "image_size": 512,
        "grid_resolution": 16,
    },
    "shape_512": {
        "model_name": DEFAULT_DINO_REPO,
        "image_size": 512,
        "grid_resolution": 32,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "shape_1024": {
        "model_name": DEFAULT_DINO_REPO,
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "tex_1024": {
        "model_name": DEFAULT_DINO_REPO,
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 1024,
    },
}


@contextmanager
def _temporary_comfy_ops(enabled: bool):
    import comfy.memory_management

    originals = {}
    original_aimdo_enabled = getattr(comfy.memory_management, "aimdo_enabled", None)

    if enabled:
        import comfy.ops

        ops = comfy.ops.manual_cast
        replacements = {
            "Linear": getattr(ops, "Linear", torch.nn.Linear),
            "Conv1d": getattr(ops, "Conv1d", torch.nn.Conv1d),
            "Conv2d": getattr(ops, "Conv2d", torch.nn.Conv2d),
            "Conv3d": getattr(ops, "Conv3d", torch.nn.Conv3d),
            "BatchNorm2d": getattr(ops, "BatchNorm2d", torch.nn.BatchNorm2d),
            "GroupNorm": getattr(ops, "GroupNorm", torch.nn.GroupNorm),
            "LayerNorm": getattr(ops, "LayerNorm", torch.nn.LayerNorm),
            "ConvTranspose1d": getattr(ops, "ConvTranspose1d", torch.nn.ConvTranspose1d),
            "ConvTranspose2d": getattr(ops, "ConvTranspose2d", torch.nn.ConvTranspose2d),
            "Embedding": getattr(ops, "Embedding", torch.nn.Embedding),
        }
        for name, replacement in replacements.items():
            if hasattr(torch.nn, name):
                originals[name] = getattr(torch.nn, name)
                setattr(torch.nn, name, replacement)

    # Pixal3D constructors inspect parameters before load_state_dict runs.
    # Comfy's Windows lazy init can temporarily create no-parameter layers,
    # which breaks those constructors. Suppress that lazy init for every
    # Pixal3D load mode because previously imported Pixal3D classes may still
    # inherit Comfy's patched torch.nn bases after switching modes.
    if original_aimdo_enabled is not None:
        comfy.memory_management.aimdo_enabled = False

    try:
        _refresh_pixal3d_precision_module_types()
        yield
    finally:
        if original_aimdo_enabled is not None:
            comfy.memory_management.aimdo_enabled = original_aimdo_enabled
        for name, original in originals.items():
            setattr(torch.nn, name, original)
        _refresh_pixal3d_precision_module_types()


def _refresh_pixal3d_precision_module_types() -> None:
    try:
        from torch.nn.modules.conv import Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d
        from torch.nn.modules.linear import Linear
        from pixal3d.modules import sparse as sparse_modules
        from pixal3d.modules import utils as pixal3d_utils
    except Exception:
        return

    module_types: list[type] = []

    def add(module_type: Any) -> None:
        if isinstance(module_type, type) and module_type not in module_types:
            module_types.append(module_type)

    for module_type in (
        Linear,
        Conv1d,
        Conv2d,
        Conv3d,
        ConvTranspose1d,
        ConvTranspose2d,
        ConvTranspose3d,
    ):
        add(module_type)
    for name in (
        "Linear",
        "Conv1d",
        "Conv2d",
        "Conv3d",
        "ConvTranspose1d",
        "ConvTranspose2d",
        "ConvTranspose3d",
    ):
        add(getattr(torch.nn, name, None))
    for name in ("SparseConv3d", "SparseInverseConv3d", "SparseLinear"):
        add(getattr(sparse_modules, name, None))
    try:
        import comfy.ops

        manual_cast_ops = comfy.ops.manual_cast
        for name in (
            "Linear",
            "Conv1d",
            "Conv2d",
            "Conv3d",
            "ConvTranspose1d",
            "ConvTranspose2d",
        ):
            add(getattr(manual_cast_ops, name, None))
    except Exception:
        pass

    pixal3d_utils.MIX_PRECISION_MODULES = tuple(module_types)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _try_import(name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "")
        return True, str(version) if version else "import ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _try_import_any(names: tuple[str, ...]) -> tuple[bool, str, str | None]:
    details = []
    for name in names:
        ok, detail = _try_import(name)
        details.append(f"{name}: {detail}")
        if ok:
            return True, detail, name
    return False, "; ".join(details), None


def _import_first(names: tuple[str, ...]):
    last_exc: Exception | None = None
    for name in names:
        try:
            return importlib.import_module(name), name
        except Exception as exc:
            last_exc = exc
    raise ImportError(f"Could not import any of {names}: {last_exc}")


def _cuda_capability() -> tuple[int, int] | None:
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return int(major), int(minor)
    except Exception:
        pass
    return None


def _is_blackwell_gpu() -> bool:
    capability = _cuda_capability()
    return capability is not None and capability[0] >= 12


def _uses_comfy_ops(vram_mode: str) -> bool:
    return vram_mode in COMFY_OPS_VRAM_MODES


def _uses_low_vram_staging(vram_mode: str) -> bool:
    return vram_mode in LOW_VRAM_MODE_CHOICES


def _uses_comfy_model_management(vram_mode: str) -> bool:
    return vram_mode != "native_low_vram"


def _release_cached_memory(aggressive: bool = False) -> None:
    gc.collect()
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass
    try:
        model_management.soft_empty_cache()
    except Exception:
        pass
    if aggressive and os.name == "nt":
        try:
            import ctypes

            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetProcessWorkingSetSize(handle, -1, -1)
        except Exception:
            LOGGER.debug("Could not trim Windows process working set", exc_info=True)


def _sanitize_module_name(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]", "_", name)


def environment_report() -> str:
    lines = ["Pixal3D-ComfyUI environment check", ""]
    lines.append(f"ComfyUI models/Pixal3D: {pixal3d_models_dir()}")
    lines.append(f"ComfyUI models/geometry_estimation: {moge_models_dir()}")
    lines.append(f"Nodepack path: {_repo_root()}")
    lines.append("")

    try:
        import torch

        lines.append(f"Python: {sys.version.split()[0]}")
        lines.append(f"PyTorch: {getattr(torch, '__version__', 'unknown')}")
        lines.append(f"torch.version.cuda: {getattr(torch.version, 'cuda', None)}")
        lines.append(f"torch.cuda.is_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
            capability = _cuda_capability()
            if capability is not None:
                lines.append(f"CUDA capability: {capability[0]}.{capability[1]}")
                lines.append(f"Blackwell/sm_120 class: {_is_blackwell_gpu()}")
    except Exception as exc:
        lines.append(f"Torch check failed: {type(exc).__name__}: {exc}")

    lines.append("")
    lines.append("Required CUDA/Pixal3D modules:")
    required_ok = True
    for label, names in REQUIRED_CUDA_MODULE_GROUPS:
        ok, detail, imported_name = _try_import_any(names)
        required_ok = required_ok and ok
        if ok:
            lines.append(f"- {label}: OK via {imported_name} ({detail})")
        else:
            lines.append(f"- {label}: MISSING ({detail})")

    lines.append("")
    lines.append("Attention modules, need at least one:")
    fa2_ok, fa2_detail = _try_import("flash_attn")
    fa3_ok, fa3_detail = _try_import("flash_attn_interface")
    lines.append(f"- flash_attn: {'OK' if fa2_ok else 'MISSING'} ({fa2_detail})")
    lines.append(f"- flash_attn_interface: {'OK' if fa3_ok else 'MISSING'} ({fa3_detail})")
    attention_ok = fa2_ok or fa3_ok

    lines.append("")
    lines.append("Optional modules:")
    for name in OPTIONAL_CUDA_MODULES:
        ok, detail = _try_import(name)
        lines.append(f"- {name}: {'OK' if ok else 'not found'} ({detail})")

    lines.append("")
    lines.append("NAF / NATTEN:")
    natten_ok, natten_detail = _try_import("natten")
    lines.append(f"- natten: {'OK' if natten_ok else 'MISSING'} ({natten_detail})")
    if natten_ok:
        try:
            import natten

            lines.append(f"- natten.HAS_LIBNATTEN: {getattr(natten, 'HAS_LIBNATTEN', 'unknown')}")
        except Exception as exc:
            lines.append(f"- natten.HAS_LIBNATTEN: check failed ({type(exc).__name__}: {exc})")
    else:
        lines.append("- strict NAF mode: unavailable until a CUDA-enabled NATTEN build imports")

    lines.append("")
    if fa3_ok:
        lines.append("Attention auto mode would use: flash_attn_3")
    elif fa2_ok:
        lines.append("Attention auto mode would use: flash_attn_2")
    else:
        lines.append("Attention auto mode would fail: install FlashAttention 2 or 3.")

    if required_ok and attention_ok:
        lines.append("Result: required imports passed. Pixal3D can try to load the model.")
    else:
        lines.append("Result: required imports missing or failed. Install matching wheels for this Python/Torch/CUDA stack.")
    return "\n".join(lines)


def pixal3d_models_dir() -> Path:
    models_dir = Path(folder_paths.models_dir) / "Pixal3D"
    models_dir.mkdir(parents=True, exist_ok=True)
    try:
        folder_paths.add_model_folder_path("Pixal3D", str(models_dir))
    except Exception:
        pass
    return models_dir


def moge_models_dir() -> Path:
    models_dir = Path(folder_paths.models_dir) / NATIVE_COMFY_MOGE_SUBDIR
    models_dir.mkdir(parents=True, exist_ok=True)
    try:
        folder_paths.add_model_folder_path("moge", str(models_dir))
    except Exception:
        pass
    return models_dir


def _repo_folder_name(model_repo: str) -> str:
    repo = model_repo.strip().strip("/")
    if not repo:
        return "TencentARC_Pixal3D"
    if re.match(r"^[a-zA-Z]:\\", repo) or repo.startswith(("/", "\\")):
        return Path(repo).name
    return re.sub(r"[^0-9a-zA-Z_.-]+", "_", repo)


def _repo_basename(model_repo: str) -> str:
    repo = model_repo.strip().strip("/")
    if not repo:
        return _repo_folder_name(model_repo)
    return repo.replace("\\", "/").split("/")[-1]


def _repo_cache_folder_name(model_repo: str) -> str:
    repo = model_repo.strip().strip("/").replace("\\", "/")
    parts = [part for part in repo.split("/") if part]
    if len(parts) >= 2:
        return "models--" + "--".join(parts[-2:])
    return "models--" + _repo_folder_name(model_repo)


def _normalize_hf_repo_id(repo_id: str) -> str:
    repo_id = repo_id.strip().strip("/")
    for prefix in ("https://huggingface.co/", "http://huggingface.co/"):
        if repo_id.startswith(prefix):
            repo_id = repo_id[len(prefix) :]
            break
    return repo_id.strip("/") or NATIVE_COMFY_MOGE_REPO


def _snapshot_subdirs(cache_dir: Path) -> list[Path]:
    snapshots = cache_dir / "snapshots"
    if not snapshots.exists() or not snapshots.is_dir():
        return []
    return sorted(
        [path for path in snapshots.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _candidate_snapshot_dirs(repo_id: str) -> list[Path]:
    root = pixal3d_models_dir()
    candidates: list[Path] = []
    for name in dict.fromkeys(
        [
            _repo_folder_name(repo_id),
            _repo_basename(repo_id),
            _repo_cache_folder_name(repo_id),
        ]
    ):
        candidate = root / name
        candidates.append(candidate)
        candidates.extend(_snapshot_subdirs(candidate))
    return candidates


def _missing_pixal3d_model_files(model_dir: Path) -> list[Path]:
    pipeline_file = model_dir / "pipeline.json"
    if not pipeline_file.exists():
        return [pipeline_file]
    try:
        data = json.loads(pipeline_file.read_text(encoding="utf-8"))
    except Exception:
        return [pipeline_file]
    missing: list[Path] = []
    models = data.get("args", {}).get("models", {})
    for rel_path in models.values():
        base = model_dir / rel_path
        for suffix in (".json", ".safetensors"):
            candidate = Path(str(base) + suffix)
            if not candidate.exists():
                missing.append(candidate)
    return missing


def _snapshot_has_files(model_dir: Path) -> bool:
    if not model_dir.exists() or not model_dir.is_dir():
        return False
    seen_dirs: set[str] = set()
    for current, dirnames, filenames in os.walk(model_dir, followlinks=True):
        current_path = Path(current)
        real_current = os.path.realpath(current)
        if real_current in seen_dirs:
            dirnames[:] = []
            continue
        seen_dirs.add(real_current)
        if ".cache" in current_path.parts or ".git" in current_path.parts:
            dirnames[:] = []
            continue
        dirnames[:] = [name for name in dirnames if name not in {".cache", ".git"}]
        if filenames:
            return True
    return False


def _find_local_model_file(model_dir: Path, names: tuple[str, ...]) -> Path | None:
    if model_dir.is_file():
        return model_dir
    for name in names:
        candidate = model_dir / name
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _download_snapshot(repo_id: str, target: Path, hf_endpoint: str | None = None) -> None:
    from huggingface_hub import snapshot_download

    endpoint = hf_endpoint.strip() if hf_endpoint else None
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        resume_download=True,
        endpoint=endpoint,
    )
    _remove_snapshot_metadata(target)


def _download_hf_file_to_path(repo_id: str, remote_filename: str, destination: Path, hf_endpoint: str | None = None) -> None:
    from huggingface_hub import hf_hub_download

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=".hf_download_", dir=str(destination.parent)))
    try:
        endpoint = hf_endpoint.strip() if hf_endpoint else "https://huggingface.co"
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=remote_filename,
                repo_type="model",
                local_dir=str(temp_root),
                local_dir_use_symlinks=False,
                resume_download=True,
                endpoint=endpoint,
            )
        )
        shutil.move(str(downloaded), str(destination))
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _remove_snapshot_metadata(target: Path) -> None:
    target = target.resolve()
    root = pixal3d_models_dir().resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return
    for name in (".cache", ".git"):
        metadata_dir = target / name
        if metadata_dir.exists() and metadata_dir.is_dir():
            shutil.rmtree(metadata_dir, ignore_errors=True)


def resolve_model_path(model_repo: str, download_if_missing: bool, hf_endpoint: str = "") -> str:
    model_repo = model_repo.strip() or DEFAULT_MODEL_REPO
    local_candidate = Path(model_repo).expanduser()
    if local_candidate.exists():
        missing = _missing_pixal3d_model_files(local_candidate)
        if missing:
            preview = "\n".join(str(path) for path in missing[:8])
            raise FileNotFoundError(f"Local Pixal3D model folder is incomplete:\n{preview}")
        return str(local_candidate)

    models_root = pixal3d_models_dir()
    target = models_root / _repo_folder_name(model_repo)
    found_model_dir = _find_pixal3d_model_dir(model_repo)
    if found_model_dir is not None:
        return str(found_model_dir)
    missing = _missing_pixal3d_model_files(target)

    if not download_if_missing:
        raise FileNotFoundError(
            "Pixal3D model is missing. Enable download_if_missing on the loader to download it to "
            f"{target}, or place a complete model snapshot there. Missing files include:\n"
            + "\n".join(str(path) for path in missing[:8])
        )

    LOGGER.info("Downloading Pixal3D model %s to %s", model_repo, target)
    _download_snapshot(model_repo, target, hf_endpoint)
    return str(target)


def _find_pixal3d_model_dir(model_repo: str) -> Path | None:
    for candidate in _candidate_snapshot_dirs(model_repo):
        if candidate.exists() and not _missing_pixal3d_model_files(candidate):
            return candidate
    return None


def resolve_helper_model_path(
    repo_id: str,
    download_if_missing: bool,
    label: str,
    hf_endpoint: str = "",
    preferred_files: tuple[str, ...] = (),
) -> str:
    repo_id = repo_id.strip()
    local_candidate = Path(repo_id).expanduser()
    if local_candidate.exists():
        local_file = _find_local_model_file(local_candidate, preferred_files)
        if local_file is not None:
            return str(local_file)
        return str(local_candidate)

    target = pixal3d_models_dir() / _repo_folder_name(repo_id)
    for candidate in _candidate_snapshot_dirs(repo_id):
        if not _snapshot_has_files(candidate):
            continue
        local_file = _find_local_model_file(candidate, preferred_files)
        if local_file is not None:
            return str(local_file)
        return str(candidate)

    if not download_if_missing:
        raise FileNotFoundError(
            f"{label} helper model is missing at {target}.\n"
            "download_if_missing is false, so Pixal3D-ComfyUI will not download it.\n"
            f"Enable download_if_missing once, or place a complete snapshot of {repo_id} there."
        )

    LOGGER.info("Downloading Pixal3D helper model %s to %s", repo_id, target)
    _download_snapshot(repo_id, target, hf_endpoint)
    local_file = _find_local_model_file(target, preferred_files)
    return str(local_file or target)


def resolve_native_comfy_moge_path(
    repo_id: str,
    download_if_missing: bool,
    hf_endpoint: str = "",
) -> str:
    repo_id = _normalize_hf_repo_id(repo_id or NATIVE_COMFY_MOGE_REPO)
    if repo_id != NATIVE_COMFY_MOGE_REPO:
        LOGGER.warning(
            "Ignoring moge_repo=%s. Pixal3D-ComfyUI uses native ComfyUI MoGe from %s under %s.",
            repo_id,
            NATIVE_COMFY_MOGE_REPO,
            moge_models_dir(),
        )
        repo_id = NATIVE_COMFY_MOGE_REPO

    target_dir = moge_models_dir()
    legacy_dir = Path(folder_paths.models_dir) / "moge"

    if download_if_missing:
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename, remote_filename in NATIVE_COMFY_MOGE_FILES.items():
            destination = target_dir / filename
            if not destination.exists():
                LOGGER.info("Downloading native ComfyUI MoGe file %s from %s to %s", remote_filename, repo_id, destination)
                _download_hf_file_to_path(repo_id, remote_filename, destination, hf_endpoint)

    for search_dir in (target_dir, legacy_dir):
        required_path = search_dir / NATIVE_COMFY_MOGE_MODEL
        if required_path.exists() and required_path.is_file():
            return str(required_path)

    expected = "\n".join(str(target_dir / filename) for filename in NATIVE_COMFY_MOGE_FILES)
    raise FileNotFoundError(
        "Native ComfyUI MoGe is missing.\n"
        f"Download link: https://huggingface.co/{NATIVE_COMFY_MOGE_REPO}\n"
        f"Place the files in ComfyUI/models/{NATIVE_COMFY_MOGE_SUBDIR}/, not in a Hugging Face snapshot folder:\n"
        f"{expected}\n"
        "Enable download_if_missing once to download these files automatically."
    )


def _native_comfy_moge_v2_vitl_config() -> dict[str, Any]:
    head_common = {
        "dim_in": [1024, 256, 128, 64, 32],
        "dim_res_blocks": [1024, 256, 128, 64, 32],
        "num_res_blocks": [0, 1, 1, 1, 0],
        "res_block_in_norm": "none",
        "res_block_hidden_norm": "none",
        "resamplers": ["conv_transpose", "conv_transpose", "conv_transpose", "bilinear"],
    }
    return {
        "encoder": {
            "backbone": "dinov2_vitl14",
            "intermediate_layers": [5, 11, 17, 23],
            "dim_out": 1024,
        },
        "neck": {
            "dim_in": [1026, 2, 2, 2, 2],
            "dim_out": None,
            "dim_res_blocks": [1024, 256, 128, 64, 32],
            "num_res_blocks": [0, 2, 2, 2, 0],
            "res_block_in_norm": "none",
            "res_block_hidden_norm": "none",
            "resamplers": ["conv_transpose", "conv_transpose", "conv_transpose", "bilinear"],
        },
        "points_head": {**head_common, "dim_out": [None, None, None, None, 3]},
        "normal_head": {**head_common, "dim_out": [None, None, None, None, 3]},
        "mask_head": {**head_common, "dim_out": [None, None, None, None, 1]},
        "scale_head": {"dims": [1024, 1024, 1024, 1]},
        "remap_output": "exp",
        "num_tokens_range": [1200, 3600],
    }


def install_disabled_rembg() -> None:
    from pixal3d.pipelines import rembg

    if hasattr(rembg, "DisabledRMBG"):
        return

    class DisabledRMBG:
        def to(self, _device: str):
            return self

        def cuda(self):
            return self

        def cpu(self):
            return self

        def __call__(self, _image: Image.Image) -> Image.Image:
            raise RuntimeError(
                "Pixal3D built-in background removal is disabled or missing. "
                "Use background_mode=none, pass an RGBA image with alpha and use keep_alpha, "
                "or enable load_rembg and provide/download briaai/RMBG-2.0."
            )

    rembg.DisabledRMBG = DisabledRMBG


def _rembg_is_disabled(pipeline: Any) -> bool:
    return type(getattr(pipeline, "rembg_model", None)).__name__ == "DisabledRMBG"


def _move_rembg_model(pipeline: Any, device: torch.device) -> None:
    rembg_model = getattr(pipeline, "rembg_model", None)
    if rembg_model is not None and not _rembg_is_disabled(pipeline) and hasattr(rembg_model, "to"):
        rembg_model.to(device)


def _cpu_module(module: Any) -> None:
    if isinstance(module, torch.nn.Module):
        try:
            module.cpu()
        except Exception:
            LOGGER.debug("Could not move Pixal3D module to CPU", exc_info=True)


def _offload_low_vram_pipeline_modules(pipeline: Any) -> None:
    for module in getattr(pipeline, "models", {}).values():
        _cpu_module(module)
    for attr in (
        "image_cond_model_ss",
        "image_cond_model_shape_512",
        "image_cond_model_shape_1024",
        "image_cond_model_tex_1024",
        "rembg_model",
    ):
        module = getattr(pipeline, attr, None)
        if module is not None and not (attr == "rembg_model" and _rembg_is_disabled(pipeline)):
            _cpu_module(module)


def prepare_pipeline_config(
    model_path: str,
    download_if_missing: bool,
    load_rembg: bool = False,
    hf_endpoint: str = "",
) -> str:
    model_dir = Path(model_path)
    pipeline_file = model_dir / "pipeline.json"
    data = json.loads(pipeline_file.read_text(encoding="utf-8"))
    rembg_config = data.get("args", {}).setdefault("rembg_model", {})
    rembg_args = rembg_config.setdefault("args", {})
    if load_rembg:
        rembg_repo = rembg_args.get("model_name", DEFAULT_REMBG_REPO) or DEFAULT_REMBG_REPO
        rembg_args["model_name"] = resolve_helper_model_path(
            rembg_repo,
            download_if_missing,
            "Background remover (RMBG)",
            hf_endpoint,
        )
    else:
        install_disabled_rembg()
        rembg_config["name"] = "DisabledRMBG"
        rembg_config["args"] = {}
    generated = model_dir / "_pixal3d_comfy_pipeline.json"
    generated.write_text(json.dumps(data, indent=4), encoding="utf-8")
    return generated.name


def resolve_image_cond_config(
    config: dict[str, Any],
    download_if_missing: bool,
    hf_endpoint: str = "",
    naf_mode: str = "fallback_if_missing",
    naf_target_size: str = "upstream",
) -> dict[str, Any]:
    resolved = dict(config)
    model_name = resolved.get("model_name", DEFAULT_DINO_REPO) or DEFAULT_DINO_REPO
    resolved["model_name"] = resolve_helper_model_path(
        model_name,
        download_if_missing,
        "DINOv3 image encoder",
        hf_endpoint,
    )
    if resolved.get("use_naf_upsample", False):
        resolved["naf_download_if_missing"] = bool(download_if_missing)
        resolved["naf_fallback_mode"] = "strict" if naf_mode == "strict" else "duplicate_lr"
        if naf_target_size != "upstream":
            resolved["naf_target_size"] = int(naf_target_size)
    return resolved


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    if image.ndim == 4:
        image = image[0]
    image = image.detach().float().cpu().clamp(0, 1).numpy()
    image = (image * 255.0).round().astype(np.uint8)
    return Image.fromarray(image)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None]


def _pil_has_useful_alpha(image: Image.Image) -> bool:
    if image.mode != "RGBA":
        return False
    alpha = np.asarray(image.getchannel("A"))
    return bool(np.any(alpha < 255))


def _pad_to_square(image: Image.Image, bg_color: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    size = max(width, height)
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", (size, size), (*bg_color, 0))
        canvas.paste(image, ((size - width) // 2, (size - height) // 2), image)
        return canvas
    canvas = Image.new("RGB", (size, size), bg_color)
    canvas.paste(image.convert("RGB"), ((size - width) // 2, (size - height) // 2))
    return canvas


def configure_torch_hub_cache() -> None:
    hub_dir = pixal3d_models_dir() / "torch_hub"
    hub_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(hub_dir))
    try:
        torch.hub.set_dir(str(hub_dir / "hub"))
    except Exception:
        LOGGER.debug("Could not set torch hub dir", exc_info=True)


def ensure_pixal3d_source(pixal3d_repo_path: str | None = None) -> Path:
    root = _repo_root()
    source_root = root
    if pixal3d_repo_path:
        candidate = Path(pixal3d_repo_path).expanduser()
        if not candidate.exists():
            raise FileNotFoundError(f"Pixal3D source path does not exist: {candidate}")
        if (candidate / "pixal3d").is_dir():
            source_root = candidate
        elif candidate.name == "pixal3d" and candidate.is_dir():
            source_root = candidate.parent
        else:
            raise FileNotFoundError(f"Pixal3D source path must contain a pixal3d package: {candidate}")

    source_root_str = str(source_root)
    if source_root_str not in sys.path:
        sys.path.insert(0, source_root_str)
    return source_root


def ensure_cuda_aliases() -> None:
    try:
        cumesh_impl, cumesh_name = _import_first(("cumesh_vb", "cumesh"))
        sys.modules.setdefault("cumesh", cumesh_impl)
        sys.modules.setdefault("cumesh.bvh", importlib.import_module(f"{cumesh_name}.bvh"))
    except Exception as exc:
        LOGGER.debug("cumesh alias unavailable yet: %s", exc)

    try:
        flex_impl, flex_name = _import_first(("flex_gemm_ap", "flex_gemm"))
        sys.modules.setdefault("flex_gemm", flex_impl)
        for name in ("ops", "ops.grid_sample", "ops.spconv", "utils", "kernels"):
            sys.modules.setdefault(f"flex_gemm.{name}", importlib.import_module(f"{flex_name}.{name}"))
    except Exception as exc:
        LOGGER.debug("flex_gemm alias unavailable yet: %s", exc)

    try:
        o_voxel_impl, o_voxel_name = _import_first(("o_voxel_vb_ap", "o_voxel"))
        sys.modules.setdefault("o_voxel", o_voxel_impl)
        for name in ("convert", "io", "serialize"):
            sys.modules.setdefault(f"o_voxel.{name}", importlib.import_module(f"{o_voxel_name}.{name}"))
        postprocess = importlib.import_module("pixal3d_comfy.postprocess")
        sys.modules["o_voxel.postprocess"] = postprocess
        setattr(sys.modules["o_voxel"], "postprocess", postprocess)
    except Exception as exc:
        LOGGER.debug("o_voxel alias unavailable yet: %s", exc)


def resolve_attention_backend(attention_backend: str) -> str:
    if attention_backend == "flash_attn_2":
        ok, detail = _try_import("flash_attn")
        if not ok:
            raise RuntimeError(f"FlashAttention 2 was selected, but flash_attn could not import: {detail}")
        return "flash_attn"
    if attention_backend == "flash_attn_3":
        ok, detail = _try_import("flash_attn_interface")
        if not ok:
            raise RuntimeError(f"FlashAttention 3 was selected, but flash_attn_interface could not import: {detail}")
        return "flash_attn_3"
    if attention_backend != "auto":
        raise ValueError(f"Unsupported attention backend: {attention_backend}")
    fa3_ok, _ = _try_import("flash_attn_interface")
    if fa3_ok:
        return "flash_attn_3"
    fa2_ok, _ = _try_import("flash_attn")
    if fa2_ok:
        return "flash_attn"
    raise RuntimeError("Pixal3D sparse attention needs FlashAttention 2 or 3. Install one wheel in ComfyUI's venv.")


def apply_attention_backend(attention_backend: str) -> str:
    backend = resolve_attention_backend(attention_backend)
    os.environ["ATTN_BACKEND"] = backend
    os.environ["SPARSE_ATTN_BACKEND"] = backend
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("FLEX_GEMM_AUTOTUNER_VERBOSE", "1")
    os.environ.setdefault("FLEX_GEMM_AUTOTUNE_CACHE_PATH", str(_repo_root() / "autotune_cache.json"))

    try:
        from pixal3d.modules.attention import config as dense_config

        dense_config.set_backend(backend)
    except Exception:
        pass
    try:
        from pixal3d.modules.sparse import config as sparse_config

        sparse_config.set_attn_backend(backend)
    except Exception:
        pass
    return backend


def build_image_cond_model(
    config: dict[str, Any],
    download_if_missing: bool,
    hf_endpoint: str = "",
    naf_mode: str = "fallback_if_missing",
    naf_target_size: str = "upstream",
    naf_low_vram: bool = False,
):
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor

    model = DinoV3ProjFeatureExtractor(
        **resolve_image_cond_config(config, download_if_missing, hf_endpoint, naf_mode, naf_target_size),
        naf_low_vram=bool(naf_low_vram),
    )
    model.eval()
    return model


def load_moge_model(
    model_name: str = DEFAULT_MOGE_REPO,
    download_if_missing: bool = False,
    hf_endpoint: str = "",
):
    import comfy.utils
    from moge.model.v2 import MoGeModel

    model_path = resolve_native_comfy_moge_path(
        model_name,
        download_if_missing,
        hf_endpoint,
    )
    state_dict = comfy.utils.load_torch_file(model_path)
    model = MoGeModel(**_native_comfy_moge_v2_vitl_config())
    if str(model_path).lower().endswith("_fp16.safetensors"):
        model.to(dtype=torch.float16)
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Native ComfyUI MoGe safetensors did not match the expected MoGe v2 ViT-L architecture.\n"
            f"Missing keys: {load_result.missing_keys[:12]}\n"
            f"Unexpected keys: {load_result.unexpected_keys[:12]}"
        )
    model.eval()
    if hasattr(model, "enable_pytorch_native_sdpa"):
        model.enable_pytorch_native_sdpa()
    return model


def _device_from_to_args(args, kwargs) -> torch.device | None:
    if "device" in kwargs and kwargs["device"] is not None:
        return torch.device(kwargs["device"])
    for arg in args:
        if isinstance(arg, torch.Tensor):
            return arg.device
        if isinstance(arg, torch.device):
            return arg
        if isinstance(arg, str):
            return torch.device(arg)
        if isinstance(arg, int):
            return torch.device("cuda", arg)
    return None


def _module_unique_tensors(modules: list[torch.nn.Module]) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    seen: set[int] = set()
    for module in modules:
        for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
            if tensor is None:
                continue
            key = id(tensor)
            if key in seen:
                continue
            seen.add(key)
            tensors.append(tensor)
    return tensors


class _ModuleResidencyVBar:
    page_size = 32 * 1024 * 1024

    def __init__(self, modules: list[torch.nn.Module]):
        self.tensors = _module_unique_tensors(modules)
        self.total_size = sum(tensor.nelement() * tensor.element_size() for tensor in self.tensors)

    @property
    def offset(self) -> int:
        return self.total_size

    def _cuda_size(self) -> int:
        return sum(
            tensor.nelement() * tensor.element_size()
            for tensor in self.tensors
            if tensor.device.type == "cuda"
        )

    def loaded_size(self) -> int:
        return self._cuda_size()

    def get_watermark(self) -> int:
        return self.loaded_size()

    def get_residency(self) -> list[int]:
        cuda = self._cuda_size()
        if self.total_size <= 0:
            return []
        pages = max(1, math.ceil(self.total_size / self.page_size))
        cuda_pages = min(pages, math.ceil(cuda / self.page_size)) if cuda > 0 else 0
        return [1] * cuda_pages + [0] * (pages - cuda_pages)


class Pixal3DTorchWrapper(torch.nn.Module):
    def __init__(self, pipeline, moge_model, vram_mode: str):
        super().__init__()
        self.pipeline = pipeline
        self.vram_mode = vram_mode
        self.active_device = torch.device("cpu")
        self.dynamic_vbars = {}
        self.pipeline_modules = torch.nn.ModuleDict(
            {_sanitize_module_name(name): module for name, module in pipeline.models.items()}
        )
        for attr in (
            "image_cond_model_ss",
            "image_cond_model_shape_512",
            "image_cond_model_shape_1024",
            "image_cond_model_tex_1024",
        ):
            module = getattr(pipeline, attr, None)
            if isinstance(module, torch.nn.Module):
                self.add_module(attr, module)
        if isinstance(getattr(pipeline, "rembg_model", None), torch.nn.Module):
            self.add_module("rembg_model", pipeline.rembg_model)
        if isinstance(moge_model, torch.nn.Module):
            self.moge_model = moge_model
        else:
            self.moge_model = None
        self._install_helper_vbars()

    def _install_helper_vbars(self) -> None:
        rembg = getattr(self, "rembg_model", None)
        if isinstance(rembg, torch.nn.Module):
            self.dynamic_vbars["Pixal3D RMBG helper"] = _ModuleResidencyVBar([rembg])
        if isinstance(self.moge_model, torch.nn.Module):
            self.dynamic_vbars["Pixal3D MoGe helper"] = _ModuleResidencyVBar([self.moge_model])

    def to(self, *args, **kwargs):
        device = _device_from_to_args(args, kwargs)
        if device is not None:
            self.active_device = device
            self.pipeline._device = device
        if _uses_low_vram_staging(self.vram_mode):
            return self
        result = super().to(*args, **kwargs)
        if device is not None:
            self.active_device = device
            self.pipeline._device = device
        return result

    def activate_device(self, device: torch.device) -> None:
        self.active_device = torch.device(device)
        if _uses_low_vram_staging(self.vram_mode):
            _offload_low_vram_pipeline_modules(self.pipeline)
            if self.moge_model is not None:
                self.moge_model.cpu()
            _release_cached_memory()
        self.pipeline._device = self.active_device

    def forward(self, *args, **kwargs):
        raise RuntimeError("Pixal3DTorchWrapper is a Comfy model-management wrapper, not a direct nn.Module.")

    def get_dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32

    def release(self) -> None:
        """Break Python references so stale loader settings do not keep RAM alive."""
        pipeline = getattr(self, "pipeline", None)
        try:
            self.cpu()
        except Exception:
            try:
                if pipeline is not None and hasattr(pipeline, "cpu"):
                    pipeline.cpu()
            except Exception:
                LOGGER.debug("Could not move Pixal3D pipeline to CPU before release", exc_info=True)

        if pipeline is not None:
            for attr in (
                "image_cond_model_ss",
                "image_cond_model_shape_512",
                "image_cond_model_shape_1024",
                "image_cond_model_tex_1024",
                "rembg_model",
            ):
                try:
                    setattr(pipeline, attr, None)
                except Exception:
                    pass
            try:
                pipeline.models.clear()
            except Exception:
                pass

        try:
            self.dynamic_vbars.clear()
        except Exception:
            self.dynamic_vbars = {}
        self.pipeline = None
        self.moge_model = None
        self.pipeline_modules = torch.nn.ModuleDict()
        for name in list(self._modules.keys()):
            self._modules.pop(name, None)


@dataclass
class Pixal3DHandle:
    patcher: comfy.model_patcher.CoreModelPatcher
    model_repo: str
    moge_repo: str
    attention_backend: str
    resolved_attention_backend: str
    vram_mode: str
    source_root: Path
    destroyed: bool = False

    def load_for_inference(self, memory_required: int = 0) -> Pixal3DTorchWrapper:
        if self.destroyed:
            raise RuntimeError("This Pixal3D model handle was unloaded. Run the Pixal3D Model Loader again.")
        prune_stale_loaded_models()
        wrapper = self.patcher.model
        if not _uses_comfy_model_management(self.vram_mode):
            device = model_management.get_torch_device()
            if memory_required:
                try:
                    model_management.free_memory(memory_required, device)
                except Exception:
                    LOGGER.debug("Could not pre-free ComfyUI memory for native Pixal3D low-VRAM run", exc_info=True)
            wrapper.activate_device(device)
            return wrapper
        model_management.load_models_gpu(
            [self.patcher],
            memory_required=memory_required,
            force_full_load=(self.vram_mode == "full_gpu"),
        )
        wrapper.activate_device(model_management.get_torch_device())
        return wrapper

    def offload(self) -> None:
        try:
            patcher = getattr(self, "patcher", None)
            if patcher is not None:
                unloaded = remove_loaded_model_entries_for_patcher(patcher)
                if not unloaded:
                    patcher.detach()
        finally:
            release_pixal3d_runtime_memory()

    def destroy(self) -> None:
        if self.destroyed:
            return
        try:
            self.offload()
        finally:
            patcher = getattr(self, "patcher", None)
            wrapper = getattr(patcher, "model", None) if patcher is not None else None
            if isinstance(wrapper, Pixal3DTorchWrapper):
                wrapper.release()
            if patcher is not None:
                try:
                    patcher.cleanup()
                except Exception:
                    pass
            self.destroyed = True
            release_pixal3d_runtime_memory(aggressive=True)


@dataclass
class Pixal3DResult:
    mesh: Any
    shape_slat: Any
    tex_slat: Any
    resolution: int
    attr_layout: dict[str, slice]
    camera_params: dict[str, float]
    rembg_image: Image.Image | None = None
    rembg_used: bool = False


def prune_stale_loaded_models() -> None:
    loaded = getattr(model_management, "current_loaded_models", None)
    if loaded is None:
        return
    for loaded_model in list(loaded):
        try:
            patcher = getattr(loaded_model, "model", None)
            real_model = getattr(patcher, "model", None) if patcher is not None else None
            real_ref = getattr(loaded_model, "real_model", None)
            loaded_real_model = real_ref() if callable(real_ref) else None
            model_finalizer = getattr(loaded_model, "model_finalizer", None)
        except Exception:
            patcher = None
            real_model = None
            loaded_real_model = None
            model_finalizer = None
        if patcher is None or real_model is None or loaded_real_model is None or model_finalizer is None:
            for attr in ("model_finalizer", "_patcher_finalizer"):
                finalizer = getattr(loaded_model, attr, None)
                if finalizer is not None:
                    try:
                        finalizer.detach()
                    except Exception:
                        pass
                    try:
                        setattr(loaded_model, attr, None)
                    except Exception:
                        pass
            try:
                loaded.remove(loaded_model)
            except ValueError:
                pass


def release_pixal3d_runtime_memory(aggressive: bool = False) -> None:
    prune_stale_loaded_models()
    _release_cached_memory(aggressive=aggressive)


def remove_loaded_model_entries_for_patcher(patcher) -> bool:
    loaded = getattr(model_management, "current_loaded_models", None)
    if loaded is None:
        return False
    removed = False
    for loaded_model in list(loaded):
        if getattr(loaded_model, "model", None) is patcher:
            try:
                model_unload = getattr(loaded_model, "model_unload", None)
                if callable(model_unload):
                    try:
                        model_unload()
                    except AttributeError as exc:
                        if "NoneType" not in str(exc):
                            raise
                        LOGGER.debug("Ignoring already-detached Pixal3D loaded model entry during offload.")
                elif getattr(patcher, "model", None) is not None:
                    patcher.detach()
            finally:
                try:
                    loaded.remove(loaded_model)
                    removed = True
                except ValueError:
                    pass
    return removed


def estimate_inference_memory_required(pipeline_type: str, max_num_tokens: int) -> int:
    base_gb = 6 if pipeline_type == "1536_cascade" else 4
    token_gb = max(0, int(max_num_tokens) - 49152) / 49152
    return int((base_gb + token_gb * 2) * 1024**3)


def count_comfy_cast_modules(module: torch.nn.Module) -> tuple[int, int]:
    castable = 0
    parameterized = 0
    for child in module.modules():
        if any(True for _ in child.parameters(recurse=False)):
            parameterized += 1
            if hasattr(child, "comfy_cast_weights"):
                castable += 1
    return castable, parameterized


def load_pixal3d_model(
    model_repo: str = DEFAULT_MODEL_REPO,
    moge_repo: str = DEFAULT_MOGE_REPO,
    attention_backend: str = "auto",
    vram_mode: str = "dynamic_vram",
    download_if_missing: bool = False,
    load_moge: bool = True,
    load_rembg: bool = False,
    naf_mode: str = "fallback_if_missing",
    naf_target_size: str = "upstream",
    preload_naf: bool = False,
    hf_endpoint: str = "",
    pixal3d_repo_path: str = "",
) -> Pixal3DHandle:
    if vram_mode not in VRAM_MODE_CHOICES:
        raise ValueError(f"Unsupported VRAM mode: {vram_mode}")
    if naf_mode not in NAF_MODE_CHOICES:
        raise ValueError(f"Unsupported NAF mode: {naf_mode}")
    if naf_target_size not in NAF_TARGET_SIZE_CHOICES:
        raise ValueError(f"Unsupported NAF target size: {naf_target_size}")
    source_root = ensure_pixal3d_source(pixal3d_repo_path)
    configure_torch_hub_cache()
    ensure_cuda_aliases()
    use_comfy_ops = _uses_comfy_ops(vram_mode)
    use_low_vram_staging = _uses_low_vram_staging(vram_mode)
    if use_comfy_ops:
        LOGGER.info("Building Pixal3D with Comfy/Aimdo-aware torch.nn ops for DynamicVRAM.")
    if vram_mode == "hybrid_low_vram":
        LOGGER.info("Using hybrid low-VRAM mode: Comfy/Aimdo-aware modules with native Pixal3D stage offload.")

    model_path = resolve_model_path(model_repo, download_if_missing, hf_endpoint)
    with _temporary_comfy_ops(use_comfy_ops):
        resolved_backend = apply_attention_backend(attention_backend)
        from pixal3d.pipelines import Pixal3DImageTo3DPipeline

        LOGGER.info("Loading Pixal3D pipeline from %s with %s", model_path, resolved_backend)
        pipeline_config = prepare_pipeline_config(model_path, download_if_missing, load_rembg, hf_endpoint)
        pipeline = Pixal3DImageTo3DPipeline.from_pretrained(model_path, config_file=pipeline_config)
        pipeline.low_vram = use_low_vram_staging

        pipeline.image_cond_model_ss = build_image_cond_model(
            IMAGE_COND_CONFIGS["ss"],
            download_if_missing,
            hf_endpoint,
            naf_mode,
            naf_target_size,
            use_low_vram_staging,
        )
        pipeline.image_cond_model_shape_512 = build_image_cond_model(
            IMAGE_COND_CONFIGS["shape_512"],
            download_if_missing,
            hf_endpoint,
            naf_mode,
            naf_target_size,
            use_low_vram_staging,
        )
        pipeline.image_cond_model_shape_1024 = build_image_cond_model(
            IMAGE_COND_CONFIGS["shape_1024"],
            download_if_missing,
            hf_endpoint,
            naf_mode,
            naf_target_size,
            use_low_vram_staging,
        )
        pipeline.image_cond_model_tex_1024 = build_image_cond_model(
            IMAGE_COND_CONFIGS["tex_1024"],
            download_if_missing,
            hf_endpoint,
            naf_mode,
            naf_target_size,
            use_low_vram_staging,
        )

    if preload_naf and naf_mode != "strict":
        LOGGER.info("Skipping Pixal3D NAF preload because naf_mode=%s uses fallback when libnatten is unavailable", naf_mode)
    elif preload_naf:
        for attr in ("image_cond_model_shape_512", "image_cond_model_shape_1024", "image_cond_model_tex_1024"):
            model = getattr(pipeline, attr, None)
            if model is not None and getattr(model, "use_naf_upsample", False):
                LOGGER.info("Preloading Pixal3D NAF upsampler for %s", attr)
                model._load_naf()

    moge_model = load_moge_model(moge_repo, download_if_missing, hf_endpoint) if load_moge else None
    wrapper = Pixal3DTorchWrapper(pipeline, moge_model, vram_mode).eval()
    if use_comfy_ops:
        castable, parameterized = count_comfy_cast_modules(wrapper)
        LOGGER.info("Pixal3D DynamicVRAM Aimdo-aware modules: %s/%s parameterized modules", castable, parameterized)
    model_management.archive_model_dtypes(wrapper)
    load_device = model_management.get_torch_device()
    patcher = comfy.model_patcher.CoreModelPatcher(
        wrapper,
        load_device=load_device,
        offload_device=model_management.unet_offload_device(),
    )
    return Pixal3DHandle(
        patcher=patcher,
        model_repo=model_path,
        moge_repo=moge_repo,
        attention_backend=attention_backend,
        resolved_attention_backend=resolved_backend,
        vram_mode=vram_mode,
        source_root=source_root,
    )


def compute_f_pixels(camera_angle_x: float, resolution: int) -> float:
    focal_length = 16.0 / torch.tan(torch.tensor(camera_angle_x / 2.0))
    f_pixels = focal_length * resolution / 32.0
    return float(f_pixels.item())


def distance_from_fov(camera_angle_x, grid_point, target_point, mesh_scale, image_resolution):
    rotation_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    gp = grid_point.to(torch.float32) @ rotation_matrix.T
    gp = gp / mesh_scale / 2
    xw, yw = gp[0].item(), gp[1].item()
    xt, yt = float(target_point[0].item()), float(target_point[1].item())
    f_pixels = compute_f_pixels(camera_angle_x, image_resolution)
    x_ndc = xt - image_resolution / 2.0
    distance_x = f_pixels * xw / x_ndc - yw
    return {"distance_from_x": float(distance_x), "f_pixels": float(f_pixels)}


def estimate_camera_params(
    pil_image: Image.Image,
    moge_model: torch.nn.Module,
    device: torch.device,
    mesh_scale: float,
    extend_pixel: int,
    image_resolution: int,
    offload_after: bool,
) -> dict[str, float]:
    if moge_model is None:
        raise RuntimeError("Camera mode is MoGe, but the Pixal3D model was loaded with load_moge=False.")
    moge_model.to(device)
    width, height = pil_image.size
    image_np = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(device)
    with torch.no_grad():
        output = moge_model.infer(image_tensor)
    if offload_after:
        moge_model.cpu()
        _release_cached_memory()
    intrinsics = output["intrinsics"].squeeze().detach().cpu().numpy()
    fx = float(intrinsics[0, 0] * width)
    camera_angle_x = 2 * math.atan(width / (2 * fx))
    distance = distance_from_fov(
        camera_angle_x,
        torch.tensor([-1.0, 0.0, 0.0]),
        torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
        mesh_scale,
        image_resolution,
    )["distance_from_x"]
    return {"camera_angle_x": float(camera_angle_x), "distance": float(distance), "mesh_scale": float(mesh_scale)}


def run_pixal3d(
    handle: Pixal3DHandle,
    image: Image.Image,
    seed: int = 42,
    pipeline_type: str = "1024_cascade",
    background_mode: str = "auto_remove",
    camera_mode: str = "moge",
    manual_camera_angle_x: float = 0.8575560450553894,
    manual_distance: float = 2.0,
    mesh_scale: float = 1.0,
    extend_pixel: int = 0,
    camera_resolution: int = 512,
    steps: int = 12,
    guidance: float = 7.5,
    texture_guidance: float = 1.0,
    max_num_tokens: int = 49152,
    force_offload: bool = False,
    node_id: str | None = None,
) -> Pixal3DResult:
    import comfy.utils

    pbar = comfy.utils.ProgressBar(5, node_id=node_id)
    wrapper_preflight = getattr(getattr(handle, "patcher", None), "model", None)
    pipeline_preflight = getattr(wrapper_preflight, "pipeline", None)
    if pipeline_preflight is not None:
        if background_mode == "auto_remove" and _rembg_is_disabled(pipeline_preflight):
            raise RuntimeError(
                "background_mode=auto_remove needs the RMBG helper model. "
                "Set Pixal3D Model Loader load_rembg=true and provide/download briaai/RMBG-2.0, "
                "or use background_mode=none / keep_alpha."
            )
        if background_mode == "keep_alpha" and image.mode != "RGBA" and _rembg_is_disabled(pipeline_preflight):
            raise RuntimeError(
                "background_mode=keep_alpha received no alpha channel and RMBG is disabled. "
                "Use an RGBA image with alpha, use background_mode=none, or enable load_rembg."
            )

    wrapper = handle.load_for_inference(
        memory_required=estimate_inference_memory_required(pipeline_type, int(max_num_tokens))
    )
    pipeline = wrapper.pipeline
    device = wrapper.active_device

    try:
        model_management.throw_exception_if_processing_interrupted()
        rembg_used = False
        if background_mode == "auto_remove":
            if _rembg_is_disabled(pipeline):
                raise RuntimeError(
                    "background_mode=auto_remove needs the RMBG helper model. "
                    "Either set Pixal3D Model Loader load_rembg=true and download/provide briaai/RMBG-2.0, "
                    "or use background_mode=none / keep_alpha."
                )
            rembg_used = not _pil_has_useful_alpha(image)
            _move_rembg_model(pipeline, device)
            image_preprocessed = pipeline.preprocess_image(image.convert("RGBA" if image.mode == "RGBA" else "RGB"))
            if _uses_low_vram_staging(handle.vram_mode):
                _release_cached_memory()
        elif background_mode == "keep_alpha":
            if not _pil_has_useful_alpha(image):
                if _rembg_is_disabled(pipeline):
                    raise RuntimeError(
                        "background_mode=keep_alpha received no alpha channel and RMBG is disabled. "
                        "Use an RGBA image with alpha, use background_mode=none, or enable load_rembg."
                    )
                LOGGER.warning("background_mode=keep_alpha received no usable alpha channel; using auto_remove.")
                rembg_used = True
                _move_rembg_model(pipeline, device)
                image_preprocessed = pipeline.preprocess_image(image.convert("RGBA" if image.mode == "RGBA" else "RGB"))
                if _uses_low_vram_staging(handle.vram_mode):
                    _release_cached_memory()
            else:
                image_preprocessed = pipeline.preprocess_image(image.convert("RGBA"))
        elif background_mode == "none":
            image_preprocessed = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported background mode: {background_mode}")
        image_preprocessed = _pad_to_square(image_preprocessed).convert("RGB")
        pbar.update_absolute(1, 5)

        if camera_mode == "moge":
            camera_params = estimate_camera_params(
                image_preprocessed,
                wrapper.moge_model,
                device,
                mesh_scale,
                extend_pixel,
                camera_resolution,
                offload_after=_uses_low_vram_staging(handle.vram_mode),
            )
        elif camera_mode == "manual":
            camera_params = {
                "camera_angle_x": float(manual_camera_angle_x),
                "distance": float(manual_distance),
                "mesh_scale": float(mesh_scale),
            }
        else:
            raise ValueError(f"Unsupported camera mode: {camera_mode}")
        pbar.update_absolute(2, 5)

        torch.manual_seed(int(seed))
        ss_sampler_override = {
            "steps": int(steps),
            "guidance_strength": float(guidance),
            "guidance_rescale": 0.7,
            "rescale_t": 5.0,
        }
        shape_sampler_override = {
            "steps": int(steps),
            "guidance_strength": float(guidance),
            "guidance_rescale": 0.5,
            "rescale_t": 3.0,
        }
        tex_sampler_override = {
            "steps": int(steps),
            "guidance_strength": float(texture_guidance),
            "guidance_rescale": 0.0,
            "rescale_t": 3.0,
        }
        pbar.update_absolute(3, 5)

        mesh_list, (shape_slat, tex_slat, resolution) = pipeline.run(
            image_preprocessed,
            camera_params=camera_params,
            seed=int(seed),
            sparse_structure_sampler_params=ss_sampler_override,
            shape_slat_sampler_params=shape_sampler_override,
            tex_slat_sampler_params=tex_sampler_override,
            preprocess_image=False,
            return_latent=True,
            pipeline_type=pipeline_type,
            max_num_tokens=int(max_num_tokens),
        )
        pbar.update_absolute(5, 5)
        mesh = mesh_list[0]
        result = Pixal3DResult(
            mesh=mesh,
            shape_slat=None,
            tex_slat=None,
            resolution=int(resolution),
            attr_layout=dict(pipeline.pbr_attr_layout),
            camera_params=camera_params,
            rembg_image=image_preprocessed.copy(),
            rembg_used=rembg_used,
        )
        del shape_slat, tex_slat, mesh_list
        _release_cached_memory()
        return result
    except torch.OutOfMemoryError:
        LOGGER.warning("Pixal3D hit CUDA OOM; offloading the active handle before re-raising.")
        handle.offload()
        _release_cached_memory()
        raise
    finally:
        if force_offload:
            handle.offload()


def export_glb(
    result: Pixal3DResult,
    decimation_target: int = 1000000,
    texture_size: int = 4096,
    remesh: bool = True,
    filename_prefix: str = "pixal3d",
) -> str:
    ensure_cuda_aliases()
    import o_voxel

    model_management.throw_exception_if_processing_interrupted()
    mesh = result.mesh
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=result.attr_layout,
        grid_size=result.resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=int(decimation_target),
        texture_size=int(texture_size),
        remesh=bool(remesh),
        remesh_band=1,
        remesh_project=0,
        use_tqdm=True,
    )

    rot = np.array(
        [
            [-1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    glb.apply_transform(rot)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_prefix = re.sub(r"[^0-9a-zA-Z_.-]+", "_", filename_prefix).strip("._") or "pixal3d"
    output_dir = Path(folder_paths.get_output_directory())
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{safe_prefix}_{timestamp}.glb"
    glb.export(str(output_path), extension_webp=False)
    gc.collect()
    model_management.soft_empty_cache()
    return str(output_path)
