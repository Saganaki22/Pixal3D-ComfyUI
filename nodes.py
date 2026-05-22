from __future__ import annotations

import logging
import math
import os

from .pixal3d_comfy import (
    ATTENTION_CHOICES,
    DEFAULT_MOGE_REPO,
    DEFAULT_MODEL_REPO,
    NAF_MODE_CHOICES,
    NAF_TARGET_SIZE_CHOICES,
    VRAM_MODE_CHOICES,
    environment_report,
    export_glb,
    load_pixal3d_model,
    pil_to_tensor,
    release_pixal3d_runtime_memory,
    run_pixal3d,
    tensor_to_pil,
)

LOGGER = logging.getLogger("Pixal3D_ComfyUI")

_LOADED_MODELS = {}

TOOLTIPS = {
    "model_repo": "Hugging Face repo id or a local folder containing pipeline.json. Default downloads/loads TencentARC/Pixal3D.",
    "hf_endpoint": "Hugging Face endpoint used only when download_if_missing is enabled. Default is https://huggingface.co; Chinese users can use https://hf-mirror.com.",
    "attention_backend": "auto uses FlashAttention 3 if flash_attn_interface imports, otherwise FlashAttention 2 if flash_attn imports.",
    "vram_mode": "dynamic_vram uses Comfy/Aimdo-aware ops. hybrid_low_vram adds native Pixal3D stage offload. native_low_vram uses stage offload without Comfy/Aimdo ops. full_gpu keeps modules resident.",
    "download_if_missing": "When enabled, downloads Pixal3D/helper models into ComfyUI/models/Pixal3D and native MoGe files into ComfyUI/models/geometry_estimation. When disabled, no model downloads are attempted.",
    "load_moge": "Load MoGe for camera_mode=moge. Disable only if you will use manual camera mode.",
    "load_rembg": "Load the gated briaai/RMBG-2.0 helper model for background_mode=auto_remove. Requires local files or download_if_missing with HF access/token.",
    "naf_mode": "fallback_if_missing keeps Pixal3D's required 2048-channel shape by duplicating DINO features if CUDA NATTEN/NAF is unavailable. strict requires real NAF.",
    "naf_target_size": "Target size for real NAF upsampling. upstream keeps Pixal3D defaults; lower values reduce VRAM. Ignored by fallback_if_missing.",
    "preload_naf": "Preload Pixal3D's NAF upsampler during model load. Only useful with naf_mode=strict and CUDA NATTEN/libnatten available.",
    "force_reload": "Ignore the cached model handle and rebuild it from disk.",
    "model": "Pixal3D model handle from Pixal3D Model Loader. Re-run the loader after changing backend, VRAM, helper-model, or NAF settings.",
    "image": "Input subject image. RGBA alpha is respected when background_mode is keep_alpha or auto_remove.",
    "seed": "Random seed for all Pixal3D sampling stages.",
    "pipeline_type": "1024_cascade is the normal path. 1536_cascade can improve detail but needs more VRAM and may lower resolution if token count is too high.",
    "background_mode": "auto_remove uses Pixal3D/rembg unless alpha exists. keep_alpha prefers alpha. none skips background removal.",
    "camera_mode": "moge estimates camera/FOV from the image. manual uses manual_camera_angle_x, manual_distance, and mesh_scale unless a Pixal3D Camera Control manual_fov input is connected.",
    "manual_camera_angle_x": "Horizontal field of view in radians for manual camera mode.",
    "manual_distance": "Camera distance for manual camera mode.",
    "camera_fov_degrees": "Horizontal field of view in degrees. The camera helper converts this to manual_camera_angle_x radians.",
    "camera_distance": "Camera distance for Pixal3D manual camera mode.",
    "camera_passthrough_image": "Optional preview image for the camera widget. This node does not output an image; connect Load Image directly to Pixal3D Image To 3D.",
    "manual_fov": "Bundled manual camera values from Pixal3D Camera Control. Only used when camera_mode=manual. When connected, it overrides manual_camera_angle_x, manual_distance, and mesh_scale. Ignored when camera_mode=moge.",
    "camera_info": "Readable summary of the manual camera values produced by the camera helper.",
    "mesh_scale": "Scale used for camera fitting. Usually keep 1.0.",
    "extend_pixel": "Offsets the camera fitting target. Useful if MoGe framing is slightly too tight or loose.",
    "camera_resolution": "Resolution used for MoGe camera fitting math. 512 matches upstream defaults.",
    "steps": "Sampling steps for sparse structure, shape, and texture stages. Upstream default is 12.",
    "guidance": "Guidance strength for sparse structure and shape sampling. Upstream default is 7.5.",
    "texture_guidance": "Guidance strength for texture sampling. Upstream default is 1.0.",
    "max_num_tokens": "Caps high-resolution sparse tokens. Lower values reduce VRAM, but can reduce detail.",
    "force_offload": "Unload the Pixal3D model from Comfy model management after generation.",
    "pixal3d_result": "Pixal3D result from Pixal3D Image To 3D. Contains the decoded mesh and texture attributes needed for GLB export.",
    "rembg_image": "Preview image after Pixal3D background preprocessing. Shows the RMBG-removed image when RMBG was used.",
    "decimation_target": "Target face count for textured GLB export simplification. 1000000 matches the Pixal3D demo default; very low values such as 5000 can lose detail.",
    "texture_size": "Baked texture size for GLB export. 4096 matches the Pixal3D demo default and preserves more material detail.",
    "remesh": "Use Pixal3D/o_voxel remesh path during GLB export. When enabled, the node passes it through to o_voxel; disable it if cleanup fragments the mesh.",
    "filename_prefix": "Prefix for the exported GLB in ComfyUI/output.",
    "unload_model": "Pixal3D model handle to remove from VRAM and the Pixal3D Python cache. Use this to free CPU RAM after experiments.",
}


def _destroy_handle(handle) -> None:
    if handle is None:
        return
    try:
        destroy = getattr(handle, "destroy", None)
        if callable(destroy):
            destroy()
        else:
            handle.offload()
    except Exception:
        LOGGER.warning("Failed to destroy stale Pixal3D model handle", exc_info=True)
    finally:
        _flush_pixal3d_runtime_memory()


def _clear_model_cache(keep_key=None) -> int:
    cleared = 0
    for key, handle in list(_LOADED_MODELS.items()):
        if keep_key is not None and key == keep_key:
            continue
        _destroy_handle(handle)
        _LOADED_MODELS.pop(key, None)
        cleared += 1
    return cleared


def _flush_pixal3d_runtime_memory() -> None:
    try:
        _prune_stale_comfy_loaded_models()
        release_pixal3d_runtime_memory(aggressive=True)
        _prune_stale_comfy_loaded_models()
    except Exception:
        LOGGER.debug("Could not fully flush Pixal3D runtime memory", exc_info=True)


def _prune_stale_comfy_loaded_models() -> int:
    try:
        import comfy.model_management as model_management
    except Exception:
        return 0

    loaded = getattr(model_management, "current_loaded_models", None)
    if loaded is None:
        return 0

    removed = 0
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

        if patcher is not None and real_model is not None and loaded_real_model is not None and model_finalizer is not None:
            continue

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
            removed += 1
        except ValueError:
            pass
    return removed


def _install_global_unload_hook() -> None:
    try:
        import comfy.model_management as model_management
    except Exception:
        LOGGER.debug("Could not install Pixal3D global unload hook", exc_info=True)
        return

    original = getattr(model_management, "unload_all_models", None)
    if not callable(original) or getattr(original, "_pixal3d_hooked", False):
        return

    def unload_all_models_with_pixal3d_cache(*args, **kwargs):
        _prune_stale_comfy_loaded_models()
        try:
            result = original(*args, **kwargs)
        except AttributeError as exc:
            if "NoneType" not in str(exc):
                raise
            pruned = _prune_stale_comfy_loaded_models()
            if not pruned:
                raise
            LOGGER.warning("Pruned %s stale ComfyUI loaded model entry/entries and retried global unload.", pruned)
            result = original(*args, **kwargs)
        finally:
            cleared = _clear_model_cache()
            _flush_pixal3d_runtime_memory()
            _prune_stale_comfy_loaded_models()
            if cleared:
                LOGGER.info("Cleared %s Pixal3D cached model handle(s) after ComfyUI global unload.", cleared)
        return result

    unload_all_models_with_pixal3d_cache._pixal3d_hooked = True
    unload_all_models_with_pixal3d_cache._pixal3d_original = original
    model_management.unload_all_models = unload_all_models_with_pixal3d_cache


_install_global_unload_hook()


class Pixal3DModelLoader:
    DESCRIPTION = "Loads Pixal3D as a Comfy-managed model handle. Use Pixal3D Environment Check first if you are unsure about CUDA wheels."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_repo": ("STRING", {"default": DEFAULT_MODEL_REPO, "tooltip": TOOLTIPS["model_repo"]}),
                "hf_endpoint": ("STRING", {"default": "https://huggingface.co", "tooltip": TOOLTIPS["hf_endpoint"]}),
                "attention_backend": (ATTENTION_CHOICES, {"default": "auto", "tooltip": TOOLTIPS["attention_backend"]}),
                "vram_mode": (VRAM_MODE_CHOICES, {"default": "dynamic_vram", "tooltip": TOOLTIPS["vram_mode"]}),
                "download_if_missing": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["download_if_missing"]}),
                "load_moge": ("BOOLEAN", {"default": True, "tooltip": TOOLTIPS["load_moge"]}),
                "load_rembg": ("BOOLEAN", {"default": True, "tooltip": TOOLTIPS["load_rembg"]}),
                "naf_mode": (NAF_MODE_CHOICES, {"default": "fallback_if_missing", "tooltip": TOOLTIPS["naf_mode"]}),
                "naf_target_size": (NAF_TARGET_SIZE_CHOICES, {"default": "upstream", "tooltip": TOOLTIPS["naf_target_size"]}),
                "preload_naf": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["preload_naf"]}),
                "force_reload": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["force_reload"]}),
            }
        }

    RETURN_TYPES = ("PIXAL3D_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "Pixal3D"

    def load(
        self,
        model_repo,
        hf_endpoint,
        attention_backend,
        vram_mode,
        download_if_missing,
        load_moge,
        load_rembg,
        naf_mode="fallback_if_missing",
        naf_target_size="upstream",
        preload_naf=False,
        force_reload=False,
        moge_repo=DEFAULT_MOGE_REPO,
    ):
        pixal3d_repo_path = os.environ.get("PIXAL3D_REPO_PATH", "").strip()
        native_moge_repo = DEFAULT_MOGE_REPO
        key = (
            model_repo,
            native_moge_repo,
            hf_endpoint,
            attention_backend,
            vram_mode,
            bool(download_if_missing),
            bool(load_moge),
            bool(load_rembg),
            naf_mode,
            naf_target_size,
            bool(preload_naf),
            pixal3d_repo_path,
        )
        if force_reload or key not in _LOADED_MODELS:
            cleared = _clear_model_cache()
            _flush_pixal3d_runtime_memory()
            if cleared:
                LOGGER.info("Cleared %s Pixal3D cached model handle(s) before loading changed settings.", cleared)
            _LOADED_MODELS[key] = load_pixal3d_model(
                model_repo=model_repo.strip() or DEFAULT_MODEL_REPO,
                moge_repo=native_moge_repo,
                hf_endpoint=hf_endpoint.strip(),
                attention_backend=attention_backend,
                vram_mode=vram_mode,
                download_if_missing=bool(download_if_missing),
                load_moge=bool(load_moge),
                load_rembg=bool(load_rembg),
                naf_mode=naf_mode,
                naf_target_size=naf_target_size,
                preload_naf=bool(preload_naf),
                pixal3d_repo_path=pixal3d_repo_path,
            )
        else:
            _clear_model_cache(keep_key=key)
            _flush_pixal3d_runtime_memory()
        return (_LOADED_MODELS[key],)


class Pixal3DEnvironmentCheck:
    DESCRIPTION = "Checks the current ComfyUI Python environment for torch, CUDA, FlashAttention, Triton, and Pixal3D CUDA extension imports."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "check"
    OUTPUT_NODE = True
    CATEGORY = "Pixal3D"

    def check(self):
        report = environment_report()
        return {"ui": {"text": [report]}, "result": (report,)}


class Pixal3DImageTo3D:
    DESCRIPTION = "Generates Pixal3D geometry and texture data from an input IMAGE. Use Export GLB for Preview3D."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PIXAL3D_MODEL", {"tooltip": TOOLTIPS["model"]}),
                "image": ("IMAGE", {"tooltip": TOOLTIPS["image"]}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "tooltip": TOOLTIPS["seed"]}),
                "pipeline_type": (["1024_cascade", "1536_cascade"], {"default": "1536_cascade", "tooltip": TOOLTIPS["pipeline_type"]}),
                "background_mode": (["auto_remove", "keep_alpha", "none"], {"default": "auto_remove", "tooltip": TOOLTIPS["background_mode"]}),
                "camera_mode": (["moge", "manual"], {"default": "moge", "tooltip": TOOLTIPS["camera_mode"]}),
                "manual_camera_angle_x": ("FLOAT", {"default": 0.857556, "min": 0.1, "max": 3.0, "step": 0.001, "tooltip": TOOLTIPS["manual_camera_angle_x"]}),
                "manual_distance": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 20.0, "step": 0.01, "tooltip": TOOLTIPS["manual_distance"]}),
                "mesh_scale": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 10.0, "step": 0.05, "tooltip": TOOLTIPS["mesh_scale"]}),
                "extend_pixel": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1, "tooltip": TOOLTIPS["extend_pixel"]}),
                "camera_resolution": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64, "tooltip": TOOLTIPS["camera_resolution"]}),
                "steps": ("INT", {"default": 12, "min": 1, "max": 100, "step": 1, "tooltip": TOOLTIPS["steps"]}),
                "guidance": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1, "tooltip": TOOLTIPS["guidance"]}),
                "texture_guidance": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1, "tooltip": TOOLTIPS["texture_guidance"]}),
                "max_num_tokens": ("INT", {"default": 49152, "min": 4096, "max": 200000, "step": 1024, "tooltip": TOOLTIPS["max_num_tokens"]}),
                "force_offload": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS["force_offload"]}),
            },
            "optional": {
                "manual_fov": ("PIXAL3D_CAMERA", {"tooltip": TOOLTIPS["manual_fov"]}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("PIXAL3D_RESULT", "IMAGE")
    RETURN_NAMES = ("pixal3d_result", "rembg_image")
    FUNCTION = "generate"
    CATEGORY = "Pixal3D"

    def generate(
        self,
        model,
        image,
        seed,
        pipeline_type,
        background_mode,
        camera_mode,
        manual_camera_angle_x,
        manual_distance,
        mesh_scale,
        extend_pixel,
        camera_resolution,
        steps,
        guidance,
        texture_guidance,
        max_num_tokens,
        force_offload,
        unique_id=None,
        manual_fov=None,
    ):
        if camera_mode == "manual" and isinstance(manual_fov, dict):
            manual_camera_angle_x = manual_fov.get("manual_camera_angle_x", manual_camera_angle_x)
            manual_distance = manual_fov.get("manual_distance", manual_distance)
            mesh_scale = manual_fov.get("mesh_scale", mesh_scale)
        pil_image = tensor_to_pil(image)
        result = run_pixal3d(
            model,
            pil_image,
            seed=seed,
            pipeline_type=pipeline_type,
            background_mode=background_mode,
            camera_mode=camera_mode,
            manual_camera_angle_x=manual_camera_angle_x,
            manual_distance=manual_distance,
            mesh_scale=mesh_scale,
            extend_pixel=extend_pixel,
            camera_resolution=camera_resolution,
            steps=steps,
            guidance=guidance,
            texture_guidance=texture_guidance,
            max_num_tokens=max_num_tokens,
            force_offload=force_offload,
            node_id=str(unique_id) if unique_id is not None else None,
        )
        rembg_image = getattr(result, "rembg_image", None) or pil_image
        return (result, pil_to_tensor(rembg_image))


class Pixal3DCameraControl:
    DESCRIPTION = "Interactive manual camera helper. Outputs one bundled manual_fov value for Pixal3D Image To 3D."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fov_degrees": ("FLOAT", {"default": 49.134, "min": 5.0, "max": 140.0, "step": 0.1, "tooltip": TOOLTIPS["camera_fov_degrees"]}),
                "distance": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 20.0, "step": 0.01, "tooltip": TOOLTIPS["camera_distance"]}),
                "mesh_scale": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 10.0, "step": 0.05, "tooltip": TOOLTIPS["mesh_scale"]}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": TOOLTIPS["camera_passthrough_image"]}),
            },
        }

    RETURN_TYPES = ("PIXAL3D_CAMERA",)
    RETURN_NAMES = ("manual_fov",)
    FUNCTION = "camera"
    CATEGORY = "Pixal3D"

    def camera(self, fov_degrees, distance, mesh_scale, image=None):
        fov_degrees = max(5.0, min(140.0, float(fov_degrees)))
        distance = max(0.1, float(distance))
        mesh_scale = max(0.05, float(mesh_scale))
        camera_angle_x = math.radians(fov_degrees)
        info = (
            f"manual camera: manual_camera_angle_x={camera_angle_x:.6f} rad "
            f"({fov_degrees:.3f} deg), manual_distance={distance:.3f}, mesh_scale={mesh_scale:.3f}"
        )
        manual_fov = {
            "fov_degrees": fov_degrees,
            "manual_camera_angle_x": camera_angle_x,
            "manual_distance": distance,
            "mesh_scale": mesh_scale,
            "camera_info": info,
        }
        return (manual_fov,)


class Pixal3DExportGLB:
    DESCRIPTION = "Exports a textured GLB from Pixal3D output. Connect glb_path to ComfyUI Preview3D model_file."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixal3d_result": ("PIXAL3D_RESULT", {"tooltip": TOOLTIPS["pixal3d_result"]}),
                "decimation_target": ("INT", {"default": 1000000, "min": 5000, "max": 5000000, "step": 5000, "tooltip": TOOLTIPS["decimation_target"]}),
                "texture_size": ("INT", {"default": 4096, "min": 512, "max": 8192, "step": 512, "tooltip": TOOLTIPS["texture_size"]}),
                "remesh": ("BOOLEAN", {"default": True, "tooltip": TOOLTIPS["remesh"]}),
                "filename_prefix": ("STRING", {"default": "pixal3d", "tooltip": TOOLTIPS["filename_prefix"]}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    FUNCTION = "export"
    OUTPUT_NODE = True
    CATEGORY = "Pixal3D"

    def export(self, pixal3d_result, decimation_target, texture_size, remesh, filename_prefix):
        glb_path = export_glb(
            pixal3d_result,
            decimation_target=decimation_target,
            texture_size=texture_size,
            remesh=remesh,
            filename_prefix=filename_prefix,
        )
        return {"ui": {"text": [glb_path]}, "result": (glb_path,)}


class Pixal3DUnloadModel:
    DESCRIPTION = "Fully unloads the Pixal3D model handle from ComfyUI model management and the Pixal3D Python cache."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("PIXAL3D_MODEL", {"tooltip": TOOLTIPS["unload_model"]})}}

    RETURN_TYPES = ()
    FUNCTION = "unload"
    OUTPUT_NODE = True
    CATEGORY = "Pixal3D"

    def unload(self, model):
        removed = 0
        for key, handle in list(_LOADED_MODELS.items()):
            if handle is model:
                _LOADED_MODELS.pop(key, None)
                removed += 1
        _destroy_handle(model)
        _flush_pixal3d_runtime_memory()
        suffix = f" ({removed} cached handle removed)" if removed else ""
        return {"ui": {"text": [f"Pixal3D model unloaded from VRAM and Python cache{suffix}"]}, "result": ()}


NODE_CLASS_MAPPINGS = {
    "Pixal3DEnvironmentCheck": Pixal3DEnvironmentCheck,
    "Pixal3DModelLoader": Pixal3DModelLoader,
    "Pixal3DCameraControl": Pixal3DCameraControl,
    "Pixal3DImageTo3D": Pixal3DImageTo3D,
    "Pixal3DExportGLB": Pixal3DExportGLB,
    "Pixal3DUnloadModel": Pixal3DUnloadModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Pixal3DEnvironmentCheck": "Pixal3D Environment Check",
    "Pixal3DModelLoader": "Pixal3D Model Loader",
    "Pixal3DCameraControl": "Pixal3D Camera Control",
    "Pixal3DImageTo3D": "Pixal3D Image To 3D",
    "Pixal3DExportGLB": "Pixal3D Export GLB",
    "Pixal3DUnloadModel": "Pixal3D Unload Model",
}
