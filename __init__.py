from __future__ import annotations

import logging

__version__ = "0.1.6"

LOGGER = logging.getLogger("Pixal3D_ComfyUI")

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception as exc:
    LOGGER.exception("Failed to import Pixal3D-ComfyUI nodes: %s", exc)
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
