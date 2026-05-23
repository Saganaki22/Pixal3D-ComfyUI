from __future__ import annotations

import logging

__version__ = "0.2.3"

LOGGER = logging.getLogger("Pixal3D_ComfyUI")


def _install_birefnet_partial_load_patch() -> None:
    try:
        import torch
        import comfy.background_removal.birefnet as birefnet
    except Exception:
        LOGGER.debug("Native BiRefNet compatibility patch was not installed", exc_info=True)
        return

    window_attention = getattr(birefnet, "WindowAttention", None)
    if window_attention is None or getattr(window_attention, "_pixal3d_partial_load_patch", False):
        return

    def forward(self, x, mask=None):
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        bias_table = self.relative_position_bias_table.to(device=attn.device, dtype=attn.dtype)
        bias_index = self.relative_position_index.to(device=attn.device).long().view(-1)
        relative_position_bias = bias_table[bias_index].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            mask = mask.to(device=attn.device, dtype=attn.dtype)
            windows = mask.shape[0]
            attn = attn.view(batch // windows, windows, self.num_heads, tokens, tokens) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, tokens, tokens)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch, tokens, channels)
        x = self.proj(x)
        return x

    window_attention.forward = forward
    window_attention._pixal3d_partial_load_patch = True


_install_birefnet_partial_load_patch()

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception as exc:
    LOGGER.exception("Failed to import Pixal3D-ComfyUI nodes: %s", exc)
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
