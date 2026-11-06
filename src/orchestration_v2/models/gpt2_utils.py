"""Small GPT-2 helpers (same behaviour as ~/orchestration's music_small_finetune versions)."""

from __future__ import annotations

from torch import nn


def gpt2_blocks(model: nn.Module):
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "h"):
        return model.h
    raise TypeError("expected a GPT-2 model with .h or .transformer.h blocks")


def enable_bidirectional_gpt2_attention(model: nn.Module) -> None:
    """Disable GPT-2's lower-triangular causal mask in every self-attention block (eager attention)."""
    config = getattr(model, "config", None)
    if config is not None:
        config._attn_implementation = "eager"
        if hasattr(config, "use_cache"):
            config.use_cache = False
    for block in gpt2_blocks(model):
        bias = getattr(block.attn, "bias", None)
        if bias is not None:
            bias.fill_(True)
