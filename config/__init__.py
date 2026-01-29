"""SunFish configuration module."""

from .qwen_masked_config import (
    QwenMaskedDiffusionConfig,
    get_qwen_masked_config,
    get_qwen_masked_config_cpu,
)

__all__ = [
    "QwenMaskedDiffusionConfig",
    "get_qwen_masked_config",
    "get_qwen_masked_config_cpu",
]
