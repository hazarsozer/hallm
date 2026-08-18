"""Model package: config, GPT modules, and the weight-sharing mechanisms."""

from hallm.model.config import ARMS, SHAPES, ModelConfig, arm_config
from hallm.model.gpt import GPT, Block

__all__ = ["ModelConfig", "arm_config", "ARMS", "SHAPES", "GPT", "Block"]
