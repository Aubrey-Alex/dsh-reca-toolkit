"""Generic OpenAI-compatible chat agent (openai.com or any compatible gateway).

Use ``OpenAICompatibleAgent`` directly when you know the endpoint isn't
DashScope; use the ``QwenAgent`` factory in ``..qwen`` when you want the
right concrete class chosen by ``config.base_url``.

``OpenAICompatibleConfig`` is an alias of ``QwenConfig`` — same dataclass
shape (``model`` / ``api_key`` / ``base_url`` / ``system_prompt`` / ...),
just exported under the symmetric name.
"""

from .agent import OpenAICompatibleAgent
from ..qwen.config import QwenConfig as OpenAICompatibleConfig

__all__ = ["OpenAICompatibleAgent", "OpenAICompatibleConfig"]
