"""OpenAI-compatible media backends.

Routes through the ``openai`` provider (``backends._common.providers.openai``).
Endpoint is env-driven (``OPENAI_BASE_URL``) so
the same code works against openai.com, a self-hosted gateway, or any other
OpenAI-compatible endpoint.

Currently exposes ``gpt-image-2`` for T2I + I2I (image edit). Returns
``b64_json`` inline; the backend decodes to ``output_path`` locally and
publishes to OSS when OSS env is configured, returning the public URL as
``RenderResult.output_url``.
"""
from . import image as image  # noqa: F401
