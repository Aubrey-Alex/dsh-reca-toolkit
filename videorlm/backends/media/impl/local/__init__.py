"""Local-inference video backends (no API).

Currently exposes the LTX-2.3 placeholder. Runtime is not wired yet — see
``ltx_v2_3/_runner.py`` for the marker that the model weights still need
to be downloaded.
"""
from . import ltx_v2_3 as ltx_v2_3  # noqa: F401
