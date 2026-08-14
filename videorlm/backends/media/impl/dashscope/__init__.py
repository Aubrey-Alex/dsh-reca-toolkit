"""DashScope-hosted media backends.

Submodules register their backends at import time:
  - ``image/``      : Wan / Qwen image families
  - ``video/``      : Wan video family + HappyHorse 1.0 family (segment + bridge)

All share DashScope auth + key pool + async task polling via
``_platform`` and the per-domain ``_primitives`` modules.
"""
from . import image as image  # noqa: F401
from . import video as video  # noqa: F401
