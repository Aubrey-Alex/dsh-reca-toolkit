"""DashScope video backends — wan family + happyhorse family.

Each backend file registers itself at import time via ``register_backend``.

Segment backends (i2v / r2v): expose ``render_segment`` + ``render_bridge``.

All t2v / multi-shot backends are deleted (no longer in the pipeline path).
"""
from . import wan_i2v as wan_i2v  # noqa: F401
from . import wan_r2v as wan_r2v  # noqa: F401
from . import wan30 as wan30  # noqa: F401
from . import happyhorse_i2v as happyhorse_i2v  # noqa: F401
from . import happyhorse_r2v as happyhorse_r2v  # noqa: F401
