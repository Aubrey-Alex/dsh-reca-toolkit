"""DashScope image backends.

One file per registered backend; each module registers itself on import.
``wanx-v1`` (legacy pure-T2I) is removed — pure T2I requirements are now
handled by wan2.7-image (which has native T2I support).
"""
from . import wan26 as wan26  # noqa: F401
from . import wan27 as wan27  # noqa: F401
from . import qwen as qwen  # noqa: F401
