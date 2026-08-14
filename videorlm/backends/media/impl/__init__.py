"""Concrete VideoBackend implementations.

Each ``impl/<vendor>/`` package registers its backends at import time. The
top-level lazy autoload is triggered by ``interface.registry._autoload_default_backends``.
"""
