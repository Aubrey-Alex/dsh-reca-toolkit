"""Legacy ABC shim.

The old ``VideoBackend`` ABC has been replaced by:

  - ``VideoSegmentBackend``  (segment + bridge)

Image backends are plain classes with a ``render(ImageRequest) -> ImageResult``
method; no Protocol forced on them.

This module is kept only to host shared registry / capability helpers if
needed in the future. The old ABC is deleted.
"""
from __future__ import annotations


__all__: list[str] = []
