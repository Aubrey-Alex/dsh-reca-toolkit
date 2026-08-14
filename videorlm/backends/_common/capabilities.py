"""Common capability base type."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Capabilities:
    backend_name: str
    provider: str = ""
    model_family: str = ""
