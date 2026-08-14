"""Shared backend registry primitive."""

from __future__ import annotations


class BackendRegistry:
    """Generic name -> backend instance + kind -> backend chain registry."""

    def __init__(self) -> None:
        self._by_name: dict[str, object] = {}
        self._kind_default: dict[str, tuple[str, ...]] = {}

    def register(self, backend: object, *, name: str) -> None:
        if not name:
            raise ValueError("backend name is empty")
        self._by_name[name] = backend

    def get(self, name: str) -> object | None:
        return self._by_name.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def __getitem__(self, name: str) -> object:
        return self._by_name[name]

    def keys(self):
        return self._by_name.keys()

    def clear(self) -> None:
        self._by_name.clear()

    def update(self, items: dict[str, object]) -> None:
        self._by_name.update(items)

    def list_names(self) -> list[str]:
        return sorted(self._by_name)

    def set_kind_default(self, kind: str, chain: tuple[str, ...]) -> None:
        if not kind:
            raise ValueError("kind is empty")
        if not chain:
            raise ValueError(f"default chain for kind={kind!r} is empty")
        self._kind_default[kind] = tuple(chain)

    def for_kind(self, kind: str) -> tuple[str, ...]:
        return self._kind_default.get(kind, ())
