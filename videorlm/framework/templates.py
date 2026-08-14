"""``ChatPromptTemplate`` — file-driven prompts with variable injection.

Replaces the giant ``PARENT_SYSTEM_PROMPT = '''...'''`` strings in
``framework/{shot_planner,segment_planner}/prompts.py``
(11+ k chars each, mixed with f-string interpolation in the call site).

Pattern lifted from langchain's ``ChatPromptTemplate`` / ``PromptTemplate``:
  - prompts live in ``framework/prompts/<stage>/{system.md, user.j2, ...}``
    as text/Markdown/Jinja2 files;
  - ``ChatPromptTemplate.from_files(...)`` reads them at construction;
  - ``.format_messages(**vars)`` substitutes Jinja2 variables and returns
    a list of ``{role, content}`` dicts ready for the agent;
  - A stage can swap templates at runtime via ``.with_prompt(path)``,
    enabling A/B prompt tests without forking code.

We don't take a jinja2 dependency for the basic case — variable
substitution uses Python's ``str.Template`` with ``$var`` syntax (the
default for ``user_template_kind="dollar"``). When richer logic is
needed (loops, conditionals), pass ``user_template_kind="jinja2"`` and
the import happens lazily.
"""
from __future__ import annotations

import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


# ─── one chat message ──────────────────────────────────────────────────────


@dataclass
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


# ─── the template ──────────────────────────────────────────────────────────


@dataclass
class ChatPromptTemplate:
    """A multi-message template. Holds the *raw* template text per role
    plus the substitution style; ``format_messages(**vars)`` does the
    interpolation."""

    system_template: str = ""
    user_template: str = ""
    extra: list[tuple[Literal["system", "user", "assistant"], str]] = field(default_factory=list)
    style: Literal["dollar", "jinja2", "format"] = "dollar"
    # source paths (informational; used by callbacks / debug)
    sources: dict[str, str] = field(default_factory=dict)

    # ── builders ────────────────────────────────────────────────────────

    @classmethod
    def from_files(
        cls,
        system: str | Path | None = None,
        user: str | Path | None = None,
        *,
        extra: list[tuple[str, str | Path]] | None = None,
        style: Literal["dollar", "jinja2", "format"] = "dollar",
    ) -> "ChatPromptTemplate":
        """Construct from filesystem paths. Missing paths → empty string
        (you can still pass the corresponding role inline later)."""
        sources: dict[str, str] = {}
        sys_txt = _read_or_empty(system, sources, "system")
        usr_txt = _read_or_empty(user, sources, "user")
        ex: list[tuple[Any, str]] = []
        for role, path in extra or []:
            ex.append((role, _read_or_empty(path, sources, f"extra/{role}")))
        return cls(
            system_template=sys_txt,
            user_template=usr_txt,
            extra=ex,
            style=style,
            sources=sources,
        )

    @classmethod
    def from_messages(
        cls,
        messages: list[tuple[Literal["system", "user", "assistant"], str]],
        *,
        style: Literal["dollar", "jinja2", "format"] = "dollar",
    ) -> "ChatPromptTemplate":
        """Construct from inline (role, template_string) tuples — useful
        for tests / quick experiments without spinning up files."""
        sys_t = ""
        usr_t = ""
        extras: list[tuple[Any, str]] = []
        for role, txt in messages:
            if role == "system" and not sys_t:
                sys_t = txt
            elif role == "user" and not usr_t:
                usr_t = txt
            else:
                extras.append((role, txt))
        return cls(system_template=sys_t, user_template=usr_t, extra=extras, style=style)

    # ── runtime substitution ────────────────────────────────────────────

    def format_messages(self, **variables: Any) -> list[ChatMessage]:
        """Substitute ``variables`` into each template and return the
        ordered message list. System comes first (if non-empty), then
        user, then any ``extra`` messages in registration order."""
        out: list[ChatMessage] = []
        if self.system_template:
            out.append(ChatMessage(role="system", content=self._fill(self.system_template, variables)))
        if self.user_template:
            out.append(ChatMessage(role="user", content=self._fill(self.user_template, variables)))
        for role, txt in self.extra:
            out.append(ChatMessage(role=role, content=self._fill(txt, variables)))  # type: ignore[arg-type]
        return out

    def format_system(self, **variables: Any) -> str:
        """System message body, post-substitution. Handy for backends
        that only accept a single system string + user message text
        (the videorlm agent's existing API)."""
        return self._fill(self.system_template, variables)

    def format_user(self, **variables: Any) -> str:
        return self._fill(self.user_template, variables)

    # ── swap-on-the-fly knob (for A/B prompt experiments) ───────────────

    def with_user(self, new_user_template: str | Path) -> "ChatPromptTemplate":
        txt = _read_text(new_user_template) if isinstance(new_user_template, (str, Path)) and Path(str(new_user_template)).exists() else str(new_user_template)
        return ChatPromptTemplate(
            system_template=self.system_template,
            user_template=txt,
            extra=list(self.extra),
            style=self.style,
            sources={**self.sources, "user": str(new_user_template)},
        )

    def with_system(self, new_system: str | Path) -> "ChatPromptTemplate":
        txt = _read_text(new_system) if isinstance(new_system, (str, Path)) and Path(str(new_system)).exists() else str(new_system)
        return ChatPromptTemplate(
            system_template=txt,
            user_template=self.user_template,
            extra=list(self.extra),
            style=self.style,
            sources={**self.sources, "system": str(new_system)},
        )

    # ── internal: substitution dispatch ─────────────────────────────────

    def _fill(self, template: str, variables: dict[str, Any]) -> str:
        if not template:
            return ""
        if self.style == "dollar":
            # $var / ${var} — safe substitute leaves unknown placeholders intact
            return string.Template(template).safe_substitute(**variables)
        if self.style == "format":
            try:
                return template.format(**variables)
            except KeyError as exc:
                raise KeyError(
                    f"ChatPromptTemplate.format substitution failed: missing var {exc.args[0]!r}; "
                    "you can keep unsubstituted placeholders by switching to style='dollar'"
                ) from exc
        if self.style == "jinja2":
            # Lazy import — keeps jinja2 optional
            try:
                from jinja2 import Environment, StrictUndefined
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "ChatPromptTemplate(style='jinja2') requires jinja2; pip install jinja2"
                ) from exc
            env = Environment(undefined=StrictUndefined)
            return env.from_string(template).render(**variables)
        raise ValueError(f"unknown style: {self.style!r}")


# ─── helpers ───────────────────────────────────────────────────────────────


def _read_or_empty(path: str | Path | None, sources: dict[str, str], key: str) -> str:
    if path is None:
        return ""
    p = Path(path)
    sources[key] = str(p)
    return p.read_text(encoding="utf-8")


def _read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


__all__ = ["ChatPromptTemplate", "ChatMessage"]
