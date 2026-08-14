"""Pydantic schemas for the framework's planner / render-plan / segment objects.

This is the single source of truth for the shape of:
  - Skeleton          (output of shot_planner)
  - RenderPlan        (skeleton + per-segment + reference_inputs, what render consumes)
  - Shot / Segment / BoundaryAnchor / Portrait / Location / Zone / Prop / Transition

Why Pydantic and not stick with ``dict[str, Any]``:
  - one ``model_json_schema()`` call gives the planner agent its format
    instructions (langchain ``PydanticOutputParser`` pattern)
  - cross-field invariants (``len(shots) == len(boundary_anchors)``,
    ``∑ segment.duration == shot.duration``) become declarative
    ``@model_validator`` methods instead of three duplicate
    ``_validate_*`` functions in shot_planner / segment_planner /
  - IDE / mypy catch field typos at edit time rather than at
    plan_skeleton attempt 27

Schemas are **deliberately permissive** (``extra="allow"``) on fields the
planner may grow on its own — we only enforce the fields *we read*. New
planner output keys land in ``model_extra`` and pass through to
``model_dump()`` unchanged, so downstream consumers that need them keep
working without a schema bump.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ─── leaf-ish primitives ───────────────────────────────────────────────────


class Source(BaseModel):
    """``skeleton.source`` / ``render_plan.source`` — provenance + run config."""
    model_config = ConfigDict(extra="allow")

    input_path: str = ""
    n_shots: int = Field(ge=1)
    per_shot_duration_s: list[int]
    seed: int = 0


# Skeleton stage keeps the prompt at top level; render_plan stage moves
# it into a nested ``image_request.prompt``. We accept both shapes by
# making top-level ``prompt`` optional and exposing ``prompt_text()``
# helpers that pick whichever is present.


class _RefImageRequest(BaseModel):
    """Generic nested image_request block (portrait / prop / zone in
    render_plan). Mirrors ``_AnchorImageRequest`` for boundary anchors."""
    model_config = ConfigDict(extra="allow")
    request_id: str = ""
    prompt: str = ""
    negative_prompt: str = ""


class Portrait(BaseModel):
    """One character reference. ``portrait_plan`` is a dict keyed by ``id``."""
    model_config = ConfigDict(extra="allow")

    id: str
    name: str = ""
    reference_role: Literal["portrait", "supporting_portrait"] = "portrait"
    prompt: str = ""
    negative_prompt: str = ""
    image_request: _RefImageRequest | None = None

    def prompt_text(self) -> str:
        return self.prompt or (self.image_request.prompt if self.image_request else "")


class Prop(BaseModel):
    """A prop / object reference. Lives under ``location_plan.{loc}.props``
    *or* at top-level ``prop_plan`` after render_plan promotion."""
    model_config = ConfigDict(extra="allow")

    id: str = ""              # filled in via dict key when absent
    name: str = ""
    owner: str = ""           # portrait_id or location_id
    prompt: str = ""
    negative_prompt: str = ""
    image_request: _RefImageRequest | None = None

    def prompt_text(self) -> str:
        return self.prompt or (self.image_request.prompt if self.image_request else "")


class Location(BaseModel):
    """A scene/location reference. ``location_plan`` is a dict keyed by ``id``.

    NB: ``prompt`` is empty in render_plan.json — there the per-cell
    render prompt lives in a nested ``image_request.prompt`` field while
    the top-level keeps only structural info. Both shapes flow through.
    """
    model_config = ConfigDict(extra="allow")

    id: str = ""
    reference_name: str = ""
    props: dict[str, Prop] = Field(default_factory=dict)
    prompt: str = ""
    negative_prompt: str = ""


class ReferenceInputs(BaseModel):
    """How an anchor / segment chooses portrait + place + prop refs.
    Values may be comma-separated lists of asset ids (e.g.
    ``"yang_jian,sun_wukong"``)."""
    model_config = ConfigDict(extra="allow")

    portrait: str = ""
    place: str = ""
    prop: str = ""


class BoundaryAnchor(BaseModel):
    """One ``shot[i]`` start-state anchor frame. shots[i] ↔ anchors[i].

    Two shapes are valid:
      Skeleton:    ``{id, prompt, negative_prompt, reference_inputs}``
      RenderPlan:  ``{id, request_type, kind, image_request: {prompt, negative_prompt},
                      reference_inputs}``
    ``prompt_text`` helper unifies both — use it instead of reading
    ``.prompt`` directly when downstream code needs the actual prompt
    string regardless of stage.
    """
    model_config = ConfigDict(extra="allow")

    id: str = Field(pattern=r"^a\d{2}_\w+")
    reference_inputs: ReferenceInputs = Field(default_factory=ReferenceInputs)
    prompt: str = ""
    negative_prompt: str = ""
    image_request: _RefImageRequest | None = None

    def prompt_text(self) -> str:
        """Return the effective prompt regardless of stage shape."""
        if self.prompt:
            return self.prompt
        if self.image_request:
            return self.image_request.prompt
        return ""

    def negative_text(self) -> str:
        if self.negative_prompt:
            return self.negative_prompt
        if self.image_request:
            return self.image_request.negative_prompt
        return ""


class BoundaryGroup(BaseModel):
    """Wrapper for ``boundarys.boundary_anchors`` in the skeleton."""
    model_config = ConfigDict(extra="allow")

    boundary_anchors: list[BoundaryAnchor]


class Shot(BaseModel):
    """One narrative shot."""
    model_config = ConfigDict(extra="allow")

    id: str = Field(pattern=r"^shot\d{2}_\w+")
    duration_s: int = Field(ge=3, le=240)   # PARENT_SYSTEM_PROMPT hard rule 1
    story_goal: str
    start_state: str
    end_state: str
    visual_intent: str


class Transition(BaseModel):
    """Cut or bridge between two adjacent shots."""
    model_config = ConfigDict(extra="allow")

    id: str
    from_shot: str
    to_shot: str
    mode: Literal["cut", "bridge"]
    # bridge-only fields (validated only when mode=="bridge"):
    duration_s: int | None = None
    first_frame: str | None = None
    last_frame: str | None = None
    prompt: str | None = None
    negative_prompt: str | None = None

    @model_validator(mode="after")
    def bridge_fields_set(self) -> "Transition":
        if self.mode == "bridge":
            missing = [
                f for f in ("duration_s", "first_frame", "last_frame", "prompt")
                if getattr(self, f) is None
            ]
            if missing:
                raise ValueError(
                    f"bridge transition {self.id} missing required fields: {missing}"
                )
        return self


# ─── skeleton: what shot_planner emits ─────────────────────────────────────


class Skeleton(BaseModel):
    """Output of ``plan_skeleton``. The render_plan downstream is built from
    this + segment_planner output."""
    model_config = ConfigDict(extra="allow")

    source: Source
    portrait_plan: list[Portrait]
    location_plan: dict[str, Location]
    boundarys: BoundaryGroup
    shots: list[Shot]
    transitions: list[Transition]

    @model_validator(mode="after")
    def shots_anchors_align(self) -> "Skeleton":
        n_shots = len(self.shots)
        n_anc = len(self.boundarys.boundary_anchors)
        if n_shots != n_anc:
            raise ValueError(
                f"len(shots)={n_shots} != len(boundary_anchors)={n_anc} — "
                "shots[i] must correspond 1:1 to boundary_anchors[i]"
            )
        n_tr = len(self.transitions)
        if n_tr != max(0, n_shots - 1):
            raise ValueError(
                f"len(transitions)={n_tr} != len(shots)-1={n_shots - 1}"
            )
        return self


# ─── segment_planner output (per-shot dict[seg_id → Segment]) ──────────────


class SegmentRequest(BaseModel):
    """The raw render request a segment carries — what dispatch_segment sees."""
    model_config = ConfigDict(extra="allow")

    request_id: str
    prompt: str
    first_url: str = ""
    duration_s: int
    reference_image_urls: list[str] = Field(default_factory=list)
    seed: int = 0
    resolution: str = ""
    output_path: str = ""
    log_dir: str | None = None
    negative_prompt: str = ""


class Segment(BaseModel):
    """One render-able segment in a shot's segment list.

    Two shapes flow through here:
      segment_planner (per-shot dict):
        ``{id, duration_s, segment_index_in_shot, start_anchor,
          first_frame_path, end_state}``
      render_plan (post-enrichment):
        ``{id, request_type, shot_id, segment_request: {prompt, duration_s, ...},
          reference_inputs, start_anchor (nullable for non-first),
          first_frame_path, end_state, segment_index_in_shot}``

    ``duration_s_eff`` / ``start_anchor_or_empty`` helpers normalise the
    two views; don't access ``.duration_s`` directly in v2 callers.
    """
    model_config = ConfigDict(extra="allow")

    id: str = Field(pattern=r"^seg_shot\d{2}_\w+")
    duration_s: int | None = None       # may live under segment_request instead
    segment_index_in_shot: int = Field(ge=0)
    start_anchor: str | None = ""        # None on non-first segments in render_plan
    first_frame_path: Literal["shot_start_anchor", "previous_segment_last_frame"]
    end_state: str                       # single-frame snapshot description
    # render_plan-stage extras:
    request_type: str = ""
    shot_id: str = ""
    segment_request: SegmentRequest | None = None
    reference_inputs: ReferenceInputs | None = None

    def duration_s_eff(self) -> int:
        """Effective duration regardless of stage shape."""
        if self.duration_s is not None:
            return self.duration_s
        if self.segment_request is not None:
            return self.segment_request.duration_s
        raise ValueError(f"segment {self.id}: no duration_s found at top-level or in segment_request")

    def start_anchor_or_empty(self) -> str:
        return self.start_anchor or ""


# ─── render plan: skeleton + segments + per-cell render requests ──────────


class RenderPlan(BaseModel):
    """What the render pipeline consumes. Built by ``plan_segments_all`` +
    enrichment in pipeline.py.

    Schema is permissive — we only declare the fields the framework reads,
    everything else (e.g. ``boundary_policies``) flows through unchanged
    via ``extra="allow"``.
    """
    model_config = ConfigDict(extra="allow")

    source: Source
    portrait_plan: dict[str, Portrait]              # keyed by id after promotion
    location_plan: dict[str, Location]
    prop_plan: dict[str, Prop] = Field(default_factory=dict)
    boundary_anchors: list[BoundaryAnchor]          # promoted out of boundarys.boundary_anchors
    shots: list[Shot]
    transitions: list[Transition]
    segments: dict[str, Segment]                    # keyed by seg_id

    @model_validator(mode="after")
    def shot_segment_alignment(self) -> "RenderPlan":
        shots_by_id = {s.id: s for s in self.shots}
        seg_by_shot: dict[str, list[Segment]] = {}
        for seg in self.segments.values():
            seg_by_shot.setdefault(seg.shot_id, []).append(seg)
        for shot_id, segs in seg_by_shot.items():
            shot = shots_by_id.get(shot_id)
            if shot is None:
                raise ValueError(f"segment references unknown shot_id={shot_id}")
            total = sum(s.duration_s_eff() for s in segs)
            if total != shot.duration_s:
                raise ValueError(
                    f"shot {shot_id}: Σ segment duration ({total}s) != "
                    f"shot.duration_s ({shot.duration_s}s)"
                )
            ordered = sorted(segs, key=lambda s: s.segment_index_in_shot)
            for i, seg in enumerate(ordered):
                if seg.segment_index_in_shot != i:
                    raise ValueError(
                        f"shot {shot_id}: segment_index gap at {i} (got {seg.segment_index_in_shot})"
                    )
                if i == 0 and seg.first_frame_path != "shot_start_anchor":
                    raise ValueError(
                        f"shot {shot_id}: seg0.first_frame_path must be 'shot_start_anchor', "
                        f"got {seg.first_frame_path!r}"
                    )
                if i > 0 and seg.first_frame_path != "previous_segment_last_frame":
                    raise ValueError(
                        f"shot {shot_id}: seg{i}.first_frame_path must be "
                        f"'previous_segment_last_frame', got {seg.first_frame_path!r}"
                    )
        return self


# ─── per-shot segment dict (segment_planner ``plan_segments_for_shot`` output) ─


class ShotSegments(BaseModel):
    """``plan_segments_for_shot`` returns ``dict[seg_id → Segment]`` for one
    shot. We wrap this for validation; ``model_dump()`` gives back the
    plain dict the existing callers expect.
    """
    model_config = ConfigDict(extra="allow")

    segments: dict[str, Segment]

    @classmethod
    def validate_for_shot(
        cls, raw: dict[str, Any], shot: Shot, anchor: BoundaryAnchor,
    ) -> "ShotSegments":
        # Coerce raw dict (seg_id → seg_data) into the wrapper.
        wrapper = cls(segments=raw)
        # Cross-field rules vs the parent Shot + anchor:
        if not wrapper.segments:
            raise ValueError(f"segments for shot {shot.id} is empty")
        total = sum(s.duration_s_eff() for s in wrapper.segments.values())
        if total != shot.duration_s:
            raise ValueError(
                f"shot {shot.id}: Σ segment duration ({total}s) != "
                f"shot.duration_s ({shot.duration_s}s)"
            )
        ordered = sorted(wrapper.segments.values(),
                         key=lambda s: s.segment_index_in_shot)
        for i, seg in enumerate(ordered):
            if seg.segment_index_in_shot != i:
                raise ValueError(
                    f"shot {shot.id}: segment_index gap at {i} "
                    f"(got {seg.segment_index_in_shot})"
                )
            if not seg.end_state.strip():
                raise ValueError(
                    f"shot {shot.id}: seg{i}.end_state required "
                    "(single-frame snapshot, must align with prompt's closing pose)"
                )
            if i == 0:
                if (seg.start_anchor or "") != anchor.id:
                    raise ValueError(
                        f"shot {shot.id}: seg0.start_anchor={seg.start_anchor!r} != {anchor.id}"
                    )
                if seg.first_frame_path != "shot_start_anchor":
                    raise ValueError(
                        f"shot {shot.id}: seg0.first_frame_path must be 'shot_start_anchor'"
                    )
            else:
                if seg.first_frame_path != "previous_segment_last_frame":
                    raise ValueError(
                        f"shot {shot.id}: seg{i}.first_frame_path must be "
                        "'previous_segment_last_frame'"
                    )
        return wrapper


__all__ = [
    "Source",
    "Portrait",
    "Zone",
    "Prop",
    "Location",
    "ReferenceInputs",
    "BoundaryAnchor",
    "BoundaryGroup",
    "Shot",
    "Transition",
    "Skeleton",
    "SegmentRequest",
    "Segment",
    "RenderPlan",
    "ShotSegments",
]
