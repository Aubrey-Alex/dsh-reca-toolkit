"""End-to-end pipeline: raw story prompt -> planner output -> render plan -> mp4.

Each step is a top-level function. Compose freely or call `run_planner()` /
`run_render()` for the canonical orchestration.

Layers
------
1. Planning (LLM agents) — `plan_skeleton` (in `shot_planner.plan`),
   `plan_segments_for_shot`, `plan_segments_all`, `merge_into_planner_output`.
2. Conversion — `to_render_plan` produces a JSON-friendly dict matching
   the for_render.json schema (every entity wrapped in dispatch-Request
   dict form, with placeholder URLs to be filled at render time).
3. Render (media backends) — `_render_image_dag` (single DAG for
   portrait / location / prop / anchor), `render_segments` (per-shot serial
   chain + inline bridge dispatch), `concat_final`.

Failure model:
  - any render failure raises; no placeholder substitution
  - .url cache survives for --resume
  - no cross-backend fallback; single-model retry inside dispatch_*
"""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from videorlm.backends.media import (
    BridgeRequest,
    ImageRef,
    ImageRequest,
    SegmentRequest,
    VideoResult,
    dispatch_bridge,
    dispatch_image,
    dispatch_segment,
    for_kind,
)

from .shot_planner import PARENT_SYSTEM_PROMPT, plan_skeleton
from .segment_planner import (
    plan_segments_all,
    plan_segments_for_shot,
    reconstruct_segment_planner_state,
)

if TYPE_CHECKING:
    from videorlm.backends.llm.agents import Agent


# ── Validator params ───────────────────────────────────────────────────


@dataclass
class ValidatorParams:
    """Anchor validation + repair pass between i2i render and R2V render.

    See videorlm/framework/validator/anchor/ for the validator agent + repair
    branches. Passed to `run_render(..., validator=ValidatorParams(...))`.
    """
    planner: dict[str, Any]
    story: str
    validator_cfg: Any
    max_repair_attempts: int = 2
    repair_dir: Path | None = None


@dataclass
class SegmentValidatorParams:
    """Segment validation + repair pass after each segment chain renders.

    Enables `run_segment_validator` on every segment's **full mp4** (NOT
    just a tail PNG): qwen3-vl-plus samples its own frames over the video
    timeline so the action_consistency axis can judge motif / identity /
    physical contact continuity.

    On fail, the pipeline runs a repair strategy. Two strategies — BOTH
    use gpt-5.5 with SP memory inheritance, the difference is scope:

      micro_adjust — light. ``micro_adjust_single_segment`` forks an
                     SP agent with SP's full post-turn state (story +
                     skeleton + sp_user + segments_for_shot), swaps
                     system to ``SEGMENT_MICRO_ADJUST_SYSTEM_PROMPT``,
                     and rewrites JUST THIS ONE segment's prompt
                     (+ optional end_state) with minimal change. Preserves
                     SP's original cinematic style, no re-decomposition.
      replan      — heavy. ``replan_segments_for_shot`` rewrites the
                     WHOLE shot's segments (may re-decompose, change
                     segment count). Used for structural failures
                     (identity drift across multiple segs, story_goal
                     mismatch, decomposition itself is wrong).

    When ``router_cfg`` is set, a text-only router LLM (qwen3.6-plus)
    decides per-fail which strategy to use. When router_cfg is None, the
    pipeline defaults to ``seed_reroll`` every time (cheapest path).

    Loop up to ``max_repair_attempts`` times; if still failing on the last
    attempt, accept the render and log a warning so concat can complete.

    When set to None on run_render, segment validator is skipped (default).
    """
    planner: dict[str, Any]
    story: str
    # validator_cfg is built per-request inside run_segment_validator
    # (qwen3-vl-plus video input — see validator/segment/validate.py).
    enabled: bool = True
    max_repair_attempts: int = 2
    # Repair-strategy router config (qwen3.6-plus text-only). When None,
    # repair always uses ``seed_reroll`` (no router consult).
    router_cfg: Any | None = None
    # Segment micro-adjust config. When None and router returns micro_adjust,
    # raises RuntimeError — caller must supply this config for micro_adjust.
    sp_micro_adjust_cfg: Any | None = None
    # Segment-planner replan config. When None and router returns replan,
    # raises RuntimeError — caller must supply this config for replan.
    sp_replan_cfg: Any | None = None

    # ── Pluggable knobs ────────────────────────────────────────────────
    # Every field below defaults to None / the historical constant, so an
    # untouched SegmentValidatorParams behaves exactly as before.
    #
    # Pass thresholds. None ⇒ use validator module defaults
    # (0.6 / 0.7, themselves overridable via RECA_SEGMENT_AXIS_THRESHOLD /
    # RECA_SEGMENT_OVERALL_THRESHOLD).
    axis_pass_threshold: float | None = None
    overall_pass_threshold: float | None = None
    # Vision model for the judgment call. None ⇒ module default
    # (qwen3-vl-plus, or RECA_SEGMENT_VALIDATOR_MODEL).
    validator_model: str | None = None
    # Server-side frame sampling rate for the video input. None ⇒ 10.
    video_sample_fps: int | None = None
    # Which repair strategies the router is allowed to pick. A strategy
    # outside this set is downgraded to ``allowed_strategies[0]`` with a
    # loud log line. Narrow it to e.g. ``("seed_reroll",)`` to keep repairs
    # cheap, or to ``()`` to validate-only (score + log, never re-render).
    allowed_strategies: tuple[str, ...] = ("seed_reroll", "micro_adjust", "replan")
    # Keep per-attempt snapshots and restore the highest-scoring one at the
    # end. False ⇒ always keep the last render (no snapshots, no disk cost).
    best_of_n: bool = True
    # What to do when the validator CALL itself fails (API 4xx/5xx, malformed
    # JSON, ...) — as opposed to the segment being judged a fail.
    #   "raise"  — propagate; the whole shot chain fails (historical behaviour)
    #   "accept" — log it, keep the current render, move on
    on_error: str = "raise"


# ── helpers ────────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


def parse_json_block(text: str) -> dict[str, Any]:
    m = _FENCE_RE.search(text)
    payload = m.group(1).strip() if m else text.strip()
    return json.loads(payload)


def _log_agent_state(label: str, agent: Any) -> None:
    msgs = agent.state.get("messages", []) or []
    summary = " | ".join(f"{m.role}({len(m.content)} chars)" for m in msgs)
    print(f"[agent-trace] {label}: messages={len(msgs)} [{summary}]", flush=True)


def _sidecar_path(output_path: str | None) -> Path | None:
    if not output_path:
        return None
    return Path(output_path + ".url")


def _load_cached_url(output_path: str | None) -> str | None:
    if not output_path:
        return None
    p = Path(output_path)
    side = _sidecar_path(output_path)
    if p.exists() and side is not None and side.exists():
        url = side.read_text(encoding="utf-8").strip()
        if url:
            return url
    return None


def _save_cached_url(output_path: str | None, url: str) -> None:
    side = _sidecar_path(output_path)
    if side is None or not url:
        return
    side.parent.mkdir(parents=True, exist_ok=True)
    side.write_text(url, encoding="utf-8")


def _republish_local_to_oss(
    local_path: str,
    *,
    request_id: str,
    kind: str,
    log_dir: str | None = None,
) -> str | None:
    """Re-upload an existing local image file to OSS without re-rendering."""
    from videorlm.backends._common.oss_publisher import upload_file as _oss_upload_file

    if not Path(local_path).exists():
        return None
    return _oss_upload_file(
        local_path,
        prefix=f"republish/{kind}",
        request_id=request_id,
        log_dir=log_dir,
    )


# ── 1. Plan skeleton — moved to shot_planner.plan ──────────────────────
# `plan_skeleton` is re-exported above for backward compat at the package
# level (videorlm.framework.plan_skeleton).


# ── 2 & 3. Segment planning — in segment_planner package ───────────────


# ── 4. Merge skeleton + segments → complete planner output ─────────────

def merge_into_planner_output(skeleton: dict[str, Any], segments: dict[str, Any]) -> dict[str, Any]:
    return {**skeleton, "segments": segments}


# ── 5. Planner output → render plan (for_render.json shape) ────────────

def to_render_plan(
    planner: dict[str, Any],
    run_dir: str | Path,
    *,
    seed: int = 0,
    resolution: str = "1280x720",
    video_resolution: str = "1920x1080",
) -> dict[str, Any]:
    """Build the JSON-friendly render plan from `planner` output.

    Schema (post-F7, segments carry `segment_request`, bridges `bridge_request`):
      - portraits / locations / props / boundary_anchors carry an
        ``image_request`` dict (use ``resolution``, default 1280x720 since
        gpt-image-2 origin only natively supports 1024² / 1024x1536 /
        1536x1024 and the gateway maps 1280x720 through cleanly)
      - segments carry a ``segment_request`` dict (use ``video_resolution``,
        default 1920x1080 so happyhorse-r2v / wan2.7-r2v outputs match
        wan2.7-i2v bridges)
      - bridges (in ``boundary_policies``) carry a ``bridge_request`` dict
        (also ``video_resolution``)
      - the legacy ``kind`` field is dropped (dispatch.* routes by env +
        request type/mode internally)
    """
    run_dir = Path(run_dir)
    log_dir = str(run_dir / "logs")

    portrait_plan = {
        p["id"]: {
            "id": p["id"],
            "request_type": "ImageRequest",
            "kind": "portrait",
            "image_request": _portrait_req(p, run_dir, log_dir, seed, resolution),
        }
        for p in planner["portrait_plan"]
    }

    location_plan: dict[str, Any] = {}
    prop_plan: dict[str, Any] = {}
    for loc_id, loc in planner["location_plan"].items():
        location_plan[loc_id] = {
            "id": loc_id,
            "request_type": "ImageRequest",
            "kind": "location",
            "image_request": _location_req(loc_id, loc, run_dir, log_dir, seed, resolution),
        }
        for prop_id, prop in (loc.get("props") or {}).items():
            prop_plan[prop_id] = {
                "id": prop_id,
                "request_type": "ImageRequest",
                "kind": "prop",
                "owner": prop.get("owner"),
                "image_request": _prop_req(prop_id, prop, run_dir, log_dir, seed, resolution),
            }

    boundary_anchors = [
        {
            "id": a["id"],
            "request_type": "ImageRequest",
            "kind": "anchor_image",
            "image_request": _anchor_req(a, run_dir, log_dir, seed, resolution),
            "reference_inputs": a.get("reference_inputs", {}),
        }
        for a in planner["boundarys"]["boundary_anchors"]
    ]

    segments = {}
    for seg_id, seg in planner["segments"].items():
        ri = seg.get("reference_inputs", {}) or {}
        segments[seg_id] = {
            "id": seg_id,
            "request_type": "SegmentRequest",
            "segment_request": _segment_req(seg, run_dir, log_dir, seed, video_resolution),
            "reference_inputs": ri,
            "shot_id": seg["shot_id"],
            "segment_index_in_shot": seg["segment_index_in_shot"],
            "start_anchor": seg.get("start_anchor"),
            "first_frame_path": seg["first_frame_path"],
            "end_state": seg.get("end_state", ""),
        }

    boundary_policies = [
        {
            "id": tr["id"],
            "request_type": "BridgeRequest",
            "bridge_request": _bridge_req(tr, run_dir, log_dir, seed, video_resolution),
            "from_shot": tr["from_shot"],
            "to_shot": tr["to_shot"],
        }
        for tr in planner["transitions"]
        if tr["mode"] == "bridge"
    ]

    return {
        "source": planner.get("source", {}),
        "portrait_plan": portrait_plan,
        "location_plan": location_plan,
        "prop_plan": prop_plan,
        "boundary_anchors": boundary_anchors,
        "shots": planner["shots"],
        "transitions": planner["transitions"],
        "segments": segments,
        "boundary_policies": boundary_policies,
    }


def _count_reference_inputs(ri: dict[str, Any]) -> int:
    if not ri:
        return 0
    n = 0
    for pid in _split_ids(ri.get("portrait")):
        if pid:
            n += 1
    if ri.get("place"):
        n += 1
    for prop_id in _split_ids(ri.get("prop")):
        if prop_id:
            n += 1
    return n


def _portrait_req(p, run_dir, log_dir, seed, resolution):
    return {
        "request_id": p["id"], "prompt": p["prompt"],
        "references": [], "seed": seed, "resolution": resolution,
        "output_path": str(run_dir / "portraits" / f"{p['id']}.png"), "log_dir": log_dir,
        "negative_prompt": p.get("negative_prompt", ""),
        "name": p.get("name", p["id"]),
    }


def _location_req(loc_id, loc, run_dir, log_dir, seed, resolution):
    return {
        "request_id": loc_id, "prompt": loc["prompt"],
        "references": [], "seed": seed, "resolution": resolution,
        "output_path": str(run_dir / "locations" / f"{loc_id}.png"), "log_dir": log_dir,
        "negative_prompt": loc.get("negative_prompt", ""),
    }


def _prop_req(prop_id, prop, run_dir, log_dir, seed, resolution):
    return {
        "request_id": prop_id, "prompt": prop["prompt"],
        "references": [], "seed": seed, "resolution": resolution,
        "output_path": str(run_dir / "props" / f"{prop_id}.png"), "log_dir": log_dir,
        "negative_prompt": prop.get("negative_prompt", ""),
    }


def _anchor_ref_entries(reference_inputs: dict[str, Any]) -> list[dict[str, str]]:
    """Build the reference entries (role + asset_id) for an anchor from its
    ``reference_inputs`` dict (the same form used in planner.boundary_anchors).
    URLs are blank — caller resolves via asset_pool at dispatch time.
    Shared by initial render + both repair paths so all three render
    anchors with the same ref set.
    """
    refs: list[dict[str, str]] = []
    ri = reference_inputs or {}
    for pid in _split_ids(ri.get("portrait")):
        refs.append({"role": "portrait", "url": "", "asset_id": pid})
    if ri.get("place"):
        refs.append({"role": "scene", "url": "", "asset_id": ri["place"]})
    for prop_id in _split_ids(ri.get("prop")):
        refs.append({"role": "reference", "url": "", "asset_id": prop_id})
    return refs


def _resolve_refs_from_inputs(
    reference_inputs: dict[str, Any],
    asset_pool: dict[str, str],
) -> list[ImageRef]:
    """Resolve a ``reference_inputs`` dict (portrait / place / prop) into
    ImageRef list. ``place`` is a top-level ``loc_id`` (zones were retired
    2026-05-17 — the LLM now writes loc_ids directly, no zone indirection)."""
    resolved: list[ImageRef] = []
    for r in _anchor_ref_entries(reference_inputs):
        url = asset_pool.get(r["asset_id"])
        if url:
            resolved.append(ImageRef(role=r["role"], url=url))
    return resolved


def _anchor_req(a, run_dir, log_dir, seed, resolution):
    return {
        "request_id": a["id"], "prompt": a["prompt"],
        "references": _anchor_ref_entries(a.get("reference_inputs", {}) or {}),
        "seed": seed, "resolution": resolution,
        "output_path": str(run_dir / "anchors" / f"{a['id']}.png"), "log_dir": log_dir,
        "negative_prompt": a.get("negative_prompt", ""),
    }


def _segment_req(seg, run_dir, log_dir, seed, resolution):
    return {
        "request_id": seg["id"], "prompt": seg["prompt"],
        "first_url": "", "duration_s": int(seg["duration_s"]),
        "reference_image_urls": [], "seed": seed, "resolution": resolution,
        "output_path": str(run_dir / "segments" / f"{seg['id']}.mp4"), "log_dir": log_dir,
        "negative_prompt": seg.get("negative_prompt", ""),
    }


def _bridge_req(tr, run_dir, log_dir, seed, resolution):
    return {
        "request_id": tr["id"], "prompt": tr.get("prompt", ""),
        "first_url": "", "last_url": "",
        "duration_s": int(tr.get("duration_s", 3)),
        "seed": seed, "resolution": resolution,
        "output_path": str(run_dir / "bridges" / f"{tr['id']}.mp4"), "log_dir": log_dir,
        "negative_prompt": tr.get("negative_prompt", ""),
    }


def _split_ids(field: Any) -> list[str]:
    if not field:
        return []
    if isinstance(field, list):
        return [str(x).strip() for x in field if x]
    return [s.strip() for s in str(field).split(",") if s.strip()]


# ── 6. Image DAG (F11 — replaces 4 stage functions) ────────────────────


def _dispatch_one_image(rr: dict[str, Any], kind: str) -> str:
    """Render or republish a single image; returns the OSS URL.

    Handles cache hits (.url sidecar with http URL) and local-path
    rescue (republish OSS) for resumed runs where a prior OSS publish
    failed. Raises if the result is a non-http URL — that means OSS
    publish failed and we'd poison the cache for downstream segments
    that need an https URL.
    """
    cached = _load_cached_url(rr.get("output_path"))
    if cached:
        if cached.startswith("http"):
            print(f"[render-skip] {rr['request_id']} cached -> {cached[:80]}", flush=True)
            return cached
        # Local-only runs intentionally have no OSS credentials. A readable
        # local cache is still a valid dependency for image edits and Wan
        # requests; do not make resume depend on an unrelated publisher.
        if Path(cached).is_file():
            print(f"[render-skip-local] {rr['request_id']} cached -> {cached}", flush=True)
            return cached
        new_url = _republish_local_to_oss(
            cached,
            request_id=rr["request_id"],
            kind=kind,
            log_dir=rr.get("log_dir"),
        )
        if new_url and new_url.startswith("http"):
            print(
                f"[republish] {rr['request_id']} local cache -> OSS: {new_url[:80]}",
                flush=True,
            )
            _save_cached_url(rr.get("output_path"), new_url)
            return new_url
        raise RuntimeError(
            f"_dispatch_one_image[{rr['request_id']}]: cached URL is a local path "
            f"({cached!r}) and re-upload to OSS failed"
        )
    # Some OpenAI-compatible gateways expose /images/generations but do not
    # implement /images/edits.  In that configuration, preserve continuity by
    # carrying the resolved source anchor forward instead of issuing a request
    # that can hang until the provider timeout.  The real image-edit path is
    # still used by default and can be selected per provider with
    # RECA_GPT_IMAGE_2_EDIT_MODE=api.
    if (
        kind == "image_edit"
        and os.environ.get("RECA_GPT_IMAGE_2_EDIT_MODE", "api").lower()
        in {"copy", "copy_source", "source"}
    ):
        source = next(
            (str(ref.get("url")) for ref in rr.get("references", [])
             if ref.get("role") == "source" and ref.get("url")),
            "",
        )
        if source and Path(source).is_file() and rr.get("output_path"):
            output_path = Path(str(rr["output_path"]))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, output_path)
            _save_cached_url(rr.get("output_path"), str(output_path))
            print(
                f"[render-copy-source] {rr['request_id']} <- {source}",
                flush=True,
            )
            return str(output_path)
        raise RuntimeError(
            f"_dispatch_one_image[{rr['request_id']}]: copy-source mode "
            f"requires a local source anchor, got {source!r}"
        )
    req = ImageRequest(
        request_id=rr["request_id"], kind=kind, prompt=rr["prompt"],
        references=tuple(
            ImageRef(role=r["role"], url=r["url"]) for r in rr["references"] if r.get("url")
        ),
        seed=int(rr["seed"]), resolution=rr.get("resolution", "1280x720"),
        output_path=rr.get("output_path"), log_dir=rr.get("log_dir"),
        negative_prompt=rr.get("negative_prompt", ""),
    )
    result = dispatch_image(req)
    url = getattr(result, "output_url", None) or getattr(result, "output_path", None) or ""
    if not url.startswith("http") and not Path(url).is_file():
        raise RuntimeError(
            f"_dispatch_one_image[{rr['request_id']}]: refusing to cache non-URL "
            f"({url!r}); OSS publish likely failed"
        )
    if not url.startswith("http"):
        print(f"[render-local] {rr['request_id']} -> {url}", flush=True)
    _save_cached_url(rr.get("output_path"), url)
    return url


# The source frame is the canonical continuity constraint. Keep it ahead of
# auxiliary portraits/props when a backend imposes a reference-image cap.
_REF_ROLE_PRIORITY = {"source": -1, "portrait": 0, "scene": 1, "reference": 2, "start": 3, "end": 4}


def _truncate_refs(refs: list[dict[str, Any]], max_refs: int, *, label: str = "") -> list[dict[str, Any]]:
    if max_refs <= 0 or len(refs) <= max_refs:
        return [{k: v for k, v in r.items() if k != "asset_id"} for r in refs]
    ordered = sorted(enumerate(refs), key=lambda kv: (_REF_ROLE_PRIORITY.get(kv[1]["role"], 99), kv[0]))
    keep = sorted(ordered[:max_refs], key=lambda kv: kv[0])
    dropped = [refs[i].get("asset_id", "?") for i in range(len(refs)) if i not in {idx for idx, _ in keep}]
    print(
        f"[ref-truncate] {label or 'anchor'}: backend cap={max_refs}, "
        f"requested={len(refs)}; dropping {len(dropped)} refs: {dropped}",
        flush=True,
    )
    return [{k: v for k, v in r.items() if k != "asset_id"} for _, r in keep]


def _build_image_dag(render_plan: dict[str, Any]) -> tuple[dict[str, set[str]], dict[str, tuple[dict, str]]]:
    """Build (deps, nodes) maps for the image-render DAG.

    Returns:
      deps:  node_id → set(blocking node_ids)
      nodes: node_id → (request_dict, kind_label)
    """
    deps: dict[str, set[str]] = {}
    nodes: dict[str, tuple[dict[str, Any], str]] = {}

    for p in render_plan["portrait_plan"].values():
        deps[p["id"]] = set()
        nodes[p["id"]] = (p["image_request"], "portrait")
    for loc in render_plan["location_plan"].values():
        deps[loc["id"]] = set()
        nodes[loc["id"]] = (loc["image_request"], "anchor_image")
    for prop in render_plan["prop_plan"].values():
        owner = prop.get("owner")
        deps[prop["id"]] = {owner} if owner and owner in deps else set()
        nodes[prop["id"]] = (prop["image_request"], "anchor_image")
    for a in render_plan["boundary_anchors"]:
        a_id = a["id"]
        ref_deps: set[str] = set()
        for r in a["image_request"]["references"]:
            asset_id = r.get("asset_id")
            if asset_id and asset_id in deps:
                ref_deps.add(asset_id)
        source_anchor = a["image_request"].get("source_anchor")
        if source_anchor and source_anchor in deps:
            ref_deps.add(source_anchor)
        deps[a_id] = ref_deps
        nodes[a_id] = (
            a["image_request"],
            str(a["image_request"].get("render_kind") or "anchor_image"),
        )

    return deps, nodes


def _resolve_anchor_refs(render_plan: dict[str, Any], anchor_entry: dict[str, Any], asset_pool: dict[str, str]) -> dict[str, Any]:
    """Resolve an anchor's reference URLs from the asset pool, applying
    backend max-ref truncation. ``reference_inputs.place`` is a top-level
    ``loc_id`` (zones were retired 2026-05-17)."""
    render_kind = str(anchor_entry["image_request"].get("render_kind") or "anchor_image")
    max_refs = for_kind(render_kind).capabilities().max_reference_images
    rr = dict(anchor_entry["image_request"])
    resolved: list[dict[str, Any]] = []
    for r in rr["references"]:
        url = r.get("url") or asset_pool.get(r.get("asset_id", ""))
        # Gateway inputs are commonly staged as local files. Image-edit
        # backends need HTTP(S) references, so publish local inputs here.
        if url and not str(url).startswith(("http://", "https://")) and Path(str(url)).is_file():
            url = _republish_local_to_oss(
                str(url), request_id=rr.get("request_id", "anchor"),
                kind=render_kind, log_dir=rr.get("log_dir"),
            ) or url
        if url:
            resolved.append({"role": r.get("role", "reference"), "url": url, "asset_id": r.get("asset_id", "")})
    rr["references"] = _truncate_refs(resolved, max_refs, label=rr.get("request_id", "anchor"))
    return rr


def _render_image_dag(render_plan: dict[str, Any], *, max_workers: int = 16) -> dict[str, str]:
    """Render portraits + locations + props + anchors in a single DAG.

    Replaces the legacy 4-stage serial pipeline. Topological dispatch:
    a node is submitted as soon as all its dependencies are in the
    ``done`` map. Anchors depend on their referenced portrait/place/prop
    assets; props depend on their owner portrait; portraits & locations
    have no dependencies.

    Args:
      max_workers: max concurrent image renders. 16 is sane for
        gpt-image-2-pro on the OpenAI-compatible gateway.

    Returns:
      ``{node_id: oss_url}`` for every node that succeeded. Raises
      RuntimeError listing any node that failed after its node-level
      retries (dispatch_image's retry budget already exhausted).
    """
    deps, nodes = _build_image_dag(render_plan)
    if not nodes:
        return {}

    preloaded = {
        str(node_id): str(source)
        for node_id, source in (render_plan.get("preloaded_assets") or {}).items()
        if source
    }
    done: dict[str, str] = {node_id: source for node_id, source in preloaded.items() if node_id in nodes}
    failed: dict[str, BaseException] = {}
    pending: dict[str, set[str]] = {
        nid: set(ds) for nid, ds in deps.items() if nid not in done
    }
    asset_pool_view: dict[str, str] = {}

    in_flight: dict[Future[str], str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        while pending or in_flight:
            # Submit every node whose deps are satisfied.
            ready = [nid for nid, ds in pending.items() if ds <= done.keys()]
            for nid in ready:
                rr, kind = nodes[nid]
                # Anchors need their refs resolved at submit time from the
                # current asset_pool snapshot. Portraits/locations/props
                # always have empty refs in the request body — _dispatch_one_image
                # just runs them with refs=[].
                if any(r.get("asset_id") for r in rr.get("references", [])):
                    # find the matching anchor entry
                    anchor_entry = next(
                        (a for a in render_plan["boundary_anchors"] if a["id"] == nid),
                        None,
                    )
                    if anchor_entry is not None:
                        rr = _resolve_anchor_refs(render_plan, anchor_entry, dict(done))
                in_flight[pool.submit(_dispatch_one_image, rr, kind)] = nid
                del pending[nid]
            if not in_flight:
                if pending:
                    # Cycle / unreachable nodes — should never happen.
                    raise RuntimeError(
                        f"_render_image_dag: cannot make progress; "
                        f"unreachable nodes={list(pending.keys())}"
                    )
                break
            # Wait for the first one to finish.
            finished = next(as_completed(list(in_flight.keys())))
            nid = in_flight.pop(finished)
            try:
                done[nid] = finished.result()
            except BaseException as e:
                failed[nid] = e
                print(
                    f"[render-image-dag] {nid} FAILED ({type(e).__name__}: {str(e)[:160]})",
                    flush=True,
                )
                # Continue processing — don't drag the whole DAG down on one
                # node's failure. Anchors blocked on the failure are still
                # listed in pending; they'll surface as the "stuck" subset
                # at the end. Other independent nodes keep running.

    if failed:
        msg = "; ".join(
            f"{nid}: {type(e).__name__}: {str(e)[:100]}"
            for nid, e in failed.items()
        )
        raise RuntimeError(
            f"_render_image_dag: {len(failed)} node(s) failed: {msg} — "
            f"re-run with --resume to retry"
        )
    return done


# ── 7. Render shot segments (Step 3: serial R2V per shot) ──────────────


_DEFAULT_REF_ROLES: tuple[str, ...] = ("portrait", "supporting_portrait", "place", "prop")


def _resolve_segment_refs(
    ri: dict[str, Any],
    asset_pool: dict[str, str],
    *,
    exclude_roles: frozenset[str] = frozenset(),
) -> list[str]:
    urls: list[str] = []
    for key in _DEFAULT_REF_ROLES:
        if key in exclude_roles:
            continue
        for asset_id in _split_ids(ri.get(key)):
            url = asset_pool.get(asset_id)
            if url:
                urls.append(url)
    return urls


def _resolved_ref_names(
    ri: dict[str, Any],
    asset_pool: dict[str, str],
    portrait_names: dict[str, str],
    *,
    exclude_roles: frozenset[str] = frozenset(),
) -> list[str]:
    names: list[str] = []
    for key in _DEFAULT_REF_ROLES:
        if key in exclude_roles:
            continue
        for asset_id in _split_ids(ri.get(key)):
            if asset_pool.get(asset_id):
                names.append(portrait_names.get(asset_id) or asset_id)
    return names


_BACKEND_EXCLUDE_ROLES: dict[str, frozenset[str]] = {
    "kling-3.0-pro-i2v": frozenset({"place"}),
}


def _build_ref_hint(names: list[str], backend_name: str) -> str:
    if not names:
        return ""
    if backend_name == "happyhorse-1.0-r2v":
        body = "、".join(f"[Image {i+1}]{n}" for i, n in enumerate(names))
        return f"\n\n参考图说明: {body}。"
    if backend_name in ("wan2.7-r2v", "wan2.7-i2v", "wan3.0-video"):
        body = "、".join(f"图{i+1}={n}" for i, n in enumerate(names))
        return f"\n\n参考图说明: {body}。"
    if backend_name == "seedance-2.0-r2v":
        body = "、".join(f"@Image{i+1}={n}" for i, n in enumerate(names))
        return f"\n\n参考图说明: {body}。"
    if backend_name == "kling-3.0-pro-i2v":
        body = "、".join(f"@Element{i+1}={n}" for i, n in enumerate(names))
        return f"\n\n参考图说明: {body}。"
    return ""


def _extract_last_frame(video_url: str, output_path: str | None) -> str:
    if not output_path:
        return ""
    src = output_path if Path(output_path).exists() else video_url
    if not src:
        return ""
    tail_png = Path(output_path).with_suffix(".tail.png")
    tail_png.parent.mkdir(parents=True, exist_ok=True)
    import av
    from PIL import Image
    container = av.open(src)
    last_frame = None
    for frame in container.decode(container.streams.video[0]):
        last_frame = frame
    container.close()
    if last_frame is None:
        raise RuntimeError(f"_extract_last_frame: PyAV decoded 0 frames from {src}")
    rgb = last_frame.to_ndarray(format="rgb24")
    Image.fromarray(rgb).save(str(tail_png))
    h, w = rgb.shape[:2]

    from videorlm.backends._common.oss_publisher import upload_file
    request_id = Path(output_path).stem + "_tail"
    public_url = upload_file(str(tail_png), prefix="tail_frame", request_id=request_id)
    print(
        f"[tail-frame] {Path(output_path).stem}: {w}x{h}, "
        f"size={tail_png.stat().st_size} bytes, "
        f"local={tail_png}, "
        f"oss={(public_url or '<no-oss>')[:90]}",
        flush=True,
    )
    return public_url or str(tail_png)


def _build_segment_request(
    seg: dict[str, Any],
    *,
    mode: str,
    first_url: str,
    ref_urls: list[str],
    prompt: str,
) -> SegmentRequest:
    fr = seg["segment_request"]
    return SegmentRequest(
        request_id=fr["request_id"], prompt=prompt,
        first_url=first_url, mode=mode,  # type: ignore[arg-type]
        reference_image_urls=tuple(ref_urls),
        duration_s=float(fr["duration_s"]), seed=int(fr["seed"]),
        output_path=fr.get("output_path"), log_dir=fr.get("log_dir"),
        negative_prompt=fr.get("negative_prompt", ""),
        resolution=fr.get("resolution", "1280x720"),
    )


def _last_segment_per_shot(render_plan: dict[str, Any]) -> dict[str, str]:
    out: dict[str, tuple[int, str]] = {}
    for seg_id, seg in render_plan["segments"].items():
        shot = seg["shot_id"]
        idx = int(seg["segment_index_in_shot"])
        if shot not in out or idx > out[shot][0]:
            out[shot] = (idx, seg_id)
    return {shot: seg_id for shot, (_, seg_id) in out.items()}


def _anchor_for_shot(render_plan: dict[str, Any], shot_id: str) -> str:
    n = next(i for i, sh in enumerate(render_plan["shots"]) if sh["id"] == shot_id)
    return render_plan["boundary_anchors"][n]["id"]


def _bridge_specs_by_from_shot(render_plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map from_shot_id → bridge spec (boundary_policies entry)."""
    out: dict[str, dict[str, Any]] = {}
    for entry in render_plan["boundary_policies"]:
        out[entry["from_shot"]] = entry
    return out


def render_segments(
    render_plan: dict[str, Any],
    anchor_urls: dict[str, str],
    portrait_urls: dict[str, str],
    location_urls: dict[str, str],
    prop_urls: dict[str, str],
    *,
    segment_validator: SegmentValidatorParams | None = None,
    bridge_executor: ThreadPoolExecutor | None = None,
) -> tuple[dict[str, str], list[Future[tuple[str, str]]]]:
    """Per-shot serial chain. Different shots run in parallel; segments
    inside a shot are serial.

    Returns:
      (segment_urls, bridge_futures) where each bridge future yields
      ``(bridge_id, url)``. Caller resolves the futures after this returns.
    """
    asset_pool = {**portrait_urls, **location_urls, **prop_urls}
    portrait_names: dict[str, str] = {
        pid: p.get("image_request", {}).get("name", pid)
        for pid, p in render_plan.get("portrait_plan", {}).items()
    }
    bridge_specs = _bridge_specs_by_from_shot(render_plan)
    by_shot_last_seg = _last_segment_per_shot(render_plan)
    by_shot: dict[str, list[dict[str, Any]]] = {}
    for seg in render_plan["segments"].values():
        by_shot.setdefault(seg["shot_id"], []).append(seg)
    for shot_id in by_shot:
        by_shot[shot_id].sort(key=lambda s: int(s["segment_index_in_shot"]))

    # Routing preview
    force_i2v = os.environ.get("RECA_FORCE_I2V") == "1"
    n_total = sum(len(segs) for segs in by_shot.values())
    print(
        f"[render-segments] dispatching {n_total} segments across {len(by_shot)} "
        f"shots (force_i2v={force_i2v})",
        flush=True,
    )

    out: dict[str, str] = {}
    failed_shots: list[tuple[str, Exception]] = []
    bridge_futures: list[Future[tuple[str, str]]] = []

    def _maybe_submit_outgoing_bridge(shot_id: str, shot_urls: dict[str, str]) -> None:
        """If this shot has an outgoing bridge and its end-anchor is ready,
        submit it to bridge_executor right away. Otherwise no-op."""
        if bridge_executor is None:
            return
        tr = bridge_specs.get(shot_id)
        if tr is None:
            return
        to_anchor_id = _anchor_for_shot(render_plan, tr["to_shot"])
        if to_anchor_id not in anchor_urls:
            return  # to-shot anchor not ready (shouldn't happen but defensive)
        # last segment of from_shot
        last_seg_id = by_shot_last_seg.get(shot_id)
        if last_seg_id is None or last_seg_id not in shot_urls:
            return
        from_video_url = shot_urls[last_seg_id]
        from_video_path = render_plan["segments"][last_seg_id]["segment_request"].get("output_path")
        first_url = _extract_last_frame(from_video_url, from_video_path)
        last_url = anchor_urls[to_anchor_id]
        bridge_futures.append(
            bridge_executor.submit(
                _dispatch_bridge_get_url,
                tr["bridge_request"], first_url, last_url,
            )
        )

    # Cap concurrent shot chains. Pre-round-4 default was `len(by_shot)`
    # which on long videos (18 shots) burst-launched 18 chains, each pulling
    # SP forks + happyhorse-r2v dispatch in parallel — saturated the dashscope
    # key pool and Cloudflare gateway.
    #
    # This is the framework-level "render pool" in the 3-pool trio
    # (planner / anchor_validator / render). Size is RECA_RENDER_POOL_SIZE
    # (default 8). The real per-key throttle still lives in backend KeyPool
    # (RECA_KEYPOOL_PER_KEY_CAP_*).
    from videorlm.framework._common.pools import RENDER_POOL_SIZE
    with ThreadPoolExecutor(max_workers=max(1, min(RENDER_POOL_SIZE, len(by_shot)))) as pool:
        fut_to_shot: dict[Future[dict[str, str]], str] = {
            pool.submit(
                _render_shot_chain, shot_id, segs, anchor_urls, asset_pool, portrait_names,
                segment_validator=segment_validator,
                render_plan=render_plan,
            ): shot_id
            for shot_id, segs in by_shot.items()
        }
        for fut in as_completed(fut_to_shot):
            shot_id = fut_to_shot[fut]
            try:
                shot_urls = fut.result()
                out.update(shot_urls)
                _maybe_submit_outgoing_bridge(shot_id, shot_urls)
            except Exception as e:
                print(
                    f"[render-segments] shot {shot_id} chain FAILED ({type(e).__name__}: {str(e)[:200]})",
                    flush=True,
                )
                failed_shots.append((shot_id, e))

    if failed_shots:
        msg = "; ".join(f"{sid}: {type(e).__name__}: {str(e)[:120]}" for sid, e in failed_shots)
        raise RuntimeError(
            f"render_segments: {len(failed_shots)} shot chain(s) failed — "
            f"re-run with --resume (other shots cached). Failed: {msg}"
        )
    return out, bridge_futures


def _render_shot_chain(
    shot_id: str,
    segs: list[dict[str, Any]],
    anchor_urls: dict[str, str],
    asset_pool: dict[str, str],
    portrait_names: dict[str, str],
    *,
    segment_validator: SegmentValidatorParams | None = None,
    render_plan: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Render one shot's segments serially. Each segment's tail becomes
    the next segment's first_frame. Raises on any render failure."""
    out: dict[str, str] = {}
    prev_tail_url = ""
    sp_state: dict[str, Any] | None = None
    if segment_validator is not None and segment_validator.enabled:
        try:
            sp_state = reconstruct_segment_planner_state(
                segment_validator.planner, segment_validator.story, shot_id,
            )
        except Exception as e:
            print(
                f"[segment-validate] {shot_id} cannot reconstruct SP state "
                f"({type(e).__name__}: {str(e)[:100]}); segment validation disabled for this shot",
                flush=True,
            )
            sp_state = None

    for seg in segs:
        fr = seg["segment_request"]
        sid = fr["request_id"]
        output_path = fr.get("output_path")

        cached = _load_cached_url(output_path)
        if cached:
            print(f"[render-skip] {sid} cached -> {cached[:80]}", flush=True)
            out[seg["id"]] = cached
            prev_tail_url = _extract_last_frame(cached, output_path)
            continue

        if seg["first_frame_path"] == "shot_start_anchor":
            first_url = anchor_urls.get(seg["start_anchor"], "")
            src = f"anchor[{seg['start_anchor']}]"
        else:
            first_url = prev_tail_url
            src = "prev_segment_tail"
        if not first_url:
            raise RuntimeError(f"[segment-trace] {sid}: NO first_url (src={src})")
        # DashScope's Wan HTTP backend and native Wan2.7 SDK both stage local
        # reference media before submission. The native SDK performs this
        # upload inside ``VideoSynthesis.async_call``; keep the local path here
        # so it can preserve the original hard-first-frame contract.
        local_media_ok = False
        if not first_url.startswith("http"):
            try:
                local_media_ok = (
                    for_kind("segment_i2v").capabilities().backend_name
                    in {"wan3.0-video", "wan2.7-r2v", "wan2.7-i2v"}
                    and Path(first_url).is_file()
                )
            except Exception:
                local_media_ok = False
        if not first_url.startswith("http") and not local_media_ok:
            raise RuntimeError(
                f"[segment-trace] {sid}: first_url is local path (src={src}): {first_url}"
            )
        print(f"[segment-trace] {sid}: first_url <- {src} = {first_url[:80]}", flush=True)

        ri = seg.get("reference_inputs", {}) or {}
        predicted_backend = for_kind("segment_r2v").capabilities().backend_name
        exclude_roles = _BACKEND_EXCLUDE_ROLES.get(predicted_backend, frozenset())
        ref_urls = _resolve_segment_refs(ri, asset_pool, exclude_roles=exclude_roles)
        provided_refs = render_plan.get("provided_reference_images") or []
        ref_urls.extend(
            str(item.get("path") or item.get("url"))
            for item in provided_refs
            if isinstance(item, dict) and (item.get("path") or item.get("url"))
        )

        # Dynamic mode decision (F6 contract):
        # mode = "i2v" if RECA_FORCE_I2V=1 OR len(ref_urls) == 0; else "r2v"
        if os.environ.get("RECA_FORCE_I2V") == "1":
            if ref_urls:
                print(
                    f"[segment-trace] {sid}: RECA_FORCE_I2V=1 dropping {len(ref_urls)} refs",
                    flush=True,
                )
                ref_urls = []
            seg_mode = "i2v"
        else:
            seg_mode = "i2v" if len(ref_urls) == 0 else "r2v"

        # Backend cap-truncation
        target_caps = for_kind(f"segment_{seg_mode}").capabilities()
        target_backend = target_caps.backend_name
        ref_names = _resolved_ref_names(ri, asset_pool, portrait_names, exclude_roles=exclude_roles)
        ref_names.extend(
            str(item.get("name") or item.get("role") or "reference")
            for item in provided_refs
            if isinstance(item, dict)
        )
        if target_caps.max_reference_images > 0 and len(ref_urls) > target_caps.max_reference_images:
            ref_urls = ref_urls[: target_caps.max_reference_images]
            ref_names = ref_names[: target_caps.max_reference_images]
        ref_hint = _build_ref_hint(ref_names, target_backend)
        prompt_with_hint = fr["prompt"] + ref_hint
        print(
            f"[segment-trace] {sid}: refs={len(ref_urls)} "
            f"duration={fr['duration_s']}s mode={seg_mode}",
            flush=True,
        )

        req = _build_segment_request(seg, mode=seg_mode, first_url=first_url,
                                     ref_urls=ref_urls, prompt=prompt_with_hint)
        result = dispatch_segment(req)
        video_url = getattr(result, "output_url", None) or getattr(result, "output_path", None) or ""
        tail_url = _extract_last_frame(video_url, output_path)

        # Cache the freshly-rendered segment NOW, before the validator
        # can fail. A validator hard-fail used to force a re-render on
        # resume (mp4 on disk, but ``.url`` sidecar never written → cache
        # miss). Early-save decouples "render done" from "validation done":
        # the heavy R2V/I2V API call is committed, and the cheap validator
        # call can retry next time without re-burning the GPU. The late
        # save below still runs after the validator block so best-of-N
        # can update the URL when it picks a different attempt.
        _save_cached_url(output_path, video_url)

        # ── Segment validator loop ────────────────────────────────────
        # F14: validate → router → micro_adjust / replan / seed_reroll.
        # Runs only when sp_state was successfully reconstructed above.
        # On final attempt's fail we still accept the render (no fail-hard)
        # so concat can complete; logs the un-validated state for human review.
        if sp_state is not None and segment_validator is not None and segment_validator.enabled:
            attempt_renders: list[dict[str, Any]] = []
            best_of_n_enabled = bool(getattr(segment_validator, "best_of_n", True))
            on_validator_error = str(getattr(segment_validator, "on_error", "raise") or "raise")
            # NB: an empty tuple is a MEANINGFUL value (validate-only), so we
            # test against None rather than falsiness.
            _allowed = getattr(segment_validator, "allowed_strategies", None)
            allowed_strategies = (
                tuple(_allowed) if _allowed is not None
                else ("seed_reroll", "micro_adjust", "replan")
            )
            try:
                from videorlm.framework.validator.segment import (
                    run_segment_validator,
                )
                from videorlm.framework.validator.segment.router import (
                    RouterError,
                    route_repair_strategy,
                )
                from videorlm.framework.segment_replanner import (
                    micro_adjust_single_segment,
                    replan_segments_for_shot,
                )
                portrait_urls_for_validator = [
                    asset_pool[pid] for pid in _split_ids(ri.get("portrait")) if asset_pool.get(pid)
                ]
                shot_start_state = ""
                shot_story_goal = ""
                shot_visual_intent = ""
                if render_plan is not None:
                    shot_meta = next(
                        (s for s in render_plan.get("shots", []) if s["id"] == shot_id),
                        None,
                    )
                    if shot_meta:
                        shot_start_state = shot_meta.get("start_state", "")
                        shot_story_goal = shot_meta.get("story_goal", "")
                        shot_visual_intent = shot_meta.get("visual_intent", "")
                max_attempts = max(0, int(getattr(segment_validator, "max_repair_attempts", 2)))
                current_seg = seg
                attempt_history: list = []
                for attempt_i in range(max_attempts + 1):
                    try:
                        judgment = run_segment_validator(
                            current_seg, shot_start_state, video_url, portrait_urls_for_validator, sp_state,
                            shot_story_goal=shot_story_goal,
                            shot_visual_intent=shot_visual_intent,
                            model=getattr(segment_validator, "validator_model", None),
                            axis_threshold=getattr(segment_validator, "axis_pass_threshold", None),
                            overall_threshold=getattr(segment_validator, "overall_pass_threshold", None),
                            fps=getattr(segment_validator, "video_sample_fps", None),
                        )
                    except Exception as exc:
                        if on_validator_error != "accept":
                            raise
                        print(
                            f"[segment-validate] {sid}: validator ERROR "
                            f"({type(exc).__name__}: {str(exc)[:200]}); "
                            f"on_error=accept — keeping the current render unvalidated",
                            flush=True,
                        )
                        break
                    label = f"attempt {attempt_i + 1}/{max_attempts + 1}"
                    print(
                        f"[segment-validate] {sid} {label}: pass={judgment.passed} "
                        f"overall={judgment.overall_score:.2f} "
                        f"(aes={judgment.aesthetic_score:.2f} "
                        f"glob={judgment.global_alignment_score:.2f} "
                        f"cons={judgment.action_consistency_score:.2f}) | "
                        f"{judgment.reason[:120]}",
                        flush=True,
                    )
                    is_terminal = judgment.passed or attempt_i >= max_attempts
                    snap_path = output_path or ""
                    if best_of_n_enabled and not is_terminal and output_path and Path(output_path).exists():
                        snap_path = f"{output_path}.attempt{attempt_i}.mp4"
                        try:
                            shutil.copy2(output_path, snap_path)
                        except Exception as _e:
                            print(
                                f"[segment-best-of-N] {sid}: snapshot copy FAILED "
                                f"({type(_e).__name__}: {str(_e)[:120]}); "
                                f"this attempt cannot be restored later",
                                flush=True,
                            )
                            snap_path = output_path
                    attempt_renders.append({
                        "idx": attempt_i,
                        "video_url": video_url,
                        "tail_url": tail_url,
                        "overall_score": float(judgment.overall_score),
                        "passed": bool(judgment.passed),
                        "snapshot_path": snap_path,
                    })
                    if judgment.passed:
                        break
                    if attempt_i >= max_attempts:
                        print(
                            f"[segment-validate] {sid}: max repairs exhausted "
                            f"({max_attempts}); will pick best-of-N attempt as final",
                            flush=True,
                        )
                        break

                    # Decide repair strategy via router (qwen3.6-plus text-only).
                    strategy = "seed_reroll"
                    rationale = "router unavailable (no router_cfg); defaulting to seed_reroll"
                    if getattr(segment_validator, "router_cfg", None) is not None:
                        try:
                            decision = route_repair_strategy(
                                judgment, current_seg, attempt_history,
                                segment_validator.router_cfg,
                            )
                            strategy = decision.strategy
                            rationale = decision.rationale
                        except RouterError as exc:
                            print(
                                f"[router] {sid}: ERROR ({type(exc).__name__}: "
                                f"{str(exc)[:200]}); defaulting to seed_reroll",
                                flush=True,
                            )
                    # Strategy allow-list. Configured policy wins over the
                    # router; an excluded strategy is downgraded to the first
                    # allowed one, always logged. An empty list means
                    # validate-only: score + log, never re-render.
                    if not allowed_strategies:
                        print(
                            f"[router] {sid}: allowed_strategies=() — validate-only mode, "
                            f"no repair attempted (last judgment kept)",
                            flush=True,
                        )
                        attempt_history.append(judgment)
                        break
                    if strategy not in allowed_strategies:
                        print(
                            f"[router] {sid}: strategy={strategy} not in "
                            f"allowed_strategies={allowed_strategies} — downgrading to "
                            f"{allowed_strategies[0]}",
                            flush=True,
                        )
                        strategy = allowed_strategies[0]
                        rationale = f"{rationale} [downgraded by allowed_strategies]"

                    # No silent downgrades — if a strategy's required config is
                    # missing, fail loud so the caller can fix the config.
                    if strategy == "replan" and getattr(segment_validator, "sp_replan_cfg", None) is None:
                        raise RuntimeError(
                            f"[router] {sid}: strategy=replan but sp_replan_cfg=None on "
                            f"SegmentValidatorParams — supply sp_replan_cfg or use a router "
                            f"that does not return 'replan' when the config is absent"
                        )
                    if strategy == "micro_adjust" and getattr(segment_validator, "sp_micro_adjust_cfg", None) is None and getattr(segment_validator, "sp_replan_cfg", None) is None:
                        raise RuntimeError(
                            f"[router] {sid}: strategy=micro_adjust but both "
                            f"sp_micro_adjust_cfg and sp_replan_cfg are None on "
                            f"SegmentValidatorParams — supply at least sp_micro_adjust_cfg"
                        )
                    attempt_history.append(judgment)
                    print(
                        f"[router] {sid}: strategy={strategy} | {rationale[:140]}",
                        flush=True,
                    )

                    if strategy == "seed_reroll":
                        seg_req = current_seg.setdefault("segment_request", {})
                        orig_seed = int(seg_req.get("seed", 0) or 0)
                        new_seed = (orig_seed + (attempt_i + 1) * 7919) % 2_147_483_647
                        seg_req["seed"] = new_seed
                        print(
                            f"[segment-seed-reroll] {sid}: seed "
                            f"{orig_seed} → {new_seed} (attempt {attempt_i + 1}); "
                            f"prompt unchanged ({len(seg_req.get('prompt', ''))} chars)",
                            flush=True,
                        )

                    if strategy == "replan":
                        feedback = (
                            f"segment `{sid}` 上次渲染未通过 validator "
                            f"(overall={judgment.overall_score:.2f}, "
                            f"axis aes/glob/cons={judgment.aesthetic_score:.2f}/"
                            f"{judgment.global_alignment_score:.2f}/"
                            f"{judgment.action_consistency_score:.2f}).\n"
                            f"validator reason: {judgment.reason}\n"
                            f"validator edit_prompt 建议: {judgment.edit_prompt}\n"
                            f"请重点修复 `{sid}` 这一段, 其他 segment 尽量沿用原版本。"
                            f"重点针对 weakest axis (物理接触 / motif / "
                            f"end_state 兑现) 修正。"
                        )
                        try:
                            shot_meta_full = next(
                                (s for s in render_plan.get("shots", [])
                                 if s["id"] == shot_id), None
                            ) if render_plan else None
                            sv_planner = getattr(segment_validator, "planner", {}) or {}
                            sv_shots = sv_planner.get("shots") or []
                            sv_shot_idx = next(
                                (i for i, s in enumerate(sv_shots)
                                 if s.get("id") == shot_id), None
                            )
                            sv_anchor_list = (
                                (sv_planner.get("boundarys") or {})
                                .get("boundary_anchors") or []
                            )
                            anchor_meta = (
                                sv_anchor_list[sv_shot_idx]
                                if sv_shot_idx is not None and sv_shot_idx < len(sv_anchor_list)
                                else {}
                            )
                            if sv_shot_idx is not None and sv_shot_idx < len(sv_shots):
                                shot_meta_full = sv_shots[sv_shot_idx]
                            sv_segments = sv_planner.get("segments") or {}
                            original_segs_for_shot = {
                                sid2: s2 for sid2, s2 in sv_segments.items()
                                if s2.get("shot_id") == shot_id
                            } or {
                                sid2: s2 for sid2, s2 in
                                (render_plan.get("segments") or {}).items()
                                if s2.get("shot_id") == shot_id
                            }
                            sv_skeleton = {k: v for k, v in sv_planner.items() if k != "segments"}
                            new_segs = replan_segments_for_shot(
                                sp_state, shot_meta_full, anchor_meta, sv_skeleton,
                                segment_validator.sp_replan_cfg,
                                original_segments=original_segs_for_shot,
                                user_feedback=feedback,
                            )
                            if sid in new_segs:
                                new_spec = copy.deepcopy(current_seg)
                                new_sr = new_spec.setdefault("segment_request", {})
                                replanned = new_segs[sid]
                                new_sr["prompt"] = (
                                    replanned.get("prompt")
                                    or replanned.get("segment_request", {}).get("prompt", "")
                                )
                                new_spec["end_state"] = replanned.get(
                                    "end_state", new_spec.get("end_state", "")
                                )
                                current_seg = new_spec
                                print(
                                    f"[segment-replan] {sid}: planner LLM revised; "
                                    f"new prompt len={len(new_sr['prompt'])}",
                                    flush=True,
                                )
                            else:
                                print(
                                    f"[segment-replan] {sid}: replan returned "
                                    f"but no key for sid; falling back to micro_adjust",
                                    flush=True,
                                )
                                strategy = "micro_adjust"
                        except Exception as exc:
                            print(
                                f"[segment-replan] {sid}: replan FAILED "
                                f"({type(exc).__name__}: {str(exc)[:200]}); "
                                f"falling back to micro_adjust",
                                flush=True,
                            )
                            strategy = "micro_adjust"

                    if strategy == "micro_adjust":
                        ma_cfg = (
                            getattr(segment_validator, "sp_micro_adjust_cfg", None)
                            or getattr(segment_validator, "sp_replan_cfg", None)
                        )
                        try:
                            new_spec = micro_adjust_single_segment(
                                sp_state, current_seg, judgment, ma_cfg,
                            )
                            current_seg = new_spec
                        except Exception as exc:
                            print(
                                f"[segment-micro-adjust] {sid}: FAILED "
                                f"({type(exc).__name__}: {str(exc)[:200]}); "
                                f"breaking out — keeping last successful render",
                                flush=True,
                            )
                            break
                    try:
                        repaired_fr = current_seg.get("segment_request", fr)
                        repaired_prompt = repaired_fr.get("prompt", prompt_with_hint)
                        req2 = _build_segment_request(
                            current_seg, mode=seg_mode, first_url=first_url, ref_urls=ref_urls,
                            prompt=repaired_prompt,
                        )
                        # A repair is a new paid provider submission.  The
                        # Wan adapter persists task ids for resume, so clear
                        # the segment cache before dispatching a repair;
                        # otherwise a seed reroll can silently reuse the old
                        # succeeded task and produce the identical clip.
                        repaired_output = req2.output_path
                        if repaired_output:
                            Path(f"{repaired_output}.wan30.json").unlink(missing_ok=True)
                            Path(f"{repaired_output}.url").unlink(missing_ok=True)
                        result = dispatch_segment(req2)
                        video_url = (
                            getattr(result, "output_url", None)
                            or getattr(result, "output_path", None)
                            or ""
                        )
                        tail_url = _extract_last_frame(video_url, output_path)
                    except BaseException as e:
                        print(
                            f"[segment-repair] {sid}: repair render FAILED "
                            f"({type(e).__name__}: {str(e)[:160]}); keeping last successful "
                            f"render and breaking out of validator loop",
                            flush=True,
                        )
                        break
            finally:
                # Restore best-of-N if enabled and we have multiple attempts
                if best_of_n_enabled and len(attempt_renders) > 1:
                    best = max(attempt_renders, key=lambda x: x["overall_score"])
                    best_snap = best.get("snapshot_path") or ""
                    if best_snap and best_snap != output_path and Path(best_snap).exists():
                        try:
                            shutil.copy2(best_snap, output_path)
                            video_url = best["video_url"]
                            tail_url = best["tail_url"]
                            print(
                                f"[segment-best-of-N] {sid}: restored attempt "
                                f"{best['idx']} (score={best['overall_score']:.2f}) "
                                f"as final",
                                flush=True,
                            )
                        except Exception as _e:
                            print(
                                f"[segment-best-of-N] {sid}: restore FAILED "
                                f"({type(_e).__name__}: {str(_e)[:120]})",
                                flush=True,
                            )
                    # Clean up intermediate snapshots
                    for ar in attempt_renders:
                        snap = ar.get("snapshot_path") or ""
                        if snap and snap != output_path and snap != best_snap and Path(snap).exists():
                            try:
                                Path(snap).unlink()
                            except Exception:
                                pass

        _save_cached_url(output_path, video_url)
        out[seg["id"]] = video_url
        prev_tail_url = tail_url
        print(f"[segment-trace] {sid}: done -> {video_url[:80]}", flush=True)

    return out


def _dispatch_bridge_get_url(
    bridge_req_dict: dict[str, Any],
    first_url: str,
    last_url: str,
) -> tuple[str, str]:
    """Submit a bridge render. Returns ``(bridge_id, url)``. Raises on failure."""
    output_path = bridge_req_dict.get("output_path")
    bid = bridge_req_dict["request_id"]
    cached = _load_cached_url(output_path)
    if cached:
        print(f"[render-skip] {bid} bridge cached -> {cached[:80]}", flush=True)
        return bid, cached

    req = BridgeRequest(
        request_id=bid, prompt=bridge_req_dict.get("prompt", ""),
        first_url=first_url, last_url=last_url,
        duration_s=float(bridge_req_dict["duration_s"]),
        seed=int(bridge_req_dict["seed"]),
        output_path=output_path, log_dir=bridge_req_dict.get("log_dir"),
        negative_prompt=bridge_req_dict.get("negative_prompt", ""),
        resolution=bridge_req_dict.get("resolution", "1280x720"),
    )
    result = dispatch_bridge(req)
    url = getattr(result, "output_url", None) or getattr(result, "output_path", None) or ""
    _save_cached_url(output_path, url)
    return bid, url


# ── 10. Concat final mp4 ───────────────────────────────────────────────


def concat_final(
    render_plan: dict[str, Any],
    segment_urls: dict[str, str],
    bridge_urls: dict[str, str],
    out_path: str | Path,
) -> str:
    """Concatenate per-shot segment chains + bridge clips in shot order.

    Raises if any cell is missing.
    """
    by_shot: dict[str, list[str]] = {}
    for seg_id, seg in render_plan["segments"].items():
        by_shot.setdefault(seg["shot_id"], []).append(seg_id)
    for shot_id in by_shot:
        by_shot[shot_id].sort(key=lambda sid: int(render_plan["segments"][sid]["segment_index_in_shot"]))

    bridges_by_from: dict[str, str] = {}
    for tr in render_plan["transitions"]:
        if tr["mode"] == "bridge":
            bridges_by_from[tr["from_shot"]] = tr["id"]
    transitions_to: dict[str, dict[str, Any]] = {tr["to_shot"]: tr for tr in render_plan["transitions"]}

    def _local_or_url(local_path: str | None, url: str | None) -> str | None:
        if local_path and Path(local_path).exists():
            return str(local_path)
        return url or None

    clips: list[tuple[str, bool]] = []
    for i, shot in enumerate(render_plan["shots"]):
        prev_tr = transitions_to.get(shot["id"])
        for k, seg_id in enumerate(by_shot.get(shot["id"], [])):
            seg_meta = render_plan["segments"][seg_id]["segment_request"]
            local = seg_meta.get("output_path")
            src = _local_or_url(local, segment_urls.get(seg_id))
            if not src:
                continue
            if not clips:
                trim = False
            elif k > 0:
                trim = True
            elif prev_tr is not None and prev_tr.get("mode") == "bridge":
                trim = True
            else:
                trim = False
            clips.append((src, trim))
        if i < len(render_plan["shots"]) - 1 and shot["id"] in bridges_by_from:
            tr_id = bridges_by_from[shot["id"]]
            bridge_entry = next(
                (b for b in render_plan["boundary_policies"] if b["id"] == tr_id),
                None,
            )
            local = bridge_entry["bridge_request"].get("output_path") if bridge_entry else None
            src = _local_or_url(local, bridge_urls.get(tr_id))
            if src:
                clips.append((src, True))

    if not clips:
        raise RuntimeError("[concat] no clips to concatenate")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[concat] {len(clips)} clips -> {out_path}", flush=True)
    for src, trim in clips:
        kind = "local" if Path(src).exists() else "remote"
        print(f"  [{kind}] trim={trim} {src[:90]}", flush=True)

    # Check audio
    has_audio_all = True
    for src, _ in clips:
        if not Path(src).exists():
            has_audio_all = False
            break
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_name", "-of", "default=nk=1:nw=1", src],
            capture_output=True, text=True,
        )
        if not probe.stdout.strip():
            has_audio_all = False
            break

    # Normalize: pick canonical (W, H, fps, audio sample_rate) from first
    # clip and align everything. Mixed-backend final.mp4 (happyhorse-r2v
    # 24fps/24kHz + wan2.7-i2v bridge 30fps/44.1kHz) otherwise produces
    # `moov atom not found` because ffmpeg silently writes mangled headers
    # when concat=...:a=1 sees mismatched audio rates / pts spacing.
    target_w = target_h = None
    target_fps_num = target_fps_den = None
    target_sr = None
    if clips:
        first_src = clips[0][0]
        if Path(first_src).exists():
            vprobe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height,r_frame_rate",
                 "-of", "csv=p=0", first_src],
                capture_output=True, text=True,
            )
            try:
                w, h, fps = vprobe.stdout.strip().split(",")
                target_w, target_h = int(w), int(h)
                fn, fd = fps.split("/")
                target_fps_num, target_fps_den = int(fn), int(fd) or 1
            except Exception:
                pass
            if has_audio_all:
                aprobe = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "a:0",
                     "-show_entries", "stream=sample_rate", "-of", "csv=p=0", first_src],
                    capture_output=True, text=True,
                )
                try:
                    target_sr = int(aprobe.stdout.strip())
                except Exception:
                    pass
    norm_v = (
        f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
        f"fps={target_fps_num}/{target_fps_den}"
        if target_w and target_h and target_fps_num else ""
    )
    norm_a = f"aresample={target_sr}" if target_sr else ""

    inputs: list[str] = []
    filters: list[str] = []
    for i, (src, trim) in enumerate(clips):
        inputs.extend(["-i", src])
        v_filters = []
        if trim:
            v_filters.append("select='gte(n\\,1)'")
        v_filters.append("setpts=PTS-STARTPTS")
        if norm_v:
            v_filters.append(norm_v)
        filters.append(f"[{i}:v]{','.join(v_filters)}[v{i}]")
        if has_audio_all:
            a_filters = []
            if trim:
                a_filters.append("aselect='gte(n\\,1)'")
            a_filters.append("asetpts=PTS-STARTPTS")
            if norm_a:
                a_filters.append(norm_a)
            filters.append(f"[{i}:a]{','.join(a_filters)}[a{i}]")

    if has_audio_all:
        filters.append(
            "".join(f"[v{i}][a{i}]" for i in range(len(clips)))
            + f"concat=n={len(clips)}:v=1:a=1[outv][outa]"
        )
        map_args = ["-map", "[outv]", "-map", "[outa]"]
        audio_codec = ["-c:a", "aac", "-b:a", "192k"]
    else:
        filters.append(
            "".join(f"[v{i}]" for i in range(len(clips)))
            + f"concat=n={len(clips)}:v=1:a=0[outv]"
        )
        map_args = ["-map", "[outv]"]
        audio_codec = ["-an"]

    filter_complex = ";".join(filters)
    cmd = [
        "ffmpeg", "-y",
        "-protocol_whitelist", "file,http,https,tcp,tls,crypto,data",
        *inputs,
        "-filter_complex", filter_complex,
        *map_args,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        *audio_codec,
        str(out_path),
    ]
    rsp = subprocess.run(cmd, capture_output=True, text=True)
    if rsp.returncode != 0:
        stderr_tail = "\n".join(rsp.stderr.strip().splitlines()[-30:])
        print(f"[concat] ffmpeg FAILED rc={rsp.returncode}\nffmpeg stderr (tail):\n{stderr_tail}", flush=True)
        raise RuntimeError(
            f"concat_final: ffmpeg exit {rsp.returncode}; see log for stderr tail"
        )
    return str(out_path)


# ── Top-level orchestrators ────────────────────────────────────────────


def run_planner(
    story: str,
    sp_cfg: Any,
    *,
    parent: Agent | None = None,
    parent_factory: Callable[[], Agent] | None = None,
    max_workers: int = 8,
    out_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Run the planning side end-to-end. Returns complete planner output dict.

    If ``out_dir`` is provided, ``freeze_prompts(out_dir)`` is called at
    start so the prompt module sources are captured alongside planner.json.
    """
    if out_dir is not None:
        try:
            from ._scripts._freeze_prompts import freeze_prompts
            freeze_prompts(out_dir)
        except Exception as e:
            print(f"[run_planner] freeze_prompts failed ({type(e).__name__}: {str(e)[:120]})", flush=True)

    owns_parent = parent is None
    if parent is None:
        parent = (parent_factory or _default_parent_factory)()
        parent.start()
    try:
        skeleton = plan_skeleton(parent, story)
        segments = plan_segments_all(parent.state, skeleton, sp_cfg, max_workers=max_workers)
        return merge_into_planner_output(skeleton, segments)
    finally:
        if owns_parent:
            parent.close()


def _validate_and_repair_anchors(
    render_plan: dict[str, Any],
    anchor_urls: dict[str, str],
    asset_pool: dict[str, str],
    params: "ValidatorParams",
    audit_meta: dict[str, Any] | None = None,
    protected_anchor_ids: set[str] | None = None,
) -> dict[str, str]:
    """Per-anchor validate → branch by reason tag → repair → re-validate loop."""
    from videorlm.framework.validator.anchor import (
        make_validator_from_segment_planner_state,
        validate_anchor_via_agent,
    )

    planner = params.planner
    story = params.story

    repair_root = params.repair_dir or (
        Path(render_plan["boundary_anchors"][0]["image_request"]["output_path"]).parent.parent
        / "anchors" / "repairs"
    )
    repair_root.mkdir(parents=True, exist_ok=True)

    out_urls: dict[str, str] = dict(anchor_urls)
    protected_anchor_ids = protected_anchor_ids or set()
    if audit_meta is not None:
        audit_meta.setdefault("validator_errors", 0)
        audit_meta.setdefault("repairs", 0)
        audit_meta.setdefault("protected_anchor_failures", 0)

    # id → {id, name, prompt} for the user-prompt description block.
    # planner.json schemas use lists of dicts; render_plan uses dicts of
    # specs. We read from planner.json (the design-time spec). We pass
    # TEXT descriptions only (no extra images on the vision channel) —
    # multi-image validation was slow (~60s/call w/ 4 inline portraits)
    # and v3 showed it ate attention off non-identity issues (a01/a06
    # regressed). Validator only needs to know what each ref ID *should*
    # look like; text spec is enough.
    portrait_specs: dict[str, dict[str, Any]] = {
        p["id"]: p for p in (planner.get("portrait_plan") or [])
        if isinstance(p, dict) and p.get("id")
    }
    # planner.location_plan is a dict of {loc_id: {props: {...}, ...}}.
    # anchor.reference_inputs.place references loc_id directly (zones were
    # retired 2026-05-17). Props live nested under location_plan[loc].props
    # OR at top-level planner.prop_plan — flatten both into a {id: spec} dict.
    place_specs: dict[str, dict[str, Any]] = {}
    prop_specs: dict[str, dict[str, Any]] = {
        p["id"]: p for p in (planner.get("prop_plan") or [])
        if isinstance(p, dict) and p.get("id")
    }
    location_plan = planner.get("location_plan") or {}
    if isinstance(location_plan, dict):
        for loc_id, loc in location_plan.items():
            if not isinstance(loc, dict):
                continue
            place_specs[loc_id] = loc
            for prop_id, prop in (loc.get("props") or {}).items():
                if isinstance(prop, dict):
                    prop_specs.setdefault(prop_id, {**prop, "id": prop.get("id", prop_id)})

    def _validate_one_anchor(i: int, anchor_entry: dict[str, Any]) -> tuple[str, str | None]:
        anchor_id = anchor_entry["id"]
        if anchor_id not in out_urls or not out_urls[anchor_id]:
            return anchor_id, None
        # In copy-source mode, later anchors are byte-identical continuity
        # frames. Audit the first inherited frame with GPT and propagate that
        # result down the chain; sending identical pixels repeatedly only
        # adds provider latency and can trigger gateway long-tail timeouts.
        image_request = anchor_entry.get("image_request") or {}
        source_anchor = image_request.get("source_anchor")
        if (
            os.environ.get("RECA_AUDIT_PROPAGATE_COPIED_ANCHORS", "0") == "1"
            and image_request.get("render_kind") == "image_edit"
            and source_anchor
            and source_anchor not in protected_anchor_ids
        ):
            if audit_meta is not None:
                audit_meta["propagated_copied_anchors"] = int(
                    audit_meta.get("propagated_copied_anchors", 0)
                ) + 1
            print(
                f"[validator] {anchor_id} copied from {source_anchor}; "
                "propagating prior audit result",
                flush=True,
            )
            return anchor_id, out_urls[anchor_id]
        shot = render_plan["shots"][i]
        anchor_spec = next(a for a in planner["boundarys"]["boundary_anchors"] if a["id"] == anchor_id)

        sp_state = reconstruct_segment_planner_state(planner, story, shot["id"])

        accepted_url = out_urls[anchor_id]
        # 先 OSS http(s) 后本地：validator 的下游 (QwenAgent._prompt_with_images)
        # 走 httpx.get(url) 来 inline base64，传 local path 会立刻
        # UnsupportedProtocol 挂掉。只有当 accepted_url 不是 http(s)
        # 时才退回 local path，由 _to_inline_data_uri 的本地分支兜底。
        local_anchor_path = anchor_entry["image_request"].get("output_path")
        if accepted_url and isinstance(accepted_url, str) and accepted_url.startswith(("http://", "https://")):
            validator_image_url = accepted_url
        elif local_anchor_path and Path(local_anchor_path).exists():
            validator_image_url = local_anchor_path
        else:
            validator_image_url = accepted_url

        # Resolve reference_inputs.portrait / prop / place (comma-separated
        # IDs) → text design specs so the anchor validator knows what each
        # referenced asset SHOULD look like. No images on the vision channel
        # (only anchor image goes there) — multi-image was slow + attention-
        # eating. Text descriptions of variants (e.g. "金棕毛发的猴王战神" vs
        # "山岳般巨猿") are enough to disambiguate sun_wukong vs giant_ape_wukong.
        ri = anchor_spec.get("reference_inputs", {}) or {}

        def _resolve_refs(field_name: str, specs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
            ids = [s.strip() for s in (ri.get(field_name) or "").split(",") if s.strip()]
            out: list[dict[str, Any]] = []
            for rid in ids:
                spec = specs.get(rid) or {}
                out.append({
                    "id": rid,
                    "name": spec.get("name", rid),
                    "prompt": spec.get("prompt", ""),
                })
            return out

        portrait_refs = _resolve_refs("portrait", portrait_specs)
        prop_refs = _resolve_refs("prop", prop_specs)
        place_refs = _resolve_refs("place", place_specs)

        # Multi-image variant disambiguation (passing portrait PNGs alongside
        # the anchor) was tested and reverted 2026-05-14: gpt-5.5 + images
        # scored 1/6 catch (vs 4/6 baseline text-only), and qwen3-vl-plus +
        # images hallucinated "missing 金冠/红披风" on anchors that visibly
        # had them. The text-only path with ref design-prompt descriptions
        # remains the most reliable. ``reference_portrait_urls`` is not
        # passed below — validator falls back to text refs.

        REPAIR_RENDER_TRIES = 3
        REPAIR_RENDER_BACKOFF_S = 60.0
        repairs_done = 0

        while True:
            try:
                validator = make_validator_from_segment_planner_state(sp_state, params.validator_cfg)
                with validator:
                    judgment = validate_anchor_via_agent(
                        validator_agent=validator,
                        anchor_id=anchor_id, shot_id=shot["id"],
                        anchor_prompt=anchor_spec["prompt"],
                        expected_start_state=shot["start_state"],
                        expected_end_state=shot.get("end_state", ""),
                        anchor_image_url=validator_image_url,
                        reference_inputs=ri,
                        portrait_refs=portrait_refs,
                        prop_refs=prop_refs,
                        place_refs=place_refs,
                    )
            except Exception as e:
                if audit_meta is not None:
                    audit_meta["validator_errors"] = int(audit_meta.get("validator_errors", 0)) + 1
                print(
                    f"[validator] {anchor_id} validate ERROR ({type(e).__name__}: {str(e)[:1500]}); "
                    f"accepting current URL",
                    flush=True,
                )
                return anchor_id, accepted_url

            if judgment["pass_or_not"] == "pass":
                return anchor_id, accepted_url
            if anchor_id in protected_anchor_ids:
                if audit_meta is not None:
                    audit_meta["protected_anchor_failures"] = int(
                        audit_meta.get("protected_anchor_failures", 0)
                    ) + 1
                print(
                    f"[validator] {anchor_id} provided input failed audit; "
                    "preserving user-supplied first frame",
                    flush=True,
                )
                return anchor_id, accepted_url
            if repairs_done >= params.max_repair_attempts:
                return anchor_id, accepted_url

            tag = (
                "design_bug" if "[design_bug]" in judgment["reason"] else
                "render_drift" if "[render_drift]" in judgment["reason"] else
                "low_quality" if "[low_quality]" in judgment["reason"] else "unknown"
            )
            if tag == "unknown":
                return anchor_id, accepted_url

            repair_path = repair_root / f"{anchor_id}.repair_{repairs_done+1}.png"
            new_url: str | None = None
            for render_try in range(1, REPAIR_RENDER_TRIES + 1):
                try:
                    if tag in ("design_bug", "low_quality"):
                        new_url = _repair_anchor_re_render(
                            judgment, anchor_spec, asset_pool, repair_path, tag,
                        )
                    else:
                        new_url = _repair_anchor_image_edit(
                            judgment, accepted_url, anchor_spec, asset_pool, repair_path,
                        )
                    break
                except Exception as e:
                    print(
                        f"[validator] {anchor_id} repair render_try {render_try}/{REPAIR_RENDER_TRIES} "
                        f"FAILED ({type(e).__name__}: {str(e)[:120]})",
                        flush=True,
                    )
                    if render_try >= REPAIR_RENDER_TRIES:
                        new_url = None
                        break
                    import time as _time
                    _time.sleep(REPAIR_RENDER_BACKOFF_S * render_try)

            if new_url is None or not new_url.startswith("http"):
                return anchor_id, accepted_url

            validator_image_url = str(repair_path) if repair_path.exists() else new_url
            accepted_url = new_url
            repairs_done += 1
            if audit_meta is not None:
                audit_meta["repairs"] = int(audit_meta.get("repairs", 0)) + 1

    n_anchors = len(render_plan["boundary_anchors"])
    max_workers = min(4, max(1, n_anchors))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_validate_one_anchor, i, anchor_entry): anchor_entry["id"]
            for i, anchor_entry in enumerate(render_plan["boundary_anchors"])
        }
        for fut in as_completed(futures):
            try:
                anchor_id, final_url = fut.result()
                if final_url is not None:
                    out_urls[anchor_id] = final_url
            except Exception:
                if audit_meta is not None:
                    audit_meta["validator_errors"] = int(audit_meta.get("validator_errors", 0)) + 1
                pass
    return out_urls


_VALIDATOR_PROMPT_QUOTE_RE = re.compile(r"改成[:：]\s*['\"](.+?)['\"](?:\s*[—\-]|$)", re.DOTALL)


def _repair_anchor_re_render(
    judgment: dict[str, Any],
    anchor_spec: dict[str, Any],
    asset_pool: dict[str, str],
    output_path: Path,
    tag: str,
) -> str:
    if tag == "design_bug":
        m = _VALIDATOR_PROMPT_QUOTE_RE.search(judgment["edit_prompt"])
        if not m:
            raise ValueError(
                f"{tag} edit_prompt missing quoted new prompt: {judgment['edit_prompt'][:160]!r}"
            )
        new_prompt = m.group(1).strip()
    else:
        new_prompt = anchor_spec["prompt"]
    # Build refs from the planner-level reference_inputs (portrait / place /
    # prop), same as initial render. Previously this read
    # anchor_spec.image_request.references which doesn't exist on the
    # planner-level spec → 0 refs → repair re-rendered text-only.
    refs_resolved = _resolve_refs_from_inputs(
        anchor_spec.get("reference_inputs", {}) or {}, asset_pool,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = output_path.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    req = ImageRequest(
        request_id=f"{anchor_spec['id']}_{tag}_repair",
        kind="anchor_image",
        prompt=new_prompt,
        references=tuple(refs_resolved),
        seed=0,
        resolution="1280x720",
        output_path=str(output_path),
        log_dir=str(log_dir),
        negative_prompt=judgment.get("negative_prompt", ""),
    )
    result = dispatch_image(req)
    return getattr(result, "output_url", None) or getattr(result, "output_path", None) or ""


def _repair_anchor_image_edit(
    judgment: dict[str, Any],
    source_url: str,
    anchor_spec: dict[str, Any],
    asset_pool: dict[str, str],
    output_path: Path,
) -> str:
    # Build extra refs from reference_inputs (portrait / place / prop) so
    # the image-edit call has the same context as the original render —
    # previously this also dropped to zero extras because of the same
    # anchor_spec.image_request bug.
    extras = _resolve_refs_from_inputs(
        anchor_spec.get("reference_inputs", {}) or {}, asset_pool,
    )
    refs = [ImageRef(role="source", url=source_url)] + extras
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = output_path.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    req = ImageRequest(
        request_id=f"{anchor_spec['id']}_drift_repair",
        kind="image_edit",
        prompt=judgment["edit_prompt"],
        references=tuple(refs),
        seed=0,
        resolution="1280x720",
        output_path=str(output_path),
        log_dir=str(log_dir),
        negative_prompt=judgment.get("negative_prompt", ""),
    )
    result = dispatch_image(req)
    return getattr(result, "output_url", None) or getattr(result, "output_path", None) or ""


def _stage_log(stage: str, t0: float, **extras: Any) -> float:
    import time as _time
    now = _time.time()
    dt = now - t0
    extra_str = " ".join(f"{k}={v}" for k, v in extras.items())
    print(f"[stage] {stage:20s} dt={dt:7.1f}s  {extra_str}", flush=True)
    return now


def _collect_concurrency_info(render_plan: dict[str, Any]) -> dict[str, Any]:
    info: dict[str, Any] = {
        "xsem_caps": {
            k: os.environ.get(f"RECA_XSEM_{k}", "") for k in (
                "ENABLE", "GPT_5_5", "HAPPYHORSE_R2V", "GPT_IMAGE_2_PRO",
                "WAN27_R2V", "WAN27_I2V",
            )
        },
        "backend_routing": {},
    }
    for kind in ("portrait", "anchor_image", "image_edit", "segment_r2v", "segment_i2v", "bridge"):
        try:
            caps = for_kind(kind).capabilities()
            info["backend_routing"][kind] = {
                "backend": caps.backend_name,
                "max_refs": caps.max_reference_images,
                "max_concurrency": caps.max_concurrency,
            }
        except Exception:
            info["backend_routing"][kind] = "unresolvable"
    return info


def _emit_summary(
    out_path: Path,
    timings: dict[str, float],
    counts: dict[str, int],
    render_plan: dict[str, Any],
    final_path: str,
    *,
    seed: int | None = None,
) -> None:
    total = sum(timings.values())
    conc = _collect_concurrency_info(render_plan)
    print("", flush=True)
    print("=" * 70, flush=True)
    print("  run_render summary", flush=True)
    print("=" * 70, flush=True)
    print(f"  total wall:  {total:7.1f}s ({total/60:.1f} min)", flush=True)
    print(f"  final mp4:   {final_path}", flush=True)
    print(f"  counts:      {counts}", flush=True)

    # Backend provenance: which image/video/planner backends actually produced
    # this run. Downstream OSS/manifest tooling reads this so consumers can
    # distinguish a gpt-image-2 run from a wan2.7-image-pro fallback run
    # (gateway/CF was 524-truncating gpt-image-2 on 2026-05-15; see commit
    # history for the wan-image migration). The tag is a free-form label set
    # by the launch wrapper (`RECA_IMAGE_BACKEND_TAG`); the per-kind fields
    # are the resolved registry names actually used at dispatch time.
    backend_info = {
        "tag": os.environ.get("RECA_IMAGE_BACKEND_TAG", ""),
        "planner_model": os.environ.get("RECA_PLANNER_MODEL", "qwen3.6-max-preview"),
        "render": {
            "portrait":     os.environ.get("RECA_RENDER_BACKEND_PORTRAIT", ""),
            "anchor_image": os.environ.get("RECA_RENDER_BACKEND_ANCHOR_IMAGE", ""),
            "image_edit":   os.environ.get("RECA_RENDER_BACKEND_IMAGE_EDIT", ""),
            "segment_r2v":  os.environ.get("RECA_RENDER_BACKEND_SEGMENT_R2V", ""),
            "segment_i2v":  os.environ.get("RECA_RENDER_BACKEND_SEGMENT_I2V", ""),
            "bridge":       os.environ.get("RECA_RENDER_BACKEND_BRIDGE", ""),
        },
    }
    summary_path = out_path.parent / "summary.json"
    summary_path.write_text(
        json.dumps({
            "total_s": round(total, 1),
            "final_mp4": final_path,
            "counts": counts,
            "timings_s": {k: round(v, 1) for k, v in timings.items()},
            "concurrency": conc,
            "seed": seed,
            "frozen_prompts": "_frozen_prompts/",
            "backend_info": backend_info,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[summary] written to {summary_path}", flush=True)
    # Sidecar file with the same info — easier to grep / pipeline-tag from
    # downstream tools without parsing the full summary.json.
    (out_path.parent / "_backend_info.json").write_text(
        json.dumps(backend_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_render(
    render_plan: dict[str, Any],
    out_path: str | Path,
    *,
    validator: ValidatorParams | None = None,
    segment_validator: SegmentValidatorParams | None = None,
    seed: int | None = None,
    state_callback: Callable[..., None] | None = None,
) -> str:
    """Run the render side end-to-end. Returns final mp4 path.

    Args:
      validator: anchor validator (between image DAG and segments).
      segment_validator: post-segment VLM validator (validate → router →
        micro_adjust / replan / seed_reroll per segment).
      seed: written into summary.json for reproducibility tracking.
      state_callback: optional ReCA-owned lifecycle callback. The Gateway may
        read the resulting state file, but must not derive business stages.
    """
    import time as _time
    timings: dict[str, float] = {}
    out_path = Path(out_path)
    audit_state = "audit_pending"

    def notify(stage: str, status: str = "running", **extra: Any) -> None:
        if state_callback is not None:
            state_callback(stage, status, **extra)

    # Single DAG covers portraits / locations / props / anchors.
    notify("asset_generation", "running")
    print(f"[stage] {'images-dag':20s} START", flush=True)
    t0 = _time.time()
    image_urls = _render_image_dag(render_plan, max_workers=16)
    timings["images_dag"] = _time.time() - t0
    _stage_log("images-dag", t0, n=len(image_urls))
    notify("asset_generation", "done", asset_count=len(image_urls))

    # Split out the views downstream code expects.
    portrait_urls = {pid: image_urls[pid] for pid in render_plan["portrait_plan"] if pid in image_urls}
    location_urls = {lid: image_urls[lid] for lid in render_plan["location_plan"] if lid in image_urls}
    prop_urls = {ppid: image_urls[ppid] for ppid in render_plan["prop_plan"] if ppid in image_urls}
    anchor_urls = {a["id"]: image_urls[a["id"]] for a in render_plan["boundary_anchors"] if a["id"] in image_urls}

    if validator is not None:
        notify("validating", "running", audit_state="audit_running")
        print(f"[stage] {'anchor-validator':20s} START anchors={len(anchor_urls)}", flush=True)
        t0 = _time.time()
        asset_pool = {**portrait_urls, **location_urls, **prop_urls}
        audit_meta: dict[str, Any] = {}
        anchor_urls = _validate_and_repair_anchors(
            render_plan,
            anchor_urls,
            asset_pool,
            validator,
            audit_meta,
            protected_anchor_ids=set(render_plan.get("protected_anchor_ids") or []),
        )
        timings["anchor_validator"] = _time.time() - t0
        _stage_log("anchor-validator", t0, anchors=len(anchor_urls))
        if audit_meta.get("validator_errors") or audit_meta.get("protected_anchor_failures"):
            anchor_audit_state = "audit_failed"
        elif audit_meta.get("repairs"):
            anchor_audit_state = "audit_repaired"
        else:
            anchor_audit_state = "audited"
        audit_state = anchor_audit_state
        notify("validating", "done", audit_state=anchor_audit_state, audit_meta=audit_meta)

    # Segments + inline bridge dispatch.
    notify("rendering", "running", video_state="rendering")
    print(f"[stage] {'segments':20s} START n={len(render_plan.get('segments', {}))}", flush=True)
    t0 = _time.time()
    bridge_executor = ThreadPoolExecutor(max_workers=4)
    try:
        segment_urls, bridge_futures = render_segments(
            render_plan, anchor_urls, portrait_urls, location_urls, prop_urls,
            segment_validator=segment_validator,
            bridge_executor=bridge_executor,
        )
        timings["segments"] = _time.time() - t0
        _stage_log("segments", t0, n=len(segment_urls), bridges_inflight=len(bridge_futures))
        if segment_validator is not None:
            notify("validating", "running", audit_state="audit_running")

        # Wait for inline-dispatched bridges to finish.
        print(f"[stage] {'bridges-wait':20s} START n={len(bridge_futures)}", flush=True)
        t0 = _time.time()
        bridge_urls: dict[str, str] = {}
        for fut in as_completed(bridge_futures):
            try:
                bid, url = fut.result()
                bridge_urls[bid] = url
            except Exception as e:
                print(f"[bridges-wait] FAILED: {type(e).__name__}: {str(e)[:160]}", flush=True)
                raise
        timings["bridges_wait"] = _time.time() - t0
        _stage_log("bridges-wait", t0, n=len(bridge_urls))
        if segment_validator is not None:
            # Preserve anchor audit failures. Segment validation must not
            # overwrite an earlier failed/protected-anchor verdict.
            if audit_state == "audit_pending":
                audit_state = "audited"
            notify("validating", "done", audit_state=audit_state)
        notify("rendering", "done", video_state="rendered")
    finally:
        bridge_executor.shutdown(wait=True)

    notify("concat", "running")
    print(f"[stage] {'concat':20s} START", flush=True)
    t0 = _time.time()
    final_path = concat_final(
        render_plan, segment_urls, bridge_urls, out_path,
    )
    timings["concat"] = _time.time() - t0
    _stage_log("concat", t0, path=final_path)

    counts = {
        "shots":    len(render_plan.get("shots", [])),
        "anchors":  len(render_plan.get("boundary_anchors", [])),
        "segments": len(render_plan.get("segments", {})),
        "bridges":  len(render_plan.get("boundary_policies", [])),
    }
    _emit_summary(out_path, timings, counts, render_plan, final_path, seed=seed)
    notify("succeeded", "done", video_state="complete")
    return final_path


def _default_parent_factory() -> Agent:
    from videorlm.backends.llm.agents import QwenAgent, QwenConfig
    from videorlm.framework._common.pools import PLANNER_POOL_SIZE, PLANNER_ROLE
    return QwenAgent(QwenConfig(
        system_prompt=PARENT_SYSTEM_PROMPT,
        role=PLANNER_ROLE,
        max_concurrency=PLANNER_POOL_SIZE,
    ))


__all__ = [
    "ValidatorParams",
    "SegmentValidatorParams",
    "concat_final",
    "merge_into_planner_output",
    "parse_json_block",
    "plan_skeleton",
    "render_segments",
    "run_planner",
    "run_render",
    "to_render_plan",
]
