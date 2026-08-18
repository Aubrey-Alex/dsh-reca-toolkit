"""Quick end-to-end smoke test for the framework.

Defaults
--------
- Agent: gpt-5.5 via OpenAI-compatible endpoint (OPENAI_BASE_URL/OPENAI_API_KEY
  in pipeline/.env). QwenAgent class is reused — it's a generic OpenAI
  client + base_url override.
- Render image kinds (portrait / anchor_image / image_edit): gpt-image-2-pro.
- Render video segment_r2v: happyhorse-1.0-r2v.

Usage
-----
    cd /mnt/workspace/akide/code/unirlm-02
    python3 -m videorlm.framework._scripts._smoke
    python3 -m videorlm.framework._scripts._smoke --segments --render
    python3 -m videorlm.framework._scripts._smoke --segments --render --validate
    python3 -m videorlm.framework._scripts._smoke --seed 42 --segments --render
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


# This file lives at <repo>/videorlm/framework/_scripts/_smoke.py
# parents[0] = _scripts/, [1] = framework/, [2] = videorlm/, [3] = repo root.
REPO = Path(__file__).resolve().parents[3]
ENV = REPO / ".env"
CONFIGS_DIR = REPO / "videorlm" / "configs"
OUTPUTS_DIR = REPO / "videorlm" / "outputs" / "version_2.2"
DEFAULT_STORY = REPO / "input_examples" / "example_01.txt"
DEFAULT_OUT_DIR = OUTPUTS_DIR / "ex01"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--story", default=str(DEFAULT_STORY), help="path to story txt")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="output dir")
    p.add_argument("--director-config", default=None,
                   help="JSON RunConfig written by the ReCA Director Gateway")
    p.add_argument("--label", default="smoke", help="log line prefix tag")
    p.add_argument("--segments", action="store_true", help="run plan_segments_all + to_render_plan")
    p.add_argument("--render", action="store_true", help="run full render after plan (requires --segments)")
    p.add_argument("--resume", action="store_true",
                   help="if render_plan.json exists in out_dir, skip plan + run render only")
    p.add_argument("--validate", action="store_true",
                   help="enable anchor validator+repair pass between i2i and R2V")
    p.add_argument("--validate-segments", action="store_true",
                   help="enable segment-level VLM validator + repair after each "
                        "segment render (validate → router → micro_adjust / replan). "
                        "Uses qwen3.6-plus router + qwen3.6-max-preview for repairs.")
    p.add_argument("--max-repair-attempts", type=int, default=2,
                   help="max repair attempts per anchor AND per segment")
    # ── segment verification 旋钮(全部可选;不传 = 保持历史行为) ──────────
    p.add_argument("--segment-axis-threshold", type=float, default=None,
                   metavar="F",
                   help="per-axis pass threshold for the segment validator "
                        "(default 0.6, env RECA_SEGMENT_AXIS_THRESHOLD)")
    p.add_argument("--segment-overall-threshold", type=float, default=None,
                   metavar="F",
                   help="overall pass threshold for the segment validator "
                        "(default 0.7, env RECA_SEGMENT_OVERALL_THRESHOLD)")
    p.add_argument("--segment-validator-model", default=None, metavar="NAME",
                   help="vision model used to judge segments "
                        "(default qwen3-vl-plus, env RECA_SEGMENT_VALIDATOR_MODEL)")
    p.add_argument("--segment-validator-fps", type=int, default=None, metavar="N",
                   help="server-side frame sampling rate of the judged video, 1-10 "
                        "(default 10, env RECA_SEGMENT_VALIDATOR_FPS)")
    p.add_argument("--segment-repair-strategies", default=None, metavar="CSV",
                   help="comma-separated subset of seed_reroll,micro_adjust,replan the "
                        "router may pick; anything else is downgraded to the first one. "
                        "Pass an empty string for validate-only (score + log, never "
                        "re-render). Default: all three. "
                        "env RECA_SEGMENT_REPAIR_STRATEGIES")
    p.add_argument("--no-segment-best-of-n", action="store_true",
                   help="do not snapshot/restore the highest-scoring attempt; always "
                        "keep the last render (env RECA_SEGMENT_BEST_OF_N=0)")
    p.add_argument("--segment-validator-on-error", choices=["raise", "accept"],
                   default=None,
                   help="what to do when the validator CALL fails (API/parse error, "
                        "NOT a fail verdict): raise = fail the shot chain (default), "
                        "accept = keep the current render and move on. "
                        "env RECA_SEGMENT_VALIDATOR_ON_ERROR")
    p.add_argument("--seed", type=int, default=0,
                   help="render plan seed; written into summary.json so the run is "
                        "reproducible (default 0)")
    p.add_argument("--backend", choices=["wan", "wan27", "happyhorse", "ltx"],
                   default="happyhorse",
                   help="video backend branch: wan (Wan3.0) / wan27 / happyhorse / ltx")
    p.add_argument("--force-i2v", action="store_true",
                   help="force every segment to use the i2v backend mode (drops extra refs)")
    p.add_argument("--video-resolution", default="1920x1080",
                   help="render plan video_resolution. happyhorse supports 1280x720 + "
                        "1920x1080; wan2.7 supports 1280x720 only — use 1280x720 when "
                        "--backend wan27 or render will raise BackendRenderError. "
                        "ltx-2.3 advertises 1280x720 / 1920x1080 but generates at the "
                        "nearest 64-aligned height (704/1088) and ffmpeg-pads/crops "
                        "back to exact target so concat stays clean.")
    return p.parse_args()


def load_env() -> None:
    file_values: dict[str, str] = {}
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        key = k.strip()
        value = v.strip().strip('"').strip("'")
        file_values[key] = value
        os.environ.setdefault(key, value)
    # Gateway workers inherit the long-lived gateway environment. Keep the
    # checked-in local .env authoritative for the planner model so changing a
    # provider does not require restarting the gateway process.
    if file_values.get("RECA_SP_MODEL"):
        os.environ["RECA_SP_MODEL"] = file_values["RECA_SP_MODEL"]
    if file_values.get("RECA_GPT_MODEL"):
        os.environ["RECA_GPT_MODEL"] = file_values["RECA_GPT_MODEL"]
    # The DSW host may inject a chat-only OPENAI_BASE_URL. The internal GPT
    # credential belongs to the Routify OpenAI-compatible gateway instead;
    # derive its base from the configured Responses endpoint for image calls.
    if file_values.get("RECA_GPT_RESPONSES_URL") and not file_values.get("OPENAI_BASE_URL"):
        responses_url = file_values["RECA_GPT_RESPONSES_URL"].rstrip("/")
        if responses_url.endswith("/responses"):
            os.environ["OPENAI_BASE_URL"] = responses_url[: -len("/responses")]
    # The working AVM deployment uses the DashScope OpenAI-compatible image
    # route for ReCA's image backends. Keep this fallback local to the
    # smoke/gateway entry point so the public source never contains a provider
    # credential and an explicit OPENAI_* override still wins.
    # The internal GPT credential is shared by the Responses auditor and the
    # OpenAI-compatible gpt-image-2 backend. Prefer it over the DashScope
    # fallback when no generic OpenAI key was explicitly configured.
    if not os.environ.get("OPENAI_API_KEY") and os.environ.get("RECA_GPT_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["RECA_GPT_API_KEY"]
    if not os.environ.get("OPENAI_API_KEY") and os.environ.get("DASHSCOPE_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["DASHSCOPE_API_KEY"]
        # DSW images can inject an unrelated OPENAI_BASE_URL globally. When
        # the key is inherited from DashScope, prefer its compatible endpoint
        # unless the local .env explicitly selected another image endpoint.
        if not file_values.get("OPENAI_BASE_URL"):
            os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if not os.environ.get("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if not os.environ.get("RECA_DISABLE_HTTP_PROXY"):
        _PROXY_URL = os.environ.get("RECA_HTTP_PROXY_URL", "http://127.0.0.1:20172")
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ.setdefault(k, _PROXY_URL)


def setup_path() -> None:
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(CONFIGS_DIR))


def _load_director_inputs(director_config: dict) -> dict:
    """Load optional user assets staged by the Gateway."""
    manifest_path = director_config.get("input_manifest")
    if not manifest_path:
        return {}
    try:
        value = json.loads(Path(str(manifest_path)).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"invalid director input manifest: {exc}") from exc
    return value if isinstance(value, dict) else {}


def _apply_director_inputs(render_plan: dict, inputs: dict) -> None:
    """Apply user assets while preserving ReCA's planner and asset DAG.

    A supplied first frame is the canonical visual source for the run. The
    first boundary anchor is preloaded from it; every later anchor is rendered
    serially with GPT Image2 image-edit from the previous anchor. This keeps
    the original no-input behavior unchanged while preventing independent
    T2I renders from drifting in identity, lighting, and background.
    """
    first_frame = inputs.get("first_frame")
    if isinstance(first_frame, dict):
        first_source = first_frame.get("path") or first_frame.get("url")
        anchors = render_plan.get("boundary_anchors") or []
        if first_source and anchors:
            render_plan.setdefault("preloaded_assets", {})[anchors[0]["id"]] = str(first_source)
            render_plan.setdefault("protected_anchor_ids", []).append(anchors[0]["id"])
            render_plan["canonical_reference_image"] = str(first_source)

            # Later anchors are edits of the preceding anchor, not fresh
            # text-to-image compositions. The source anchor is also a DAG
            # dependency so the image renders cannot race independently.
            for previous, current in zip(anchors, anchors[1:]):
                image_request = current.get("image_request") or {}
                image_request["render_kind"] = "image_edit"
                image_request["source_anchor"] = previous["id"]
                refs = image_request.setdefault("references", [])
                if not any(
                    r.get("role") == "source" and r.get("asset_id") == previous["id"]
                    for r in refs
                ):
                    refs.insert(0, {
                        "role": "source", "url": "", "asset_id": previous["id"],
                    })
                current["image_request"] = image_request

    refs = [item for item in inputs.get("reference_images", []) if isinstance(item, dict)]
    if not refs:
        return
    render_plan["provided_reference_images"] = refs
    # Anchor generation still happens when only reference images are supplied.
    # Preserve the direct URLs/paths so the image backend can use them without
    # requiring the planner to invent asset IDs for user-owned inputs.
    for anchor in render_plan.get("boundary_anchors") or []:
        image_request = anchor.get("image_request") or {}
        existing = image_request.setdefault("references", [])
        existing.extend(
            {
                "role": str(item.get("role") or "reference"),
                "url": str(item.get("path") or item.get("url")),
                "asset_id": "",
            }
            for item in refs
            if item.get("path") or item.get("url")
        )


def setup_render_defaults(backend: str = "happyhorse") -> None:
    """Set the RECA_RENDER_BACKEND_* env vars consumed by dispatch_*.

    Per backend caveats:
      - wan2.7-i2v ``render_segment`` raises by design — that backend is
        a bridge backend (native flf2v). So segment_i2v MUST NOT route to
        wan2.7-i2v; only bridges may.
      - ``ImageRefRole`` no longer includes ``start``/``end``; bridges carry
        first/last frames inside ``BridgeRequest`` itself.
      - Image kinds (portrait / anchor_image / image_edit) route to
        ``gpt-image-2`` (not ``-pro``) — that's the registry default.
    """
    # GPT Image2 is the canonical image backend for product runs. Wan remains
    # the video backend; operators can explicitly select a different image
    # provider with RECA_IMAGE_BACKEND.
    image_backend = os.environ.get(
        "RECA_IMAGE_BACKEND",
        "gpt-image-2",
    ).strip()
    os.environ.setdefault("RECA_RENDER_BACKEND_PORTRAIT", image_backend)
    os.environ.setdefault("RECA_RENDER_BACKEND_ANCHOR_IMAGE", image_backend)
    os.environ.setdefault("RECA_RENDER_BACKEND_IMAGE_EDIT", image_backend)
    if backend == "wan":
        wan_backend = os.environ.get("RECA_WAN30_BACKEND", "wan3.0-video").strip()
        # Keep ReCA's original Wan routing and replace only the provider model.
        # Segments remain first-frame anchored R2V/I2V; bridges remain FLF.
        os.environ.setdefault("RECA_RENDER_BACKEND_SEGMENT_R2V", wan_backend)
        os.environ.setdefault("RECA_RENDER_BACKEND_SEGMENT_I2V", wan_backend)
        os.environ.setdefault("RECA_RENDER_BACKEND_BRIDGE", wan_backend)
    elif backend == "wan27":
        # Preserve the original ReCA contract: every segment starts from a
        # hard first_frame (the previous segment's tail), while character,
        # location, and prop assets remain soft reference_image inputs.
        os.environ.setdefault("RECA_RENDER_BACKEND_SEGMENT_R2V", "wan2.7-r2v")
        os.environ.setdefault("RECA_RENDER_BACKEND_SEGMENT_I2V", "wan2.7-r2v")
        os.environ.setdefault("RECA_RENDER_BACKEND_BRIDGE", "wan2.7-i2v")
    elif backend == "ltx":
        # LTX-2.3 local-diffusion: same backend handles i2v / r2v / bridge.
        # bridge uses the keyframe-interpolation pipeline; segments use
        # ti2vid_two_stages. See backends/media/impl/local/ltx_v2_3/_runner.py.
        os.environ.setdefault("RECA_RENDER_BACKEND_SEGMENT_I2V", "ltx-2.3")
        os.environ.setdefault("RECA_RENDER_BACKEND_SEGMENT_R2V", "ltx-2.3")
        os.environ.setdefault("RECA_RENDER_BACKEND_BRIDGE", "ltx-2.3")
    else:  # happyhorse
        os.environ.setdefault("RECA_RENDER_BACKEND_SEGMENT_R2V", "happyhorse-1.0-r2v")
        os.environ.setdefault("RECA_RENDER_BACKEND_SEGMENT_I2V", "happyhorse-1.0-i2v")
        # Bridge needs native first+last anchoring; wan2.7-i2v is the bridge
        # specialist (its render_segment raises, but render_bridge is fine).
        os.environ.setdefault("RECA_RENDER_BACKEND_BRIDGE", "wan2.7-i2v")


def _log_backend_caps(tag: str) -> None:
    from videorlm.backends.media import for_kind

    kinds = [
        "portrait", "anchor_image", "image_edit",
        "segment_r2v", "segment_i2v", "bridge",
    ]
    print(f"[{tag}] === backend routing ===", flush=True)
    for kind in kinds:
        try:
            caps = for_kind(kind).capabilities()
            print(
                f"[{tag}]   {kind:14s} | {caps.backend_name:26s} | "
                f"max_refs={caps.max_reference_images} max_conc={caps.max_concurrency}",
                flush=True,
            )
        except Exception as e:
            print(f"[{tag}]   {kind:14s} | <unresolvable: {str(e)[:60]}>", flush=True)
    print(flush=True)


def main() -> None:
    args = parse_args()
    load_env()
    setup_path()
    setup_render_defaults(args.backend)
    if args.force_i2v:
        os.environ["RECA_FORCE_I2V"] = "1"
    print(
        f"[{args.label}] backend branch = {args.backend}, seed = {args.seed}, "
        f"validate = {args.validate}"
        + (f"  [force-i2v=ON]" if args.force_i2v else ""),
        flush=True,
    )

    # Cross-process semaphore — moved to _scripts/ alongside this file.
    from videorlm.framework._scripts._xprocess_semaphore import maybe_enable_from_env
    maybe_enable_from_env()

    from videorlm.backends.llm.agents import (
        OpenAIMessagesAgent,
        QwenAgent,
        QwenConfig,
    )
    from videorlm.framework import (
        PARENT_SYSTEM_PROMPT,
        SEGMENT_PLANNER_SYSTEM_PROMPT,
        ValidatorParams,
        SegmentValidatorParams,
        merge_into_planner_output,
        plan_segments_all,
        plan_skeleton,
        run_render,
        to_render_plan,
    )
    from videorlm.framework._scripts._freeze_prompts import freeze_prompts
    from videorlm.framework.validator.anchor import VALIDATOR_SYSTEM_PROMPT
    from videorlm.framework._common.pools import (
        ANCHOR_VALIDATOR_POOL_SIZE,
        ANCHOR_VALIDATOR_ROLE,
        PLANNER_POOL_SIZE,
        PLANNER_ROLE,
    )
    from videorlm.integrations.director.runtime import (
        write_artifact_manifest,
        write_audit_report,
        write_contact_sheet,
        write_run_report,
        write_state,
    )

    def _maybe_validator(planner: dict, story_text: str) -> "ValidatorParams | None":
        if not args.validate:
            return None
        # Reverted 2026-05-14 (afternoon): qwen3-vl-plus + CoT was tried
        # but its vision reading on our anchors proved unreliable —
        # hallucinated "missing 金冠 + 红披风" on anchors that visibly had
        # both → 16/16 false-FAIL after tightening hard criteria. gpt-5.5
        # baseline stayed at 4/6 recall but 0 FP (most reliable across
        # 5 ablated strategies). The DashScope key_pool wiring in
        # QwenAgent._chat_create stays in place — gpt-5.5 routes through
        # the OpenAI-compat single-key path and the pool branch no-ops.
        responses_key = os.environ.get("RECA_GPT_API_KEY", "").strip()
        responses_url = os.environ.get("RECA_GPT_RESPONSES_URL", "").strip()
        if responses_key and responses_url:
            validator_cfg = QwenConfig(
                model=os.environ.get("RECA_GPT_MODEL", "gpt-5.6-sol"),
                api_key=responses_key,
                base_url=responses_url,
                provider="openai_responses",
                system_prompt=VALIDATOR_SYSTEM_PROMPT,
                temperature=0.2,
                max_tokens=int(os.environ.get("RECA_GPT_AUDIT_MAX_TOKENS", "2048")),
                role=ANCHOR_VALIDATOR_ROLE,
                max_concurrency=ANCHOR_VALIDATOR_POOL_SIZE,
                request_timeout_s=float(os.environ.get("RECA_GPT_AUDIT_TIMEOUT_S", "45")),
                inline_images=True,
            )
            print(
                f"[{args.label}] anchor-audit provider={validator_cfg.provider or 'compat'} "
                f"model={validator_cfg.model} timeout={validator_cfg.request_timeout_s}s "
                f"minimal={os.environ.get('RECA_AUDIT_MINIMAL_CONTEXT', '0')}",
                flush=True,
            )
        else:
            api_key_v = os.environ.get("OPENAI_API_KEY", "")
            base_url_v = os.environ.get("OPENAI_BASE_URL", "") or "https://api.openai.com/v1"
            if not base_url_v.rstrip("/").endswith("/v1"):
                base_url_v = base_url_v.rstrip("/") + "/v1"
            validator_cfg = QwenConfig(
                model="gpt-5.5",
                api_key=api_key_v,
                base_url=base_url_v,
                system_prompt=VALIDATOR_SYSTEM_PROMPT,
                temperature=0.2,
                role=ANCHOR_VALIDATOR_ROLE,
                max_concurrency=ANCHOR_VALIDATOR_POOL_SIZE,
                request_timeout_s=900.0,
                inline_images=True,
            )
        return ValidatorParams(
            planner=planner,
            story=story_text,
            validator_cfg=validator_cfg,
            max_repair_attempts=args.max_repair_attempts,
        )

    def _maybe_segment_validator(planner: dict, story_text: str) -> "SegmentValidatorParams | None":
        if not args.validate_segments:
            return None
        dashscope_key_sv = os.environ.get("DASHSCOPE_API_KEY", "")
        if not dashscope_key_sv:
            raise SystemExit(
                f"[{args.label}] DASHSCOPE_API_KEY missing — needed for segment validator router"
            )
        dashscope_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        router_cfg = QwenConfig(
            model="qwen3.6-plus",
            api_key=dashscope_key_sv,
            base_url=dashscope_base,
            temperature=0.2,
            request_timeout_s=120.0,
        )
        replan_model = os.environ.get("RECA_PLANNER_MODEL", "qwen3.6-max-preview")
        sp_replan_cfg = QwenConfig(
            model=replan_model,
            api_key=dashscope_key_sv,
            base_url=dashscope_base,
            temperature=0.3,
            request_timeout_s=900.0,
        )
        # CLI > env > 模块默认(None 一路传下去,由 validator 模块决定)
        def _opt_float(cli_val, env_name):
            if cli_val is not None:
                return float(cli_val)
            raw = os.environ.get(env_name, "").strip()
            return float(raw) if raw else None

        def _opt_int(cli_val, env_name):
            if cli_val is not None:
                return int(cli_val)
            raw = os.environ.get(env_name, "").strip()
            return int(raw) if raw else None

        raw_strategies = args.segment_repair_strategies
        if raw_strategies is None:
            raw_strategies = os.environ.get("RECA_SEGMENT_REPAIR_STRATEGIES")
        if raw_strategies is None:
            strategies = ("seed_reroll", "micro_adjust", "replan")
        else:
            strategies = tuple(x.strip() for x in raw_strategies.split(",") if x.strip())
        unknown = [x for x in strategies
                   if x not in ("seed_reroll", "micro_adjust", "replan")]
        if unknown:
            raise SystemExit(
                f"[{args.label}] --segment-repair-strategies 含未知策略 {unknown}; "
                f"合法值: seed_reroll / micro_adjust / replan"
            )

        best_of_n = not args.no_segment_best_of_n
        if best_of_n and os.environ.get("RECA_SEGMENT_BEST_OF_N", "").strip() in ("0", "false", "False"):
            best_of_n = False

        on_error = (
            args.segment_validator_on_error
            or os.environ.get("RECA_SEGMENT_VALIDATOR_ON_ERROR", "").strip()
            or "raise"
        )
        if on_error not in ("raise", "accept"):
            raise SystemExit(
                f"[{args.label}] --segment-validator-on-error 只能是 raise / accept, 得到 {on_error!r}"
            )

        params = SegmentValidatorParams(
            planner=planner,
            story=story_text,
            router_cfg=router_cfg,
            sp_replan_cfg=sp_replan_cfg,
            sp_micro_adjust_cfg=sp_replan_cfg,
            max_repair_attempts=args.max_repair_attempts,
            axis_pass_threshold=_opt_float(args.segment_axis_threshold,
                                           "RECA_SEGMENT_AXIS_THRESHOLD"),
            overall_pass_threshold=_opt_float(args.segment_overall_threshold,
                                              "RECA_SEGMENT_OVERALL_THRESHOLD"),
            validator_model=(args.segment_validator_model
                             or os.environ.get("RECA_SEGMENT_VALIDATOR_MODEL", "").strip()
                             or None),
            video_sample_fps=_opt_int(args.segment_validator_fps,
                                      "RECA_SEGMENT_VALIDATOR_FPS"),
            allowed_strategies=strategies,
            best_of_n=best_of_n,
            on_error=on_error,
        )
        print(
            f"[{args.label}] segment-verification: ON  "
            f"model={params.validator_model or '<default qwen3-vl-plus>'}  "
            f"thresholds=axis:{params.axis_pass_threshold if params.axis_pass_threshold is not None else '<0.6>'}"
            f"/overall:{params.overall_pass_threshold if params.overall_pass_threshold is not None else '<0.7>'}  "
            f"max_repair={params.max_repair_attempts}  "
            f"strategies={list(params.allowed_strategies) or '[] (validate-only)'}  "
            f"best_of_n={params.best_of_n}  on_error={params.on_error}",
            flush=True,
        )
        return params

    out_dir = Path(args.out_dir)
    story_path = Path(args.story)
    tag = args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    director_config: dict = {}
    if args.director_config:
        try:
            director_config = json.loads(Path(args.director_config).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SystemExit(f"[{tag}] invalid --director-config: {exc}") from exc
    story = story_path.read_text(encoding="utf-8").strip()
    director_inputs = _load_director_inputs(director_config)
    planner_story = story
    constraints = []
    if director_config.get("duration_s"):
        constraints.append(f"目标总时长约 {int(director_config['duration_s'])} 秒")
    if director_config.get("style"):
        constraints.append(f"整体风格：{director_config['style']}")
    if director_config.get("aspect_ratio"):
        constraints.append(f"画幅比例：{director_config['aspect_ratio']}")
    if constraints:
        planner_story += "\n\n[ReCA Director 约束]\n" + "；".join(constraints)
    run_id = str(director_config.get("run_id") or out_dir.name)

    def director_state(stage: str, status: str = "running", **extra: object) -> None:
        explicit_audit_state = "audit_state" in extra
        current_state: dict = {}
        try:
            current_state = json.loads((out_dir / "run" / "reca_state.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        default_audit = current_state.get(
            "audit_state",
            "audit_pending" if director_config.get("enable_audit", True) else "audit_skipped",
        )
        audit_state = str(extra.pop("audit_state", default_audit))
        video_state = str(extra.pop("video_state", current_state.get("video_state", "pending")))
        write_state(
            out_dir,
            stage=stage,
            state="succeeded" if stage == "succeeded" else "running",
            audit_state=audit_state,
            video_state=video_state,
            run_id=run_id,
            status=status,
            run_config=director_config,
            **extra,
        )
        if explicit_audit_state or stage in {"succeeded", "failed"}:
            write_audit_report(
                out_dir,
                state=audit_state,
                details={"stage": stage, "status": status, **extra},
            )
        if stage in {"succeeded", "failed"}:
            write_run_report(out_dir, state=stage, details={"status": status, **extra})
            # The report itself is an artifact; regenerate the manifest after
            # writing it so the published list reflects the terminal run.
            write_artifact_manifest(out_dir, run_id=run_id)

    write_state(out_dir, stage="planning", state="running", run_id=run_id,
                run_config=director_config, audit_state=("audit_pending" if director_config.get("enable_audit", True) else "audit_skipped"))
    write_audit_report(
        out_dir,
        state=("audit_pending" if director_config.get("enable_audit", True) else "audit_skipped"),
        details={"stage": "planning"},
    )
    write_artifact_manifest(out_dir, run_id=run_id)

    def on_failure(exc_type, exc_value, exc_traceback) -> None:
        audit_state = "audit_failed" if director_config.get("enable_audit", True) else "audit_skipped"
        write_state(out_dir, stage="failed", state="failed", run_id=run_id,
                    audit_state=audit_state, video_state="failed", error=str(exc_value))
        write_audit_report(out_dir, state=audit_state, details={"error": str(exc_value)})
        write_run_report(out_dir, state="failed", details={"error": str(exc_value)})
        write_artifact_manifest(out_dir, run_id=run_id)
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = on_failure
    print(f"[{tag}] story={story_path.name} ({len(story)} chars), out_dir={out_dir}")

    # Snapshot prompt module sources at run start (F9).
    try:
        freeze_prompts(out_dir)
        print(f"[{tag}] frozen_prompts -> {out_dir / '_frozen_prompts'}/", flush=True)
    except Exception as e:
        print(f"[{tag}] freeze_prompts skipped ({type(e).__name__}: {str(e)[:120]})", flush=True)

    rp_path = out_dir / "render_plan.json"
    resume_mode = args.resume and rp_path.exists()
    if resume_mode:
        print(f"[{tag}] --resume + {rp_path.name} exists → skipping plan", flush=True)
        render_plan = json.loads(rp_path.read_text())
        _log_backend_caps(tag)

        t0 = time.time()
        if args.render:
            t3 = time.time()
            planner_for_v = (
                json.loads((out_dir / "planner.json").read_text())
                if (args.validate or args.validate_segments) and (out_dir / "planner.json").exists()
                else {}
            )
            director_state("rendering", "running", video_state="rendering")
            vp = _maybe_validator(planner_for_v, planner_story)
            svp = _maybe_segment_validator(planner_for_v, planner_story)
            final_mp4 = run_render(
                render_plan, out_dir / "run" / "final.mp4",
                validator=vp,
                segment_validator=svp,
                seed=args.seed,
                state_callback=director_state,
            )
            write_contact_sheet(out_dir)
            write_artifact_manifest(out_dir, run_id=run_id)
            director_state("succeeded", "done", video_state="complete", progress=1.0)
            print(f"[{tag}] run_render OK in {time.time()-t3:.1f}s -> {final_mp4}", flush=True)
        print(f"[{tag}] total wall time: {time.time()-t0:.1f}s", flush=True)
        return

    # ── Planner LLM (parent + segment_planner): qwen3.6-max-preview on DashScope. ──
    # Was gpt-5.5 via the OpenAI-compatible gateway; gateway's CF gateway truncates long stream output at
    # ~16k chars / ~3-4 min wall (and 524s the non-stream path at 100s origin
    # timeout), so plan_skeleton's ~32k-char JSON reply gets cut mid-portrait_plan
    # every attempt and json.loads always fails. Diagnosis: curl/urllib/httpx/
    # openai SDK all hit the same cap, so it's infra not code. Routing the text-
    # only planners (parent / segment_planner / sp_replan) to DashScope bypasses
    # the gateway → CF chain entirely. Image-gen backends (gpt-image-2) still
    # route through gateway via RECA_RENDER_BACKEND_*. Anchor validator stays on
    # gpt-5.5 because qwen3-vl-plus was unreliable on anchor vision (see comment
    # near _maybe_validator above). Validator output is small so the cap doesn't
    # bite there.
    # Planner endpoint override (used for A/B testing gpt-5.5 vs qwen3.6-max-preview):
    #   RECA_PLANNER_API_KEY + RECA_PLANNER_BASE_URL — when both set, route planner
    #   through that endpoint instead of DashScope. Default stays DashScope.
    override_key = os.environ.get("RECA_PLANNER_API_KEY", "").strip()
    override_url = os.environ.get("RECA_PLANNER_BASE_URL", "").strip()
    if override_key and override_url:
        planner_api_key = override_key
        planner_base_url = override_url
        if not planner_base_url.rstrip("/").endswith("/v1"):
            planner_base_url = planner_base_url.rstrip("/") + "/v1"
        planner_default_model = "gpt-5.5"
    else:
        planner_api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not planner_api_key:
            raise SystemExit(f"[{tag}] DASHSCOPE_API_KEY missing in .env (needed for qwen3.6-max-preview planner)")
        planner_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        planner_default_model = "qwen3.6-max-preview"
    planner_model = os.environ.get("RECA_PLANNER_MODEL", planner_default_model)
    dashscope_key = planner_api_key  # name kept for the cfg lines below

    # OPENAI key/url still used by anchor validator (vision) and downstream
    # image-gen backends; surface here for log clarity but no longer the
    # planner's source.
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "") or "https://api.openai.com/v1"
    if not base_url.rstrip("/").endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"

    # Planner ingress selector. ``RECA_PLANNER_API_PATH``:
    #   "auto" (default) — gpt-5* models use /v1/messages with thinking,
    #     everything else uses /v1/chat/completions (legacy path).
    #   "messages" / "chat" — force the named path regardless of model.
    api_path = os.environ.get("RECA_PLANNER_API_PATH", "auto").strip().lower()
    if api_path == "auto":
        api_path = "messages" if planner_model.lower().startswith("gpt-5") else "chat"
    if api_path not in ("messages", "chat"):
        raise SystemExit(f"[{tag}] bad RECA_PLANNER_API_PATH={api_path!r} (expected messages/chat/auto)")

    thinking_budget = int(os.environ.get("RECA_PLANNER_THINKING_BUDGET", "16000"))
    # qwen3-series thinking via DashScope extra_body. Toggle with
    # RECA_QWEN_THINKING (default on; the planner is reasoning-heavy and the
    # cost diff is small per call).
    qwen_thinking = os.environ.get("RECA_QWEN_THINKING", "1").strip() not in ("0", "", "false", "False")

    # Pick the provider-specific config: Anthropic Messages API has its
    # own thinking_budget_tokens field on ``OpenAIMessagesConfig``;
    # everything else uses ``QwenConfig`` (DashScope chat.completions).
    if api_path == "messages":
        from videorlm.backends.llm.agents.openai_messages import OpenAIMessagesConfig
        cfg = OpenAIMessagesConfig(
            model=planner_model, api_key=planner_api_key, base_url=planner_base_url,
            system_prompt=PARENT_SYSTEM_PROMPT,
            temperature=0.3, request_timeout_s=900.0,
            role=PLANNER_ROLE, max_concurrency=PLANNER_POOL_SIZE,
            thinking_budget_tokens=thinking_budget,
        )
    else:
        cfg = QwenConfig(
            model=planner_model, api_key=planner_api_key, base_url=planner_base_url,
            system_prompt=PARENT_SYSTEM_PROMPT,
            temperature=0.3, request_timeout_s=900.0,
            role=PLANNER_ROLE, max_concurrency=PLANNER_POOL_SIZE,
            enable_thinking=qwen_thinking and "qwen" in planner_model.lower(),
        )
    sp_api_key = os.environ.get("RECA_SP_API_KEY", "").strip() or planner_api_key
    sp_base_url = os.environ.get("RECA_SP_BASE_URL", "").strip() or planner_base_url
    if not sp_base_url.rstrip("/").endswith("/v1"):
        sp_base_url = sp_base_url.rstrip("/") + "/v1"
    sp_api_path = os.environ.get("RECA_SP_API_PATH", "").strip().lower() or api_path
    sp_model = os.environ.get("RECA_SP_MODEL", "qwen3.6-max-preview")
    if sp_api_path == "messages":
        from videorlm.backends.llm.agents.openai_messages import OpenAIMessagesConfig
        sp_cfg = OpenAIMessagesConfig(
            model=sp_model,
            api_key=sp_api_key,
            base_url=sp_base_url,
            system_prompt=SEGMENT_PLANNER_SYSTEM_PROMPT,
            temperature=0.3, request_timeout_s=900.0,
            role=PLANNER_ROLE, max_concurrency=PLANNER_POOL_SIZE,
            thinking_budget_tokens=int(os.environ.get("RECA_SP_THINKING_BUDGET", "8000")),
        )
    else:
        sp_cfg = QwenConfig(
            model=sp_model,
            api_key=sp_api_key,
            base_url=sp_base_url,
            system_prompt=SEGMENT_PLANNER_SYSTEM_PROMPT,
            temperature=0.3, request_timeout_s=900.0,
            role=PLANNER_ROLE, max_concurrency=PLANNER_POOL_SIZE,
            enable_thinking=qwen_thinking,
        )
    if not sp_cfg.api_key:
        raise SystemExit(f"[{tag}] segment planner API key missing")
    dashscope_key = planner_api_key  # name kept for downstream legacy refs

    print(
        f"[{tag}] planner api_path={api_path} model={planner_model} "
        f"sp_api_path={sp_api_path} sp_model={sp_model} "
        f"thinking_budget={getattr(cfg, 'thinking_budget_tokens', 0)} "
        f"qwen_thinking={getattr(cfg, 'enable_thinking', False)}",
        flush=True,
    )
    _log_backend_caps(tag)

    if api_path == "messages":
        parent_agent_ctx = OpenAIMessagesAgent(cfg)
    else:
        parent_agent_ctx = QwenAgent(cfg)

    t0 = time.time()
    with parent_agent_ctx as parent:
        skeleton = plan_skeleton(parent, planner_story)
        (out_dir / "skeleton.json").write_text(json.dumps(skeleton, ensure_ascii=False, indent=2))
        director_state("planning", "running", shot_count=len(skeleton.get("shots", [])))
        print(f"[{tag}] plan_skeleton OK; shots={len(skeleton['shots'])}", flush=True)

        if args.segments:
            segments = plan_segments_all(parent.state, skeleton, sp_cfg)
            planner = merge_into_planner_output(skeleton, segments)
            (out_dir / "planner.json").write_text(json.dumps(planner, ensure_ascii=False, indent=2))
            render_plan = to_render_plan(
                planner, out_dir / "run", seed=args.seed,
                video_resolution=args.video_resolution,
            )
            _apply_director_inputs(render_plan, director_inputs)
            (out_dir / "render_plan.json").write_text(json.dumps(render_plan, ensure_ascii=False, indent=2))
            director_state("planning", "done", segment_count=len(segments))
            print(f"[{tag}] plan_segments_all OK; segments={len(segments)}", flush=True)

            if args.render:
                t3 = time.time()
                director_state("rendering", "running", video_state="rendering")
                vp = _maybe_validator(planner, planner_story)
                svp = _maybe_segment_validator(planner, planner_story)
                final_mp4 = run_render(
                    render_plan, out_dir / "run" / "final.mp4",
                    validator=vp,
                    segment_validator=svp,
                    seed=args.seed,
                    state_callback=director_state,
                )
                write_contact_sheet(out_dir)
                write_artifact_manifest(out_dir, run_id=run_id)
                director_state("succeeded", "done", video_state="complete", progress=1.0)
                print(f"[{tag}] run_render OK in {time.time()-t3:.1f}s -> {final_mp4}", flush=True)

    print(f"[{tag}] total wall time: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
