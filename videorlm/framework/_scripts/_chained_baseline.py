"""Chained (closed-loop, fully sequential) generation baseline.

Renders a story as ONE continuous chain of segments: segment k's first frame is
segment k-1's **actually rendered** last frame — across shot boundaries too. The
only anchor image used is the very first shot's; every later shot starts from the
previous shot's tail instead of its own anchor.

Before rendering segment k (k > 0) the planner is re-consulted **closed-loop**:

    render k-1  →  caption(actual tail frame)  →  rewrite segment k's prompt
                                                  →  render k  →  ...

This is the "fully sequential" arm the parallel pipeline is compared against.
It reuses a completed run's plan so the comparison is single-variable.

Frozen from the source run (bit-identical to the parallel arm):
    shot split, segment count, per-segment duration, reference_inputs, seed,
    backend routing, resolution.
Changed (the two things under test):
    1. first frame source — previous segment's real tail, not this shot's anchor
    2. per-segment prompt / end_state — rewritten against the observed tail

Bridges are folded into the chain as ordinary segments (their duration and
prompt are kept) so total duration and clip count match the source run.

Usage
-----
    python3 -m videorlm.framework._scripts._chained_baseline \\
        --run videorlm/outputs/version_2.4/ex02_happyhorse \\
        --out videorlm/outputs/chained_pilot/ex02 \\
        --backend happyhorse --video-resolution 1920x1080

    # zero-cost assembly check (no API calls at all)
    python3 -m videorlm.framework._scripts._chained_baseline --run ... --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
_PROMPT_MD = Path(__file__).resolve().parent / "_chained_rewrite_system.md"

_OWN_BUCKET_MARKER = "intern-data-wlcb"


def _abs(path: str) -> Path:
    """render_plan stores repo-relative output paths; make them cwd-independent."""
    p = Path(path)
    return p if p.is_absolute() else (REPO / p)
_CAPTION_MODEL_DEFAULT = "qwen3-vl-plus"


# ── env / bootstrap (mirrors _smoke.py) ────────────────────────────────


def load_env() -> None:
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    if not os.environ.get("RECA_DISABLE_HTTP_PROXY"):
        url = os.environ.get("RECA_HTTP_PROXY_URL", "http://127.0.0.1:20172")
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ.setdefault(k, url)


def setup_render_defaults(backend: str) -> None:
    """Same routing table as _smoke.py, minus the bridge kind (unused here —
    every clip goes through dispatch_segment)."""
    os.environ.setdefault("RECA_RENDER_BACKEND_PORTRAIT", "gpt-image-2")
    os.environ.setdefault("RECA_RENDER_BACKEND_ANCHOR_IMAGE", "gpt-image-2")
    os.environ.setdefault("RECA_RENDER_BACKEND_IMAGE_EDIT", "gpt-image-2")
    if backend == "wan":
        os.environ.setdefault("RECA_RENDER_BACKEND_SEGMENT_R2V", "wan2.7-r2v")
        os.environ.setdefault("RECA_RENDER_BACKEND_SEGMENT_I2V", "wan2.7-r2v")
    elif backend == "ltx":
        os.environ.setdefault("RECA_RENDER_BACKEND_SEGMENT_I2V", "ltx-2.3")
        os.environ.setdefault("RECA_RENDER_BACKEND_SEGMENT_R2V", "ltx-2.3")
    else:
        os.environ.setdefault("RECA_RENDER_BACKEND_SEGMENT_R2V", "happyhorse-1.0-r2v")
        os.environ.setdefault("RECA_RENDER_BACKEND_SEGMENT_I2V", "happyhorse-1.0-i2v")


# ── chain assembly ─────────────────────────────────────────────────────


def build_chain(render_plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the render plan into one linear chain.

    Order: for each shot in ``shots`` order, its segments by
    ``segment_index_in_shot``; then that shot's outgoing bridge (if any),
    folded in as an ordinary link so total duration / clip count match the
    source run.

    Each link carries everything the renderer needs plus provenance for the
    trace: ``kind`` ("segment" | "bridge"), ``shot_id``, ``src_id``.
    """
    by_shot: dict[str, list[dict[str, Any]]] = {}
    for seg in render_plan["segments"].values():
        by_shot.setdefault(seg["shot_id"], []).append(seg)
    for sid in by_shot:
        by_shot[sid].sort(key=lambda s: int(s["segment_index_in_shot"]))

    bridge_by_from = {b["from_shot"]: b for b in render_plan.get("boundary_policies", [])}

    chain: list[dict[str, Any]] = []
    for shot in render_plan["shots"]:
        shot_id = shot["id"]
        for seg in by_shot.get(shot_id, []):
            sr = seg["segment_request"]
            chain.append({
                "kind": "segment",
                "src_id": seg["id"],
                "shot_id": shot_id,
                "duration_s": int(sr["duration_s"]),
                "seed": int(sr.get("seed", 0)),
                "resolution": sr.get("resolution", "1920x1080"),
                "negative_prompt": sr.get("negative_prompt", ""),
                "prompt": sr["prompt"],
                "end_state": seg.get("end_state", ""),
                "reference_inputs": seg.get("reference_inputs", {}) or {},
                "start_anchor": seg.get("start_anchor") or "",
                "segment_index_in_shot": int(seg["segment_index_in_shot"]),
            })
        br = bridge_by_from.get(shot_id)
        if br is not None:
            brq = br["bridge_request"]
            chain.append({
                "kind": "bridge",
                "src_id": br["id"],
                "shot_id": shot_id,
                "duration_s": int(brq["duration_s"]),
                "seed": int(brq.get("seed", 0)),
                "resolution": brq.get("resolution", "1920x1080"),
                "negative_prompt": brq.get("negative_prompt", ""),
                "prompt": brq.get("prompt", ""),
                "end_state": "",
                "reference_inputs": {},
                "start_anchor": "",
                "segment_index_in_shot": -1,
            })
    return chain


def _split_ids(field: Any) -> list[str]:
    if not field:
        return []
    if isinstance(field, list):
        return [str(x).strip() for x in field if x]
    return [s.strip() for s in str(field).split(",") if s.strip()]


_REF_ROLES = ("portrait", "supporting_portrait", "place", "prop")


def resolve_refs(link: dict[str, Any], asset_pool: dict[str, str]) -> list[str]:
    urls: list[str] = []
    for role in _REF_ROLES:
        for aid in _split_ids(link["reference_inputs"].get(role)):
            u = asset_pool.get(aid)
            if u:
                urls.append(u)
    return urls


# ── asset URLs (re-publish dead sidecars) ──────────────────────────────


def build_asset_pool(render_plan: dict[str, Any], *, dry_run: bool) -> dict[str, str]:
    """Map asset_id → live public URL.

    A run's ``.url`` sidecar may point at an expired provider-side temp URL
    (DashScope output links die after ~a day). Anything not on our own bucket
    is re-published from the local PNG so the chain always has fetchable refs.
    """
    pool: dict[str, str] = {}
    entries: list[tuple[str, dict[str, Any]]] = []
    for key in ("portrait_plan", "location_plan", "prop_plan"):
        for aid, ent in (render_plan.get(key) or {}).items():
            entries.append((aid, ent["image_request"]))
    for a in render_plan.get("boundary_anchors", []):
        entries.append((a["id"], a["image_request"]))

    n_reused = n_republished = n_missing = 0
    for aid, req in entries:
        out = _abs(req.get("output_path") or "")
        side = Path(str(out) + ".url")
        url = side.read_text(encoding="utf-8").strip() if side.exists() else ""
        if url.startswith("http") and _OWN_BUCKET_MARKER in url:
            pool[aid] = url
            n_reused += 1
            continue
        if not out.exists():
            n_missing += 1
            continue
        if dry_run:
            pool[aid] = f"<would-republish:{out.name}>"
            n_republished += 1
            continue
        from videorlm.backends._common.oss_publisher import upload_file
        new_url = upload_file(str(out), prefix="chained/assets", request_id=aid)
        if new_url:
            pool[aid] = new_url
            n_republished += 1
        else:
            n_missing += 1
    print(
        f"[chain] assets: reused={n_reused} republished={n_republished} missing={n_missing}",
        flush=True,
    )
    return pool


# ── closed-loop observation + rewrite ──────────────────────────────────


def caption_tail(tail_url: str, cfg: Any) -> str:
    """Describe the actual last frame. Fixed structured fields so the
    observation is comparable across links and auditable after the fact."""
    from videorlm.backends.llm.agents import QwenAgent

    instruction = (
        "这是一段视频的最后一帧。请客观描述画面, 只写你真正看见的, 不要推测剧情。\n"
        "按以下字段各写一行, 没有就写「无」:\n"
        "人物: <有谁、外观特征、服装>\n"
        "姿态: <每个人的身体姿态、手部位置、视线方向>\n"
        "持物/道具: <画面里出现的道具及其位置>\n"
        "场景: <地点、可辨认的环境元素>\n"
        "构图: <景别、机位角度、主体在画面中的位置>\n"
        "光线: <光源方向、色温、明暗>\n"
        "画面问题: <模糊/畸变/多手多指/穿模等, 没有就写「无」>"
    )
    with QwenAgent(cfg) as agent:
        return agent.prompt_with_images(instruction, [tail_url])


def rewrite_link(sp_state: dict[str, Any], link: dict[str, Any], observation: str, cfg: Any) -> dict[str, str]:
    """Rewrite this link's prompt/end_state so it starts from the OBSERVED
    frame while still landing on the planned narrative point.

    Forks from the segment_planner's post-turn state the same way
    ``micro_adjust_single_segment`` does (4-message tail: story, skeleton,
    per-shot task, the SP's own segments JSON), so the rewriter sees the
    original design intent.
    """
    from videorlm.backends.llm.agents import QwenAgent
    from videorlm.framework._common.fork import sub_conversation_with_system_swap
    from videorlm.framework.parsers import ParseError, retry_agent_call, strip_markdown_fence

    system = _PROMPT_MD.read_text(encoding="utf-8")
    state = sub_conversation_with_system_swap(sp_state, system, inherited_msg_count=4)

    user = (
        f"segment id = {link['src_id']}  (时长 {link['duration_s']}s)\n\n"
        f"[上一段实际末帧的客观描述 — 这是本段真实的第一帧]\n{observation}\n\n"
        f"[本段原始计划 prompt]\n{link['prompt']}\n\n"
        f"[本段原始计划 end_state]\n{link['end_state'] or '(无)'}\n\n"
        f"请按系统提示输出改写后的 JSON。"
    )

    def _parse(reply: str) -> dict[str, str]:
        d = json.loads(strip_markdown_fence(reply))
        p = (d.get("prompt") or "").strip()
        if not p:
            raise ValueError("rewrite returned empty prompt")
        return {
            "prompt": p,
            "end_state": (d.get("end_state") or link["end_state"] or "").strip(),
            "_change_summary": str(d.get("_change_summary", ""))[:200],
        }

    def _on_fail(label: str, attempt: int, exc: BaseException, head: str) -> None:
        print(f"[chain-rewrite {link['src_id']}] attempt {attempt} failed "
              f"({type(exc).__name__}: {str(exc)[:120]})", flush=True)

    agent = QwenAgent(cfg, state=state)
    try:
        with agent:
            return retry_agent_call(agent, user, _parse, max_retries=3,
                                    sleep_base_s=2.0, sleep_cap_s=60.0,
                                    on_fail=_on_fail, label=f"chain-rewrite {link['src_id']}")
    except ParseError as exc:
        raise RuntimeError(
            f"rewrite_link[{link['src_id']}]: exhausted retries; "
            f"last={type(exc.last_exc).__name__}: {str(exc.last_exc)[:200]}"
        ) from exc


# ── main chain loop ────────────────────────────────────────────────────


def run_chain(
    run_dir: Path,
    out_dir: Path,
    *,
    story: str,
    video_resolution: str,
    caption_cfg: Any,
    rewrite_cfg: Any,
    dry_run: bool,
    no_rewrite: bool,
    max_links: int = 0,
) -> dict[str, Any]:
    from videorlm.framework.segment_planner import reconstruct_segment_planner_state

    planner = json.loads((run_dir / "planner.json").read_text(encoding="utf-8"))
    render_plan = json.loads((run_dir / "render_plan.json").read_text(encoding="utf-8"))

    chain = build_chain(render_plan)
    if max_links > 0:
        chain = chain[:max_links]
        print(f"[chain] --max-links={max_links}: 只跑前 {len(chain)} 段(冒烟用)", flush=True)
    total_dur = sum(l["duration_s"] for l in chain)
    n_seg = sum(1 for l in chain if l["kind"] == "segment")
    n_br = sum(1 for l in chain if l["kind"] == "bridge")
    print(f"[chain] {run_dir.name}: {len(chain)} links ({n_seg} segment + {n_br} bridge), "
          f"总时长 {total_dur}s, shots={len(render_plan['shots'])}", flush=True)

    asset_pool = build_asset_pool(render_plan, dry_run=dry_run)

    # 只用第一个 anchor 起步
    first_anchor_id = render_plan["boundary_anchors"][0]["id"]
    first_url = asset_pool.get(first_anchor_id, "")
    if not first_url:
        raise RuntimeError(f"起始 anchor {first_anchor_id} 没有可用 URL(本地 PNG 也不在)")
    print(f"[chain] 起始首帧 = anchor[{first_anchor_id}]", flush=True)

    seg_dir = out_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    sp_cache: dict[str, dict[str, Any]] = {}
    trace: list[dict[str, Any]] = []
    t_run = time.perf_counter()

    for k, link in enumerate(chain):
        rec: dict[str, Any] = {
            "idx": k, "kind": link["kind"], "src_id": link["src_id"],
            "shot_id": link["shot_id"], "duration_s": link["duration_s"],
            "seed": link["seed"], "first_url_src": "anchor" if k == 0 else "prev_tail",
            "original_prompt": link["prompt"],
        }
        prompt = link["prompt"]

        # ① 观测 + 闭环改写(第一段没有上文,跳过)
        if k > 0 and not no_rewrite:
            t0 = time.perf_counter()
            if dry_run:
                obs, rw = "<dry-run observation>", {"prompt": prompt, "end_state": link["end_state"], "_change_summary": "<dry-run>"}
            else:
                obs = caption_tail(first_url, caption_cfg)
                rec["caption_s"] = round(time.perf_counter() - t0, 1)
                t1 = time.perf_counter()
                sp_state = sp_cache.get(link["shot_id"])
                if sp_state is None:
                    sp_state = reconstruct_segment_planner_state(planner, story, link["shot_id"])
                    sp_cache[link["shot_id"]] = sp_state
                rw = rewrite_link(sp_state, link, obs, rewrite_cfg)
                rec["rewrite_s"] = round(time.perf_counter() - t1, 1)
            prompt = rw["prompt"]
            rec.update(observation=obs, rewritten_prompt=prompt,
                       rewritten_end_state=rw["end_state"],
                       change_summary=rw["_change_summary"])

        # ② 渲染
        ref_urls = resolve_refs(link, asset_pool)
        mode = "i2v" if not ref_urls else "r2v"
        out_path = str(seg_dir / f"{k:03d}_{link['src_id']}.mp4")
        rec.update(n_refs=len(ref_urls), mode=mode, output_path=out_path)

        if dry_run:
            print(f"  [{k:02d}] {link['kind']:8s} {link['src_id']:44s} "
                  f"{link['duration_s']:2d}s refs={len(ref_urls)} mode={mode} "
                  f"first={'anchor' if k == 0 else 'prev_tail'}", flush=True)
            first_url = f"<tail-of-{k}>"
            trace.append(rec)
            continue

        from videorlm.backends.media import SegmentRequest, dispatch_segment
        from videorlm.framework.pipeline import _extract_last_frame

        t2 = time.perf_counter()
        req = SegmentRequest(
            request_id=f"chain{k:03d}_{link['src_id']}", prompt=prompt,
            first_url=first_url, mode=mode,
            reference_image_urls=tuple(ref_urls),
            duration_s=float(link["duration_s"]), seed=int(link["seed"]),
            output_path=out_path, log_dir=str(log_dir),
            negative_prompt=link["negative_prompt"],
            resolution=video_resolution,
        )
        result = dispatch_segment(req)
        video_url = getattr(result, "output_url", None) or getattr(result, "output_path", None) or ""
        rec["render_s"] = round(time.perf_counter() - t2, 1)
        rec["video_url"] = video_url
        Path(out_path + ".url").write_text(video_url, encoding="utf-8")

        tail = _extract_last_frame(video_url, out_path)
        if not tail:
            raise RuntimeError(f"link {k} ({link['src_id']}) 抽尾帧失败,链断在这里")
        first_url = tail
        rec["tail_url"] = tail
        print(f"  [{k:02d}] {link['src_id']:44s} 渲染 {rec['render_s']:6.1f}s  "
              f"caption {rec.get('caption_s', 0):5.1f}s  rewrite {rec.get('rewrite_s', 0):5.1f}s",
              flush=True)
        trace.append(rec)
        (out_dir / "chain_trace.json").write_text(
            json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - t_run
    summary = {
        "source_run": str(run_dir), "n_links": len(chain),
        "n_segment": n_seg, "n_bridge": n_br, "total_duration_s": total_dur,
        "wall_s": round(elapsed, 1),
        "render_s": round(sum(r.get("render_s", 0) for r in trace), 1),
        "caption_s": round(sum(r.get("caption_s", 0) for r in trace), 1),
        "rewrite_s": round(sum(r.get("rewrite_s", 0) for r in trace), 1),
        "dry_run": dry_run, "no_rewrite": no_rewrite,
    }

    # ③ concat —— 复用 concat_final:把整条链当成"单个 shot 的 N 段",
    #    它对 k>0 的每段丢首帧,正是链式接缝要的语义。
    if not dry_run:
        from videorlm.framework.pipeline import concat_final
        synth = {
            "shots": [{"id": "chain"}],
            "transitions": [],
            "boundary_policies": [],
            "segments": {
                f"c{r['idx']:03d}": {
                    "shot_id": "chain",
                    "segment_index_in_shot": r["idx"],
                    "segment_request": {"output_path": r["output_path"]},
                }
                for r in trace
            },
        }
        urls = {f"c{r['idx']:03d}": r.get("video_url", "") for r in trace}
        t3 = time.perf_counter()
        final = concat_final(synth, urls, {}, out_dir / "final.mp4")
        summary["concat_s"] = round(time.perf_counter() - t3, 1)
        summary["final_mp4"] = final

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "chain_trace.json").write_text(
        json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="源 run 目录(含 planner.json + render_plan.json)")
    p.add_argument("--out", required=True, help="输出目录")
    p.add_argument("--story", default=None, help="剧情 txt;不给则按 run 名推断")
    p.add_argument("--backend", choices=["wan", "happyhorse", "ltx"], default="happyhorse")
    p.add_argument("--video-resolution", default="1920x1080")
    p.add_argument("--caption-model", default=_CAPTION_MODEL_DEFAULT)
    p.add_argument("--rewrite-model", default=None,
                   help="闭环改写模型;默认跟 RECA_SP_MODEL 一致")
    p.add_argument("--dry-run", action="store_true", help="只装配链条,不调任何 API")
    p.add_argument("--no-rewrite", action="store_true",
                   help="只做链式渲染,不做闭环改写(消融用)")
    p.add_argument("--max-links", type=int, default=0,
                   help="只跑前 N 段(冒烟用, 0 = 全部)")
    args = p.parse_args()

    load_env()
    sys.path.insert(0, str(REPO))
    setup_render_defaults(args.backend)

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = REPO / run_dir
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    story_path = Path(args.story) if args.story else None
    if story_path is None:
        num = run_dir.name.split("_")[0].replace("ex", "")
        story_path = REPO / "input_examples" / "true_input" / f"example_{num}.txt"
    story = story_path.read_text(encoding="utf-8").strip() if story_path.exists() else ""
    print(f"[chain] story={story_path.name} ({len(story)} chars)"
          + ("  ⚠ 缺失,闭环改写将看不到原文" if not story else ""), flush=True)

    from videorlm.backends.llm.agents import QwenConfig
    from videorlm.framework._common.pools import PLANNER_POOL_SIZE, PLANNER_ROLE

    key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not args.dry_run and not key:
        raise SystemExit("DASHSCOPE_API_KEY 缺失")
    base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    caption_cfg = QwenConfig(model=args.caption_model, api_key=key, base_url=base,
                             temperature=0.2, request_timeout_s=300.0)
    rewrite_cfg = QwenConfig(
        model=args.rewrite_model or os.environ.get("RECA_SP_MODEL", "qwen3.6-max-preview"),
        api_key=key, base_url=base, temperature=0.3, request_timeout_s=900.0,
        role=PLANNER_ROLE, max_concurrency=PLANNER_POOL_SIZE,
    )

    s = run_chain(run_dir, out_dir, story=story, video_resolution=args.video_resolution,
                  caption_cfg=caption_cfg, rewrite_cfg=rewrite_cfg,
                  dry_run=args.dry_run, no_rewrite=args.no_rewrite,
                  max_links=args.max_links)
    print("\n[chain] " + json.dumps(s, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
