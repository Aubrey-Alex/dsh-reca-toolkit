#!/bin/bash
# Batch wrapper for the qwen-planner-on-DashScope + wan-image-pro stack.
#
# Why this exists:
#   - The legacy run_batch_18.sh routes plan_skeleton + segment_planner through
#     ziplab gpt-5.5, which Cloudflare 524-truncates at ~16k chars on long
#     JSON output. Every plan_skeleton retry fails the same way (verified
#     2026-05-15: curl + urllib + httpx + openai SDK all hit the same cap;
#     stream=False gets 524 at 100s origin timeout).
#   - The legacy stack also routes portrait/anchor/edit through ziplab
#     gpt-image-2, same CF 524 long-tail. The existing dispatch_image has
#     16-retry + sticky-shard breaker but on our pod today it still can't
#     complete a single image after 20+ minutes.
#
# What this wrapper does:
#   - Pins planner to qwen3.6-max-preview on DashScope (RECA_PLANNER_MODEL).
#   - Pins image-gen kinds to DashScope:
#       portrait     → wan2.7-image-pro (T2I via multimodal, verified)
#       anchor_image → wan2.7-image-pro
#       image_edit   → wan2.7-image-pro (qwen-image-2.0-pro has a framework
#                       bug with local-path ref refs that returns
#                       "url error" — wan2.7-image-pro accepts the same
#                       payload shape cleanly).
#   - Drops the dead KEY3 from DASHSCOPE_API_KEYS (account-level access
#     denied across every text model).
#   - Runs wan + happyhorse workers in parallel (same shape as
#     run_batch_18.sh's dual-pool design), each serial across slots.
#   - Each slot is idempotent: skipped if final.mp4 already exists.
#
# Defaults to the current MISS set:
#   wan        = 01 07 19 23 29 37 38 39 40 41 42 43 44
#   happyhorse = 01 13 17 31 32
# Override per-worker via BATCH_WAN_RANGE / BATCH_HH_RANGE.
#
# Usage:
#   nohup bash videorlm/scripts/run_batch_qwen.sh > /tmp/batch_qwen.log 2>&1 &
#   disown

set -u
cd /mnt/workspace/akide/code/unirlm-02

# Clean env (proxy strips + planner/xsem env exactly like run_ex01_qwen.sh)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export RECA_DISABLE_HTTP_PROXY=1
export DASHSCOPE_API_KEYS="${DASHSCOPE_API_KEYS:-}"
export RECA_XSEM_ENABLE=1
export RECA_PLAN_SKELETON_MAX_RETRIES=30

# Image backends → DashScope qwen-image-2.0-pro (off ziplab).
# As of 2026-05-16 qwen-image-2.0-pro handles BOTH pure T2I (0 refs) and
# i2i / image_edit (1-3 refs) via the sync multimodal-generation endpoint
# (see videorlm/backends/media/impl/dashscope/image/_primitives.py:
# generate_image_t2i_qwen). Output is photorealistic at 2048² (capped to
# the request resolution), ~8s T2I / ~14s i2i, identity-preserving on
# i2i. wan2.7-image-pro stays available — flip these env vars to revert.
export RECA_RENDER_BACKEND_PORTRAIT="${RECA_RENDER_BACKEND_PORTRAIT:-qwen-image-2.0-pro}"
export RECA_RENDER_BACKEND_ANCHOR_IMAGE="${RECA_RENDER_BACKEND_ANCHOR_IMAGE:-qwen-image-2.0-pro}"
export RECA_RENDER_BACKEND_IMAGE_EDIT="${RECA_RENDER_BACKEND_IMAGE_EDIT:-qwen-image-2.0-pro}"

# OSS-upload tag for downstream pipeline (this run's images are NOT from
# gpt-image-2; downstream tools should label exports accordingly).
export RECA_IMAGE_BACKEND_TAG="qwen-image-2.0-pro+qwen3.6-max-preview-planner"

ROOT=videorlm/outputs/version_2.4
LOG_DIR="$ROOT/_batch_logs"
mkdir -p "$LOG_DIR"

run_one() {
    local n=$1 backend=$2 resolution=$3
    local story="input_examples/true_input/example_${n}.txt"
    local out_dir="$ROOT/ex${n}_${backend}"
    local log="$LOG_DIR/ex${n}_${backend}.log"

    if [[ ! -f "$story" ]]; then
        echo "[batch-qwen] $(date +%H:%M:%S) MISS  ex${n} ${backend}  story=$story not found" >&2
        return 1
    fi
    if [[ -f "$out_dir/run/final.mp4" ]]; then
        local size=$(stat -c%s "$out_dir/run/final.mp4" 2>/dev/null || echo 0)
        echo "[batch-qwen] $(date +%H:%M:%S) SKIP  ex${n} ${backend}  final.mp4=${size}B exists" >&2
        return 0
    fi
    mkdir -p "$out_dir"

    # BATCH_RESUME=1 → pass --resume so plan_skeleton / plan_segments_all
    # skip when render_plan.json already exists. Default 0 (fresh plan via
    # qwen3.6-max-preview).
    local resume_flag=()
    if [[ "${BATCH_RESUME:-0}" == "1" && -f "$out_dir/render_plan.json" ]]; then
        resume_flag=(--resume)
        echo "[batch-qwen] $(date +%H:%M:%S) RESUME ex${n} ${backend} (render_plan.json exists, skipping plan)" >&2
    fi

    echo "[batch-qwen] $(date +%H:%M:%S) START ex${n} ${backend} res=${resolution}" >&2
    local t0=$(date +%s)
    python3 -m videorlm.framework._scripts._smoke \
        --story "$story" \
        --out-dir "$out_dir" \
        --label "ex${n}_${backend}" \
        --segments --render --validate --validate-segments \
        --backend "$backend" \
        --video-resolution "$resolution" \
        --seed 0 \
        "${resume_flag[@]}" \
        > "$log" 2>&1
    local rc=$?
    local dt=$(( $(date +%s) - t0 ))
    echo "[batch-qwen] $(date +%H:%M:%S) END   ex${n} ${backend} rc=${rc} dt=${dt}s" >&2
    return $rc
}

worker_loop() {
    local backend=$1 resolution=$2 range_var=$3 default_range=$4
    local range="${!range_var:-$default_range}"
    for n in $range; do
        run_one "$n" "$backend" "$resolution" || true
    done
    echo "[batch-qwen] $(date +%H:%M:%S) WORKER ${backend} EXIT" >&2
}

WAN_DEFAULT="01 07 19 23 29 37 38 39 40 41 42 43 44"
HH_DEFAULT="01 13 17 31 32"

worker_loop wan        1280x720  BATCH_WAN_RANGE "$WAN_DEFAULT" &
WAN_PID=$!
worker_loop happyhorse 1920x1080 BATCH_HH_RANGE  "$HH_DEFAULT"  &
HH_PID=$!

echo "[batch-qwen] $(date) spawned: wan_worker=${WAN_PID}  happyhorse_worker=${HH_PID}" >&2
wait "$WAN_PID"
echo "[batch-qwen] $(date) wan worker exited rc=$?" >&2
wait "$HH_PID"
echo "[batch-qwen] $(date) happyhorse worker exited rc=$?" >&2
echo "[batch-qwen] $(date) ALL DONE" >&2
