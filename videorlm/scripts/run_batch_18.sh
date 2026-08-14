#!/bin/bash
# Dual-pool batch launcher: 18 stories × 2 backends.
#
# Two INDEPENDENT sequential workers — wan and happyhorse don't compete
# for the same model slot pool, so they don't block each other. Each
# worker iterates example_02..example_19 in order. They share the
# DashScope key only via gpt-image-2 (anchors/portraits) and qwen3-vl-plus
# (segment validator); the video models are different (wan2.7 vs happyhorse-1.0)
# so each gets its own slot allocation.
#
# Resolutions are per-backend caps:
#   wan2.7-{i2v,r2v}        : 1280x720 only  (PROVIDER_SPEC.supports_resolutions)
#   happyhorse-1.0-{i2v,r2v}: 1280x720 / 1920x1080
# → wan = 720p, happyhorse = 1080p. There's no "common" 1080p path.
#
# Each run goes through the full pipeline:
#   plan_skeleton → plan_segments → render anchors/portraits/props/segments
#   → bridges → concat → final.mp4
# with --validate + --validate-segments enabled (anchor validator
# multi-image-aware + best-of-N segment validator).
#
# Idempotent: if outputs/version_2.4/ex<NN>_<backend>/run/final.mp4 exists,
# the slot is skipped (resume-friendly). Each worker writes a per-slot log
# to outputs/version_2.4/_batch_logs/ex<NN>_<backend>.log.
#
# Usage:
#   nohup bash videorlm/scripts/run_batch_18.sh > /tmp/batch_18.log 2>&1 &
#   disown

set -u
cd /mnt/workspace/akide/code/unirlm-02

# Pod 默认带 VSCode 的本地 HTTP proxy (http://10.32.2.229:20172) 拦截所有
# 出网流量. 对 OpenAI-compatible 流式响应 (ziplab gpt-5.5) 是灾难性的 —
# 中间盒在长 Chinese-prompt 流中会主动切连接,导致 RemoteProtocolError 大
# 量重试. 直连测试证明 pod 有外网出口 (`curl https://sub2api.ziplab.co`
# 1s 200 OK), 代理只是减慢 + 中断, 不是必需. 清掉 proxy 让所有子进程
# 走直连.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# Cross-process semaphore: ziplab 单 key 同时只能扛 1 路 gpt-5.5 长 stream.
# ex31 ✗ (两路 20:20 一起开) vs ex30 ✓ (差 19 min 启动) 实测确认: 隔离
# repro 单路 127s, 5194 字 reply 全收; 两 proc 并发立刻 RemoteProtocolError.
# 这里强制 gpt-5.5 跨进程串行 (RECA_XSEM_GPT_5_5=1), 单次 plan_skeleton
# ~2 min × 2 路 = 多 ~2 min, 完全可接受. 其他模型保持默认 pool size.
export RECA_XSEM_ENABLE=1
export RECA_XSEM_GPT_5_5=1

# Plan_skeleton 默认 max_retries=10. ex30-34 batch 数据显示成功 slot 也都有
# 1-5 次 RemoteProtocolError (ziplab gpt-5.5 stream 单次 success rate ~70-90%),
# ex31 不是配置坏, 是 10 次重试不够覆盖 ziplab 抖动. 提到 30 次,
# 30^0.3 ≈ 100% 完成率, 不增 wall time (有的 attempt 半秒就 fail 立刻重试).
export RECA_PLAN_SKELETON_MAX_RETRIES=30

ROOT=videorlm/outputs/version_2.4
LOG_DIR="$ROOT/_batch_logs"
mkdir -p "$LOG_DIR"

run_one() {
    local n=$1 backend=$2 resolution=$3
    local story="input_examples/true_input/example_${n}.txt"
    local out_dir="$ROOT/ex${n}_${backend}"
    local log="$LOG_DIR/ex${n}_${backend}.log"

    if [[ ! -f "$story" ]]; then
        echo "[batch] $(date +%H:%M:%S) MISS  ex${n} ${backend}  story=$story not found" >&2
        return 1
    fi
    if [[ -f "$out_dir/run/final.mp4" ]]; then
        local size=$(stat -c%s "$out_dir/run/final.mp4" 2>/dev/null || echo 0)
        echo "[batch] $(date +%H:%M:%S) SKIP  ex${n} ${backend}  final.mp4=${size}B exists" >&2
        return 0
    fi
    mkdir -p "$out_dir"

    # BATCH_RESUME=1 → pass --resume to _smoke so it skips plan_skeleton +
    # plan_segments_all when render_plan.json already exists, and re-renders
    # only the failed segment chains (others cached on disk). Default 0 (fresh).
    local resume_flag=()
    if [[ "${BATCH_RESUME:-0}" == "1" && -f "$out_dir/render_plan.json" ]]; then
        resume_flag=(--resume)
        echo "[batch] $(date +%H:%M:%S) RESUME ex${n} ${backend} (render_plan.json exists, skipping plan)" >&2
    fi

    echo "[batch] $(date +%H:%M:%S) START ex${n} ${backend} res=${resolution}" >&2
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
    echo "[batch] $(date +%H:%M:%S) END   ex${n} ${backend} rc=${rc} dt=${dt}s" >&2
    return $rc
}

worker_loop() {
    local backend=$1 resolution=$2
    # Range can be overridden via env BATCH_EX_RANGE (space-separated zero-
    # padded numbers), e.g. BATCH_EX_RANGE="30 31 32 33 34" bash run_batch_18.sh.
    # Default: ex02..ex19 (the original batch_18 set).
    local range="${BATCH_EX_RANGE:-02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19}"
    for n in $range; do
        run_one "$n" "$backend" "$resolution" || true
    done
    echo "[batch] $(date +%H:%M:%S) WORKER ${backend} EXIT" >&2
}

# Two independent serial queues running in parallel.
worker_loop wan 1280x720 &
WAN_PID=$!
worker_loop happyhorse 1920x1080 &
HH_PID=$!

echo "[batch] $(date)  spawned: wan_worker=${WAN_PID}  happyhorse_worker=${HH_PID}" >&2
wait "$WAN_PID"
echo "[batch] $(date)  wan worker exited rc=$?" >&2
wait "$HH_PID"
echo "[batch] $(date)  happyhorse worker exited rc=$?" >&2
echo "[batch] $(date)  ALL DONE — 18 stories × 2 backends" >&2
