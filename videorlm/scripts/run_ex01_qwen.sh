#!/bin/bash
# Launch ex01 (both wan + happyhorse backends) with the planner LLM switched
# to qwen3.6-max-preview on DashScope.
#
# Why explicit wrapper (vs inline shell):
#   The Claude harness pre-seeds http_proxy / https_proxy in its shell
#   snapshot, and inline exports don't reliably propagate to the python
#   subprocess. This script enforces:
#     - Strip pod-level HTTP proxy (it interferes with ziplab/DashScope
#       streaming, see run_batch_18.sh preamble).
#     - DASHSCOPE_API_KEYS limited to KEY1+KEY2 (KEY3 is a dead account
#       — 400 Access denied on every text model).
#     - Planner xsem / retry tuning lifted from run_batch_18.sh.
#
# Usage:
#   bash videorlm/scripts/run_ex01_qwen.sh
#   tail -f videorlm/outputs/version_2.4/_batch_logs/ex01_{wan,happyhorse}.log

set -u
cd /mnt/workspace/akide/code/unirlm-02

# ── 1. Network: no pod-level proxy. Force python to inherit a clean env.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export RECA_DISABLE_HTTP_PROXY=1

# ── 2. DashScope keys: drop KEY3 (dead account; returns 400 Access denied
#       on every model including qwen3.6-max-preview / qwen3.6-plus /
#       qwen3-vl-plus). Keep KEY1 + KEY2 for multi-key rotation in
#       DashScopeQwenAgent._chat_create.
export DASHSCOPE_API_KEYS="${DASHSCOPE_API_KEYS:-}"

# ── 3. Cross-process semaphore on media + chat dispatch.
export RECA_XSEM_ENABLE=1
# qwen3.6-max-preview xsem default cap = 12 (see _xprocess_semaphore.py:82),
# we leave it at default — DashScope key1+key2 has plenty of throughput.
# Plan_skeleton retry budget: 30 retries (covers transient flake; usually
# attempt 1 or 2 succeeds when ziplab is bypassed).
export RECA_PLAN_SKELETON_MAX_RETRIES=30

# ── 4. Image-gen backends: skip ziplab gpt-image-2 (CF 524 long-tail same
#       root cause as the planner stream cut). Route to DashScope instead:
#         portrait / anchor_image → wan2.7-image (pure-T2I capable, DashScope)
#         image_edit              → qwen-image-2.0-pro (i2i with ≥1 ref)
#       gpt-image-2 stays an option in the registry — flip this back to
#       "gpt-image-2" when ziplab/CF recovers and you want OpenAI image
#       fidelity. NOTE for OSS upload: outputs from this run are
#       wan/qwen-generated, not gpt-image-2 — tag accordingly downstream.
# 2026-05-16 update: qwen-image-2.0-pro now wired for BOTH T2I and i2i via
# the sync multimodal-generation endpoint (see
# videorlm/backends/media/impl/dashscope/image/_primitives.py). 2026-05-15's
# "qwen-image-2.0-pro doesn't support T2I" was framework wrapper limitation,
# not upstream — the SDK ImageSynthesis path 400s but the chat-style sync
# endpoint accepts 0-image input cleanly. ~8s T2I, ~14s i2i, photorealistic
# 2048² output. wan2.7-image-pro stays available — flip env to revert.
export RECA_RENDER_BACKEND_PORTRAIT="${RECA_RENDER_BACKEND_PORTRAIT:-qwen-image-2.0-pro}"
export RECA_RENDER_BACKEND_ANCHOR_IMAGE="${RECA_RENDER_BACKEND_ANCHOR_IMAGE:-qwen-image-2.0-pro}"
export RECA_RENDER_BACKEND_IMAGE_EDIT="${RECA_RENDER_BACKEND_IMAGE_EDIT:-qwen-image-2.0-pro}"

# OSS upload tag: surfaced in the manifest so downstream consumers know
# these renders came from qwen-image, not the original gpt-image-2.
export RECA_IMAGE_BACKEND_TAG="qwen-image-2.0-pro+qwen3.6-max-preview-planner"

LOG_DIR=videorlm/outputs/version_2.4/_batch_logs
mkdir -p videorlm/outputs/version_2.4/ex01_wan videorlm/outputs/version_2.4/ex01_happyhorse "$LOG_DIR"

echo "[ex01-qwen] $(date '+%H:%M:%S') launching wan + happyhorse"
echo "[ex01-qwen] DASHSCOPE keys: $(echo "$DASHSCOPE_API_KEYS" | tr ',' '\n' | wc -l) keys"

nohup python3 -m videorlm.framework._scripts._smoke \
    --story input_examples/true_input/example_01.txt \
    --out-dir videorlm/outputs/version_2.4/ex01_wan \
    --label ex01_wan \
    --segments --render --validate --validate-segments \
    --resume \
    --backend wan --video-resolution 1280x720 --seed 0 \
    > "$LOG_DIR/ex01_wan.log" 2>&1 &
WAN_PID=$!
disown $WAN_PID

nohup python3 -m videorlm.framework._scripts._smoke \
    --story input_examples/true_input/example_01.txt \
    --out-dir videorlm/outputs/version_2.4/ex01_happyhorse \
    --label ex01_happyhorse \
    --segments --render --validate --validate-segments \
    --resume \
    --backend happyhorse --video-resolution 1920x1080 --seed 0 \
    > "$LOG_DIR/ex01_happyhorse.log" 2>&1 &
HH_PID=$!
disown $HH_PID

echo "[ex01-qwen] $(date '+%H:%M:%S') wan_pid=$WAN_PID  happyhorse_pid=$HH_PID"
