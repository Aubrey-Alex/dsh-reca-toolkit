#!/bin/bash
# Batch launcher for ex62-ex68 (7 new examples in true_input/).
#
# Planner: qwen3.6-max-preview via DashScope /compatible-mode/v1/chat
# Image  : qwen-image-2.0-pro with explicit moderation-fallback chain
#          → wan2.7-image-pro → gpt-image-2  (NEW: fail-loud-or-walk chain
#          instead of legacy silent qwen swap; lives in qwen.py::render).
# Video  : wan2.7-r2v (1280x720) + happyhorse-1.0-r2v (1920x1080) per branch.
#
# Concurrency:
#   RECA_BACKEND_CONCURRENCY_QWEN_IMAGE_2_0_PRO=4  (TPS-throttle safe; qwen
#     image hits dashscope rate limit past 4 concurrent in-flight).
#   wan2.7-r2v / happyhorse-1.0-r2v via the usual xsem caps (12 / 18).
#
# Usage:
#   bash videorlm/scripts/run_batch_62_68.sh [first] [last]
# Defaults: first=62, last=68.

set -u
cd /mnt/workspace/akide/code/unirlm-02

set -a; source .env; set +a
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export RECA_DISABLE_HTTP_PROXY=1
export RECA_XSEM_ENABLE=1
export RECA_PLAN_SKELETON_MAX_RETRIES=5
export RECA_HAPPYHORSE_POLL_MAX_WAIT_S=3600
export RECA_VIDEO_CALL_TIMEOUT_S=3700
export RECA_XSEM_HAPPYHORSE_R2V=8

# Planner = qwen3.6-max-preview on DashScope (no Cloudflare gateway streaming
# truncation issues — gpt-5.5-high /v1/messages is too flaky for 24-shot output).
export RECA_PLANNER_API_KEY="$DASHSCOPE_API_KEY"
export RECA_PLANNER_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export RECA_PLANNER_MODEL=qwen3.6-max-preview
export RECA_PLANNER_API_PATH=chat
export RECA_QWEN_THINKING=1

# Image = qwen-image-2.0-pro with explicit fallback chain.
export RECA_RENDER_BACKEND_PORTRAIT=qwen-image-2.0-pro
export RECA_RENDER_BACKEND_ANCHOR_IMAGE=qwen-image-2.0-pro
export RECA_RENDER_BACKEND_IMAGE_EDIT=qwen-image-2.0-pro
export RECA_IMAGE_FALLBACK_CHAIN="wan2.7-image-pro,gpt-image-2"
export RECA_BACKEND_CONCURRENCY_QWEN_IMAGE_2_0_PRO=4   # TPS-safe
export ZIPLAB_API_KEY="$OPENAI_API_KEY"                # for gpt-image-2 fallback
export RECA_IMAGE_BACKEND_TAG="qwen-image-2.0-pro+fallback-chain+qwen3.6-planner"

FIRST=${1:-62}
LAST=${2:-68}

LOG_DIR=videorlm/outputs/version_2.4/_batch_logs
mkdir -p "$LOG_DIR"

for i in $(seq -f "%02g" "$FIRST" "$LAST"); do
    STORY="input_examples/true_input/example_${i}.txt"
    if [ ! -f "$STORY" ]; then
        echo "[skip] ex${i}: $STORY missing"; continue
    fi

    # wan branch
    W_OUT="videorlm/outputs/version_2.4/ex${i}_wan"
    W_LOG="$LOG_DIR/ex${i}_wan.log"
    if [ -f "$W_OUT/run/final.mp4" ]; then
        echo "[skip] ex${i}_wan: final.mp4 exists"
    else
        mkdir -p "$W_OUT"
        nohup python3 -u -m videorlm.framework._scripts._smoke \
            --story "$STORY" --out-dir "$W_OUT" --label "ex${i}_wan" \
            --segments --render --resume \
            --backend wan --video-resolution 1280x720 --seed 0 \
            > "$W_LOG" 2>&1 &
        disown $!
        echo "  ex${i}_wan pid=$!"
    fi

    # happyhorse branch
    H_OUT="videorlm/outputs/version_2.4/ex${i}_happyhorse"
    H_LOG="$LOG_DIR/ex${i}_happyhorse.log"
    if [ -f "$H_OUT/run/final.mp4" ]; then
        echo "[skip] ex${i}_happyhorse: final.mp4 exists"
    else
        mkdir -p "$H_OUT"
        nohup python3 -u -m videorlm.framework._scripts._smoke \
            --story "$STORY" --out-dir "$H_OUT" --label "ex${i}_hh" \
            --segments --render --resume \
            --backend happyhorse --video-resolution 1920x1080 --seed 0 \
            > "$H_LOG" 2>&1 &
        disown $!
        echo "  ex${i}_happyhorse pid=$!"
    fi
done

echo "batch launched at $(date '+%H:%M:%S')"
echo "tail -f $LOG_DIR/ex${FIRST}_wan.log to watch"
