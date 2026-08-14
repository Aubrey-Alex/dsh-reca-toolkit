#!/bin/bash
# Batch launcher for ex45-ex61 (17 new examples in true_input/).
#
# Per-example launches both wan + happyhorse pipelines simultaneously,
# each in its own version_2.4/ex{NN}_{wan,happyhorse}/ out_dir.
#
# Planner: gpt-5.5-high via /v1/messages (Anthropic Messages API path on
#   the OpenAI-compatible gateway pointed at by OPENAI_BASE_URL).
# Image  : gpt-image-2 (~5s/image).
# Video  : wan2.7-r2v (1280x720) + happyhorse-1.0-r2v (1920x1080).
#
# Concurrency control: cross-process semaphore (xsem) caps:
#   wan2.7-r2v:        12 slots
#   happyhorse-1.0-r2v: 18 slots
#   gpt-image-2:       15 slots
# All 34 pipelines compete for these caps — xsem queues so no oversubscription.
#
# Usage:
#   bash videorlm/scripts/run_batch_45_61.sh [first] [last]
# Defaults: first=45, last=61.
#
# Args:
#   $1 — first example index (default 45)
#   $2 — last example index  (default 61)

set -u
cd /mnt/workspace/akide/code/unirlm-02

set -a; source .env; set +a
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export RECA_DISABLE_HTTP_PROXY=1
export RECA_XSEM_ENABLE=1
export RECA_PLAN_SKELETON_MAX_RETRIES=5
export RECA_PLANNER_API_KEY="$OPENAI_API_KEY"
export RECA_PLANNER_BASE_URL="$OPENAI_BASE_URL"
export RECA_PLANNER_MODEL=gpt-5.5-high
export RECA_PLANNER_API_PATH=messages
export RECA_PLANNER_THINKING_BUDGET=16000
export RECA_RENDER_BACKEND_PORTRAIT=gpt-image-2
export RECA_RENDER_BACKEND_ANCHOR_IMAGE=gpt-image-2
export RECA_RENDER_BACKEND_IMAGE_EDIT=gpt-image-2
export RECA_IMAGE_BACKEND_TAG="gpt-image-2+gpt-5.5-high-messages-planner"

FIRST=${1:-45}
LAST=${2:-61}

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
echo "tail -f $LOG_DIR/ex45_wan.log to watch"
