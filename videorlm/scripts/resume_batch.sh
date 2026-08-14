#!/bin/bash
# Resume a list of cells that don't have final.mp4.
# Launches them with --resume in batches of $BATCH_SIZE (default 4).
#
# Usage: bash resume_batch.sh <cell1> <cell2> ...
# Each <cell> is of form ex{NN}_{wan,happyhorse}.

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

LOG_DIR=videorlm/outputs/version_2.4/_batch_logs
mkdir -p "$LOG_DIR"

for cell in "$@"; do
    # parse cell: ex{NN}_{backend}
    i=$(echo "$cell" | grep -oE "[0-9]+")
    backend=$(echo "$cell" | sed "s|ex${i}_||")
    case "$backend" in
        wan) res="1280x720";;
        happyhorse) res="1920x1080";;
        *) echo "[skip] unknown backend in $cell"; continue;;
    esac
    STORY="input_examples/true_input/example_${i}.txt"
    OUT="videorlm/outputs/version_2.4/${cell}"
    LOG="$LOG_DIR/${cell}.log"
    if [ -f "$OUT/run/final.mp4" ]; then
        echo "[skip] $cell: final.mp4 already exists"
        continue
    fi
    nohup python3 -u -m videorlm.framework._scripts._smoke \
        --story "$STORY" --out-dir "$OUT" --label "$cell" \
        --segments --render --resume \
        --backend "$backend" --video-resolution "$res" --seed 0 \
        > "$LOG" 2>&1 &
    pid=$!; disown $pid
    echo "  $cell pid=$pid"
done
echo "launched at $(date '+%H:%M:%S')"
