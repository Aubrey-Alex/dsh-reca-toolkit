# DSH ReCA Toolkit

Turn long-form stories into coherent, cinematic videos with consistent characters, scenes, and actions.

This repository packages ReCA as ReCA Director, a long-video creation Skill for DeepSeek Harness. DSH owns conversation and tool calls; ReCA remains the only owner of video planning, rendering, validation, repair, resume decisions, and artifact manifests.

## Runtime shape

```text
DSH Web → ReCA Director Skill → ReCA Gateway → ReCA child process → final.mp4 + audit + manifest
```

The gateway exposes asynchronous task lifecycle endpoints:

```text
POST /v1/runs
GET  /v1/runs/{run_id}
GET  /v1/runs/{run_id}/events
POST /v1/runs/{run_id}/cancel
POST /v1/runs/{run_id}/resume
GET  /v1/runs
GET  /v1/runs/{run_id}/artifacts
```

The DSH plugin registers `reca_create_video`, `reca_get_status`, `reca_cancel`, `reca_resume`, `reca_list_runs`, and `reca_get_artifact`. Compatibility aliases `reca_start` and `reca_status` remain available. The plugin never receives provider credentials; ReCA reads them from the ignored local `.env` file or the process environment.

## Local setup

```bash
bash scripts/install.sh
# Fill the provider values in .env
bash scripts/doctor.sh
bash scripts/start-gateway.sh
```

In another terminal, install the local plugin into the DSH web profile and start DSH:

```bash
dsh plugin --profile web add "file:$PWD/dsh-plugin"
dsh web
```

The gateway can also be exercised without DSH:

```bash
bash scripts/run_demo.sh
```

The demo submits [`examples/sun_wukong_battle.txt`](examples/sun_wukong_battle.txt) by default, polls the task, and reports the final artifact URL. Set `RECA_DEMO_STORY=examples/story.txt` to use the shorter generic story. Generated files are stored under `.dsh_runs/`, which is ignored by git.

The Director request supports `story`, `duration`, `resolution`, `style`,
`aspect_ratio`, `backend`, `enable_audit`, and `seed`. ReCA emits separate
Gateway, ReCA, video, and audit states. A generated video may legitimately
return `audit_skipped` or `audit_failed`.

For a direct ReCA run without DSH, use the bundled entry point:

```bash
python3 -m videorlm.framework._scripts._smoke \
  --story examples/story.txt --segments --render --backend wan \
  --video-resolution 1280x720
```

## Provider configuration

Only configuration is expected to change for the team-specific APIs. DeepSeek planner calls use the existing Claude Messages adapter, GPT internal visual validation uses the bundled Responses adapter, and the Wan 3.0 backend uses Alibaba Model Studio's async task API with temporary OSS staging for local references. The public source contains no provider credentials.

Never commit `.env`, API keys, generated videos, or run logs.
