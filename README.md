# DSH ReCA Toolkit

Turn long-form stories into coherent, cinematic videos with consistent characters, scenes, and actions.

This repository bundles the complete ReCA source tree under [`videorlm/`](videorlm/) and adds a thin DeepSeek Harness integration around it. ReCA remains the execution engine: its planner, segment planning, media backends, validators, retry logic, resume behavior, and final concatenation are reused directly.

## Runtime shape

```text
DSH Web → dsh-reca plugin → ReCA Gateway → ReCA child process → final.mp4
```

The gateway exposes asynchronous task lifecycle endpoints:

```text
POST /v1/runs
GET  /v1/runs/{run_id}
GET  /v1/runs/{run_id}/events
POST /v1/runs/{run_id}/cancel
```

The DSH plugin registers `reca_start`, `reca_status`, and `reca_cancel`. It never receives provider credentials; ReCA reads them from the ignored local `.env` file or the process environment.

## Local setup

```bash
cp .env.example .env
# Optional in a fresh environment:
python3 -m pip install -r requirements.txt
# Fill the provider values in .env
python3 -m gateway.server
```

In another terminal, install the local plugin into the DSH web profile and start DSH:

```bash
dsh plugin --profile web add "$PWD/dsh-plugin"
dsh web
```

The gateway can also be exercised without DSH:

```bash
bash scripts/run_demo.sh
```

The demo submits [`examples/sun_wukong_battle.txt`](examples/sun_wukong_battle.txt) by default, polls the task, and reports the final artifact URL. Set `RECA_DEMO_STORY=examples/story.txt` to use the shorter generic story. Generated files are stored under `.dsh_runs/`, which is ignored by git.

The default run enables the GPT visual anchor validator. ReCA's optional
segment-level validator can be enabled per DSH call with `validate_segments`.

For a direct ReCA run without DSH, use the bundled entry point:

```bash
python3 -m videorlm.framework._scripts._smoke \
  --story examples/story.txt --segments --render --backend wan \
  --video-resolution 1280x720
```

## Provider configuration

Only configuration is expected to change for the team-specific APIs. DeepSeek planner calls use the existing Claude Messages adapter, GPT internal visual validation uses the bundled Responses adapter, and the Wan 3.0 backend uses Alibaba Model Studio's async task API with temporary OSS staging for local references. The public source contains no provider credentials.

Never commit `.env`, API keys, generated videos, or run logs.
