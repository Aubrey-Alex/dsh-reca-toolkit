# DSH ReCA Toolkit

Turn long-form stories into coherent, cinematic videos with consistent characters, scenes, and actions.

This repository packages ReCA as ReCA Director, a long-video creation Skill for DeepSeek Harness. DSH owns conversation and tool calls; ReCA remains the only owner of video planning, rendering, validation, repair, resume decisions, and artifact manifests.

## What's new in 0.4.0

- Optional first-frame and named reference-image inputs across DSH, Gateway,
  and the ReCA run configuration.
- GPT Image 2 as the default portrait, location, anchor, and repair-image
  backend, including local-reference support.
- Wan3.0 pure-R2V continuity routing compatible with the media combinations
  accepted by the deployed API.
- Bounded, retryable GPT Responses visual audits across concurrent Gateway
  processes.
- A static recorded-run Demo template built from redacted ReCA artifacts,
  without exposing provider APIs to visitors.

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

For the DSH conversation model, copy
[`configs/dsh-settings.example.yaml`](configs/dsh-settings.example.yaml) to
`$DSH_HOME/settings.yaml`, then export `RECA_DSH_DEEPSEEK_API_KEY` in the DSH
process environment. The example uses DSH's `llm-pi-ai` OpenAI-compatible route
because the team gateway supports `/v1/chat/completions`; ReCA's own planner
continues to use its separate Messages adapter.

The gateway can also be exercised without DSH:

```bash
bash scripts/run_demo.sh
```

The demo submits [`examples/sun_wukong_battle.txt`](examples/sun_wukong_battle.txt) by default, polls the task, and reports the final artifact URL. Set `RECA_DEMO_STORY=examples/story.txt` to use the shorter generic story. Generated files are stored under `.dsh_runs/`, which is ignored by git.

The Director request supports `story`, `duration`, `resolution`, `style`,
`aspect_ratio`, `backend`, `enable_audit`, and `seed`. It also accepts optional
`first_frame` and `reference_images` inputs. A first frame replaces the
automatically generated anchor for the first shot; reference images are passed
to anchor planning and are forwarded to segment rendering when the selected
video backend supports reference media. Wan3.0 preserves ReCA's planner and
serial segment chain while adapting only the provider mapping: I2V sends the
current frame as its sole reference; R2V sends the current frame as
`reference_image[0]`, followed by up to three planner-selected identity, scene,
or prop references. The R2V prefix explicitly asks Wan3.0 to begin from the
first reference. Bridges continue to use the provider's real first/last-frame
pair. This is a soft start constraint because Wan3.0 does not expose a hard
first-frame slot that can be combined with additional reference images. ReCA emits separate
Gateway, ReCA, video, and audit states. A generated video may legitimately
return `audit_skipped` or `audit_failed`.

## Recorded replay demo

The `demo/` directory is a static playback surface, not a public generation
endpoint. Build it from a completed real run:

```bash
python3 scripts/build_replay_manifest.py .dsh_runs/<run_id>
python3 scripts/build_demo_bundle.py .dsh_runs/<run_id>
python3 -m http.server 8080 --directory demo
```

The replay manifest is derived from the real request, planner, render plan,
audit, events, and artifact manifest. Generated media and run-specific replay
data remain ignored by Git; publish the resulting bundle through object storage
or a dedicated Demo deployment. `scripts/generate_first_frames.py` and
`scripts/monitor_batch.py` are optional helpers for preparing and monitoring a
curated multi-run Demo batch; provider credentials still come only from the
process environment.

For a direct ReCA run without DSH, use the bundled entry point:

```bash
python3 -m videorlm.framework._scripts._smoke \
  --story examples/story.txt --segments --render --backend wan \
  --video-resolution 1280x720
```

## Provider configuration

Only configuration is expected to change for the team-specific APIs. DeepSeek planner calls use the existing Claude Messages adapter, GPT internal visual validation uses the bundled Responses adapter, and the Wan 3.0 backend uses Alibaba Model Studio's async task API with temporary OSS staging for local references. The public source contains no provider credentials.

Never commit `.env`, API keys, generated videos, run-specific replay data, or
run logs.
