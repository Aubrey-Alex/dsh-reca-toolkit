# Recorded ReCA Director Demo

This directory contains the static UI for replaying a completed, real DSH +
ReCA Director run. It does not call the Gateway or any model provider.

Build a local bundle from a completed run:

```bash
python3 scripts/build_replay_manifest.py .dsh_runs/<run_id>
python3 scripts/build_demo_bundle.py .dsh_runs/<run_id>
python3 -m http.server 8080 --directory demo
```

Open `http://127.0.0.1:8080`. The generated manifest and media under
`demo/data/` and `demo/assets/` are intentionally ignored by Git. Publish them
through the Demo deployment rather than the source repository.
