# Tools

The plugin registers these DSH tools:

- `reca_create_video`: submit a natural-language story and RunConfig.
- `reca_get_status`: return Gateway state plus ReCA stage, video state, audit state, and manifest.
- `reca_cancel`: request SIGTERM, then SIGKILL after the configured grace period.
- `reca_resume`: resume an interrupted, cancelled, or failed run from its run directory.
- `reca_list_runs`: list persisted runs.
- `reca_get_artifact`: return the manifest or a complete artifact URL.

`reca_start` and `reca_status` remain compatibility aliases.
