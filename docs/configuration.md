# Configuration

Copy `.env.example` to `.env`. Keep all keys in that ignored file; they are
never accepted as tool arguments and are removed from Gateway logs.

The product-level request fields are `duration`, `resolution`, `style`,
`aspect_ratio`, `backend`, `enable_audit`, `validate_segments`, and `seed`.
The Gateway normalizes these into `run_config.json` and appends only the
constraints to the existing ReCA planner input. It does not create shots.

Run `bash scripts/doctor.sh` before starting the Gateway. The doctor reports
only whether a key is present, never its value.

## DSH conversation provider

Copy `configs/dsh-settings.example.yaml` to `$DSH_HOME/settings.yaml` and set
`RECA_DSH_DEEPSEEK_API_KEY` only in the DSH process environment. The profile
selects `reca-deepseek` through `llm-pi-ai` and keeps the key out of settings,
tool arguments, model context, and logs. This is the conversation provider;
ReCA's planner provider remains configured independently by `RECA_PLANNER_*`.
