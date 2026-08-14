# Configuration

Copy `.env.example` to `.env`. Keep all keys in that ignored file; they are
never accepted as tool arguments and are removed from Gateway logs.

The product-level request fields are `duration`, `resolution`, `style`,
`aspect_ratio`, `backend`, `enable_audit`, `validate_segments`, and `seed`.
The Gateway normalizes these into `run_config.json` and appends only the
constraints to the existing ReCA planner input. It does not create shots.

Run `bash scripts/doctor.sh` before starting the Gateway. The doctor reports
only whether a key is present, never its value.
