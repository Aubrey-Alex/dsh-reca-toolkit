# Run Lifecycle

Gateway states are `queued`, `running`, `interrupted`, `cancelling`,
`cancelled`, `failed`, and `succeeded`.

ReCA stages are independent: `planning`, `asset_generation`, `rendering`,
`validating`, `repairing`, `concat`, `succeeded`, and `failed`.

Audit state is reported separately and can be `audit_pending`,
`audit_running`, `audit_retrying`, `audit_failed`, `audit_skipped`,
`audit_repaired`, or `audited`. A successful video with `audit_failed` is a
valid, diagnosable outcome and must not be presented as audited.

On Gateway restart, active Gateway states are marked `interrupted`. Provider
jobs are never submitted automatically during recovery. `reca_resume` is the
explicit action that lets ReCA inspect its persisted run directory and resume.
