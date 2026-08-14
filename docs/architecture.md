# Architecture

DSH is the conversation and presentation layer. The plugin exposes the
Director skill and sends lifecycle requests to the local Gateway.

The Gateway owns process isolation, queue bookkeeping, cancellation and
recovery. It never infers ReCA business stages from log text.

ReCA owns story planning, segment decomposition, provider calls, validation,
repair, retry, resume decisions, concatenation, and the artifact manifest.
ReCA writes `run/reca_state.json`, `run/audit.json`, and
`run/artifact_manifest.json`; the Gateway projects those files over HTTP.
