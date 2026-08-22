# Architecture

DSH is the conversation and presentation layer. Users add the plugin once,
open DSH Web, and describe a film — including long-form duration. The plugin
starts this repository's runtime if needed.

The runtime (Gateway) is an internal process: isolation, queue bookkeeping,
cancellation, and recovery. It is not a second product. It never infers ReCA
business stages from log text.

ReCA owns story planning, segment decomposition, provider calls, validation,
repair, retry, resume decisions, concatenation, and the artifact manifest.
ReCA writes `run/reca_state.json`, `run/audit.json`, and
`run/artifact_manifest.json`; the Gateway projects those files over HTTP.
