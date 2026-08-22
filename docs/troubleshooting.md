# Troubleshooting

If DSH cannot see the tools, add the plugin once with
`dsh plugin --profile web add "file:$PWD/dsh-plugin"` and reopen DSH Web.
The plugin starts this repository's runtime by itself. Do not ask the user to
launch a Gateway.

If a run is `interrupted`, inspect its status and call `reca_resume`; do not
submit a second run manually because Wan tasks may already be paid jobs.

If `audit_state` is `audit_failed`, the video and audit report remain separate
artifacts. Inspect `run/audit.json` and `run/run_report.json` before deciding
whether to rerun with audit disabled or repair the provider configuration.
