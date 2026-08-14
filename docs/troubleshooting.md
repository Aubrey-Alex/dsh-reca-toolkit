# Troubleshooting

If DSH cannot see the tools, check that the Gateway is running and reinstall
the local plugin with `bash scripts/install_dsh_plugin.sh`.

If a run is `interrupted`, inspect its status and call `reca_resume`; do not
submit a second run manually because Wan tasks may already be paid jobs.

If `audit_state` is `audit_failed`, the video and audit report remain separate
artifacts. Inspect `run/audit.json` and `run/run_report.json` before deciding
whether to rerun with audit disabled or repair the provider configuration.
