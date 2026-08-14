# Security

Never commit `.env`, provider API keys, signed provider URLs, or generated run
directories. Copy `.env.example` to `.env` and keep credentials in the local
environment. DSH tool arguments and Gateway request files intentionally contain
only user configuration, never provider credentials.

Runtime logs are redacted for bearer tokens, API keys, and signed OSS query
parameters. Rotate any credential that may have appeared in a log.
