# ReCA Integration Patches

The `videorlm/` directory is a vendored ReCA snapshot. Product integration
changes are kept explicit:

- `backends/media/impl/dashscope/video/wan30.py`: Wan 3.0 async backend and local media staging.
- `backends/llm/agents/openai_responses/`: OpenAI Responses adapter for GPT Vision.
- `framework/pipeline.py`: minimal local-tail compatibility and lifecycle callback hooks.
- `framework/_scripts/_smoke.py`: provider configuration and Director RunConfig ingress.
- `integrations/director/`: ReCA-owned state and artifact manifest output.
- `gateway/` and `dsh-plugin/`: DSH transport, lifecycle projection, and tools.
