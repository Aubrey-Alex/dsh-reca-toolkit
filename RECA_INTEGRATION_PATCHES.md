# ReCA Integration Patches

The `videorlm/` directory is a vendored ReCA snapshot. Product integration
changes are kept explicit:

- `backends/media/impl/dashscope/video/wan30.py`: Wan 3.0 async backend,
  local-media staging, persisted provider task ids, and the pure-R2V mapping
  used when the current frame and planner references must be sent together.
- `backends/media/impl/openai/image/gpt_image_2.py`: GPT Image 2 generation
  and editing with JSON or streaming responses and local reference files.
- `backends/llm/agents/openai_responses/`: OpenAI Responses adapter for GPT
  Vision, including cross-process request serialization, bounded retries, and
  compatible output parsing.
- `framework/pipeline.py`: Director input assets, preloaded/protected first
  anchors, serial image-edit dependencies, reference propagation, local-tail
  compatibility, repair cache invalidation, audit state preservation, and
  lifecycle callback hooks.
- `framework/_scripts/_smoke.py`: provider configuration, Director RunConfig
  ingress, and input-manifest application without moving planning into DSH.
- `framework/validator/anchor/validate.py`: optional compact GPT Vision audit
  context for OpenAI-compatible gateways.
- `integrations/director/`: ReCA-owned state and artifact manifest output.
- `gateway/`: DSH transport, lifecycle projection, isolated input staging, and
  artifact publication.
- `dsh-plugin/`: ReCA Director Skill plus create/status/cancel/resume/list and
  artifact tools.
