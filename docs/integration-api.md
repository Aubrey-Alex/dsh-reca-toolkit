# API Integration Contract

The repository already contains the complete ReCA engine and the DSH task
lifecycle. The remaining provider work is intentionally isolated to API
configuration or a media backend adapter.

## DeepSeek planner

Provide:

- `RECA_PLANNER_API_KEY`
- `RECA_PLANNER_BASE_URL`
- `RECA_PLANNER_MODEL`
- whether the endpoint speaks OpenAI-compatible `chat.completions`

The existing ReCA planner already supports an OpenAI-compatible base URL.

## GPT vision validator

Provide:

- API key and the full Responses endpoint URL
- vision model id
- accepted image input form (`input_image` with `image_url`, data URL, or uploaded file)
- structured JSON response support, if available

The repository now includes `OpenAIResponsesAgent` for this protocol. The
existing validator still owns scoring, retry decisions, and render repair;
only the model client route changes.

## Wan 3.0 media backend

The working Alibaba Model Studio contract is already wired:

- base URL: `https://dashscope.aliyuncs.com/api/v1`
- submit: `POST /services/aigc/video-generation/video-synthesis`
- poll: `GET /tasks/{task_id}`
- local media staging: `GET /uploads?action=getPolicy&model=wan3.0-video`,
  followed by the returned temporary OSS multipart upload
- async headers: `X-DashScope-Async: enable` and, for `oss://` media,
  `X-DashScope-OssResourceResolve: enable`

The adapter implements ReCA's existing segment backend contract:

```text
render_segment(request) -> local mp4 path
render_bridge(request)  -> local mp4 path (when supported)
```

No planner, validator, retry, resume, or concatenation logic belongs in this
adapter. Task ids are persisted beside each output video so a process restart
polls the existing paid task instead of submitting it again.
