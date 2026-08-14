# backends — 后端层

`videorlm/backends/` 是与外部 API 的**唯一**接触面。上层只构造 Request 调 `dispatch_*`,
具体走哪个模型由 env 决定。

两条硬规则:

1. **一次 dispatch 只用一个后端**。失败就在同一个后端上重试到预算耗尽,然后抛错。
   换后端是上层的事,不在这一层发生。
   (例外只有一处:`qwen-image-2.0*` 的内容审核 fallback,见 §7.3。)
2. **所有 DashScope SDK 调用必须走 `platforms.with_key("dashscope", ...)`**。绕过它直接调
   `dashscope.*` 会破坏多 key 负载均衡 / cooldown / EWMA 统计。

```
backends/
├── media/
│   ├── interface/     纯抽象,零 SDK 依赖
│   │   ├── requests.py         SegmentRequest / BridgeRequest / ImageRequest / VideoResult / ImageResult / ImageRef
│   │   ├── dispatch.py         dispatch_segment / dispatch_bridge / dispatch_image
│   │   ├── registry.py         for_kind / get_backend / list_backends / register_backend / DEFAULT_RENDER_BACKEND
│   │   ├── segment_backend.py  ProviderSpec / VideoSegmentBackend(Protocol) / VideoSegmentBackendBase / auto_register
│   │   ├── capabilities.py     BackendCapabilities / RenderPlanError / BackendRenderError
│   │   └── base.py             VideoBackend ABC
│   └── impl/
│       ├── dashscope/{image,video}/   wan / qwen-image / happyhorse
│       ├── openai/image/              gpt-image-2 系
│       └── local/ltx_v2_3/            ltx-2.3(本地扩散)
├── llm/                见 agents.md
└── _common/
    ├── providers/      Provider 抽象 + dashscope / openai / kling / pixverse / vidu
    ├── platforms.py    with_key 中间件
    ├── key_pool.py     多 key 评分轮转 + 分级 cooldown + EWMA
    ├── rate_limiter.py per-model 提交节流(interval + max_parallel)
    ├── concurrency.py  NamedSemaphorePool
    ├── retry.py        RetryPolicy / retry_until_exhausted
    ├── timeout.py      call_with_timeout / BackendCallTimeout
    ├── trace.py        trace_event → log_dir/trace.jsonl
    ├── errors.py       BackendError / StructuralBackendError / TransientBackendError
    ├── env.py          env_value(.env 兜底)
    ├── registry.py     BackendRegistry 容器
    ├── capabilities.py 通用 capabilities 基类
    └── oss_publisher.py 本地文件 / bytes → OSS 公网 URL
```

---

## 1. 三个 Request

```python
from videorlm.backends.media import (
    SegmentRequest, BridgeRequest, ImageRequest, ImageRef,
    VideoResult, ImageResult,
    dispatch_segment, dispatch_bridge, dispatch_image,
    for_kind, get_backend, list_backends, register_backend,
)
```

没有别的请求形状。

### SegmentRequest —— 首帧锚定 + 可选参考图 → mp4

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `request_id` | str | 必填 | trace / 日志关联 |
| `prompt` | str | 必填 | 空串抛 `ValueError` |
| `first_url` | str | 必填 | 空串抛 `ValueError` |
| `mode` | `"i2v" \| "r2v"` | 必填 | **由上层显式决定**,后端不按 `len(refs)` 自己推 |
| `reference_image_urls` | `tuple[str,...]` | `()` | 只在 `mode=="r2v"` 生效 |
| `duration_s` | float | 5.0 | ≤0 抛 `ValueError` |
| `seed` | int | 0 | |
| `output_path` | str | `""` | 本地落地路径 |
| `log_dir` | str \| None | None | trace 写入目录 |
| `negative_prompt` | str | `""` | 非空时覆盖后端默认 |
| `resolution` | str | `"1920x1080"` | |
| `prompt_extend` | bool | False | 仅 wan 系有效 |

### BridgeRequest —— 首帧 + 末帧 → 转场 mp4

字段同上,去掉 `mode` / `reference_image_urls`,增加 `last_url`(必填,空串抛 `ValueError`)。

### ImageRequest —— 一张图

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `request_id` | str | 必填 | |
| `kind` | `portrait \| anchor_image \| location \| prop \| image_edit` | 必填 | 决定路由 |
| `prompt` | str | 必填 | 空串抛 `ValueError` |
| `references` | `tuple[ImageRef,...]` | `()` | `ImageRef(role, url)`,`role ∈ {portrait, scene, reference, source}`,url 为空抛 `ValueError` |
| `seed` / `output_path` / `log_dir` / `negative_prompt` | | | 同上 |
| `resolution` | str | `"1280x720"` | |

### 返回

`VideoResult(request_id, success, output_url, output_path, rendered_duration_s, seed_used,
backend_name, error, cost_usd)`;`ImageResult` 少 `rendered_duration_s`。
两者都是**可变** dataclass —— dispatch 会回填 `backend_name` 和 `cost_usd`。

`output_url` 是公网 URL(可直接喂给下游 API),`output_path` 是本地路径。
有的视频后端只落本地不回 URL,上层要能同时处理这两种。

---

## 2. 三个 dispatch

```python
res = dispatch_segment(SegmentRequest(..., mode="r2v"))
res = dispatch_bridge(BridgeRequest(...))
res = dispatch_image(ImageRequest(..., kind="anchor_image"))
```

**后端解析顺序**:

| dispatch | 顺序 |
|---|---|
| `dispatch_segment` | `RECA_RENDER_BACKEND_SEGMENT_<MODE>` → `RECA_RENDER_BACKEND_SEGMENT` → `for_kind("segment_<mode>")` |
| `dispatch_bridge` | `RECA_RENDER_BACKEND_BRIDGE` → `for_kind("bridge")` |
| `dispatch_image` | `RECA_RENDER_BACKEND_<KIND>` → `for_kind(kind)` |
| `for_kind(kind)` | `RECA_RENDER_BACKEND_<KIND>` → `DEFAULT_RENDER_BACKEND[kind]` |

解析到的对象若没有对应方法(`render_segment` / `render_bridge` / `render`)直接抛 `RenderPlanError`。

**统一包装**(`_run_single_backend`),三个入口共用:

1. 取 per-backend 信号量,容量 = `capabilities().max_concurrency`,
   可被 `RECA_BACKEND_CONCURRENCY_<NAME>` 覆盖(NAME 大写,`-` 和 `.` 换成 `_`,
   如 `RECA_BACKEND_CONCURRENCY_WAN2_7_R2V`)
2. `retry_until_exhausted` 包一层:`max_attempts = caps.max_retries + 1`,
   指数退避 `caps.retry_backoff_base_s` → `caps.retry_backoff_max_s`
3. 每次尝试外面再套 wall-clock 超时 `RECA_VIDEO_CALL_TIMEOUT_S`(默认 **3700s**,
   必须大于后端内部最长轮询预算)
4. 成功后回填 `backend_name`;`VideoResult.cost_usd == 0` 时按
   `cost_per_call + cost_per_second × 时长` 估算填上
5. 全程写 trace:`video.dispatch_segment` / `video.dispatch_bridge` / `video.dispatch_image`
   的 `start` / `attempt_start` / `success` / `error`

**什么算可重试**:异常类型在 `caps.retryable_error_names` 里,或异常文本命中
`ratelimit / rate limit / throttling / throttled / datainspectionfailed / content filter /
concurrentlimit / timeout / temporarily unavailable / 503 / 502 / 429 / ratequota`。
其余立即抛出,不浪费预算。

---

## 3. 默认路由表

`media/interface/registry.py`:

```python
DEFAULT_RENDER_BACKEND = {
    "anchor_image":  "wan2.7-image",
    "portrait":      "wan2.7-image",
    "location":      "wan2.7-image",
    "prop":          "wan2.7-image",
    "image_edit":    "wan2.7-image",
    "segment_i2v":   "happyhorse-1.0-i2v",
    "segment_r2v":   "wan2.7-r2v",
    "bridge":        "wan2.7-i2v",
}
```

只有两层:env 覆盖 → 这张表。没有 JSON 配置文件,没有目录级覆盖。

> `location` / `prop` 两行只在有人直接用这两个 `kind` 调 `dispatch_image` 时才生效。
> framework 的图像 DAG 把 location / prop 都按 `anchor_image` 派发,
> 见 [framework.md §4.1](framework.md)。

首次 `get_backend` / `for_kind` 未命中时会惰性 import `media/impl/` 触发注册;
`openai` 和 `local` 两个子树 import 失败只打日志不阻塞(依赖可能没装)。

---

## 4. 已注册后端

### 4.1 视频(实现 `render_segment` + `render_bridge`)

| 名称 | provider | 分辨率 | segment 时长 | bridge 时长 | max refs | 并发 env | 默认并发 | $/call + $/s |
|---|---|---|---|---|---|---|---|---|
| `happyhorse-1.0-i2v` | dashscope | 720/1080 | 3–15 | 3–15 | 1 | `RECA_HAPPYHORSE_I2V_WORKERS` | 8,按 key 数 ×8 | 0.08 + 0.008 |
| `happyhorse-1.0-r2v` | dashscope | 720/1080 | 3–15 | 3–15 | 9 | `RECA_HAPPYHORSE_R2V_WORKERS` | 24,按 key 数 ×8 | 0.08 + 0.008 |
| `wan2.7-i2v` | dashscope | 720 | 3–15 | 3–8 | 2 | `RECA_WAN27_I2V_WORKERS` | 8,按 key 数 ×8 | 0.10 + 0.012 |
| `wan2.7-r2v` | dashscope | 720 | 3–15 | 3–8 | 4 | `RECA_WAN27_R2V_WORKERS` | 16,按 key 数 ×8 | 0.12 + 0.015 |
| `ltx-2.3` | local | 720/1080 | 1–25 | 2–8 | 4 | `LTX23_WORKERS` | 1 | 0 |

并发的解析顺序(`ProviderSpec.resolved_concurrency`):
`concurrency_env` 有值 → `per_key_concurrency × 该 provider 当前 key 数` → `default_concurrency`。

**mode 支持矩阵**:

| 后端 | `render_segment(mode="i2v")` | `render_segment(mode="r2v")` | `render_bridge` |
|---|---|---|---|
| `happyhorse-1.0-i2v` | ✅ | ❌ 抛 `BackendRenderError` | ✅ 忽略 `last_url`,写 `video.degrade` trace |
| `happyhorse-1.0-r2v` | ✅ 忽略 refs | ✅ 取前 8 张 | ✅ 把 `last_url` 当软目标 |
| `wan2.7-r2v` | ✅ 忽略 refs | ✅ 取前 4 张 | ✅ 把 `last_url` 当软目标 |
| `wan2.7-i2v` | ⚠️ 默认抛错 | ❌ 抛 `BackendRenderError` | ✅ 原生首尾双锚 |
| `ltx-2.3` | ✅ | ✅ | ✅ 关键帧插值 |

> `wan2.7-i2v` 是首+尾双锚端点,**纯 i2v 段默认拒绝**;要强行用它渲 segment 得设
> `RECA_WAN27_I2V_SEGMENT_REUSE_FIRST_AS_LAST=1`,它会把 `first_url` 同时当末帧,
> 产出接近静止的片段。所以 `segment_i2v` 不要指向 `wan2.7-i2v`。

happyhorse 系的额外旋钮:轮询间隔 `RECA_HAPPYHORSE_POLL_INTERVAL_S`(默认 5s)、
轮询上限 `RECA_HAPPYHORSE_POLL_MAX_WAIT_S`(默认 3600s)、宽高比透传
`RECA_HAPPYHORSE_R2V_RATIO` / `RECA_HAPPYHORSE_I2V_RATIO`(非空才写进 API 的 `ratio` 参数)。

### 4.2 图像(实现 `render(ImageRequest)`)

| 名称 | provider | kinds | T2I | I2I | max refs | 分辨率 | 并发 env(默认) | $/call |
|---|---|---|---|---|---|---|---|---|
| `wan2.7-image` | dashscope | anchor_image / portrait / image_edit | ✅ | ✅ | 9 | 1280x720, 1024x1024 | `RECA_WAN27_IMAGE_WORKERS`(8) | 0.008 |
| `wan2.7-image-pro` | dashscope | 同上 | ✅ | ✅ | 9 | 同上 | `RECA_WAN27_IMAGE_PRO_WORKERS`(8) | 0.012 |
| `wan2.6-image` | dashscope | 同上 | ❌ | ✅ | 4 | 1280x720, 1024x1024 | `RECA_WAN26_IMAGE_WORKERS`(8) | 0.005 |
| `qwen-image-2.0-pro` | dashscope | 同上 | ✅ | ✅ | 3 | 1024²/1280x720/1280²/1920x1080/2048²/1536x2688/1728x2368/2368x1728/2688x1536 | `RECA_QWEN_IMAGE_WORKERS`(8) | 0.012 |
| `qwen-image-2.0` | dashscope | 同上 | ✅ | ✅ | 3 | 同上 | 同上 | 0.012 |
| `gpt-image-2` | openai | 同上 | ✅ | ✅ | 16(单次实际取前 `RECA_GPT_IMAGE_2_REF_CAP`=4) | 1024²/1536x1024/1024x1536/2048²/3840x2160/1280x720/1024x640 | `RECA_GPT_IMAGE_2_WORKERS`(15) | 0.04 |
| `gpt-image-2-pro` | openai | 同上 | ✅ | ✅ | 同上 | 同上 | 同上 | 0.06 |

`gpt-image-2` 系返回 `b64_json`,后端解码落盘后自动传 OSS 换公网 URL;OSS 没配时
`output_url` 为 `None`。它自带更宽的重试预算:`RECA_GPT_IMAGE_2_MAX_RETRIES`(默认 16)、
退避上限 `RECA_GPT_IMAGE_2_BACKOFF_MAX_S`(默认 300s)。

`qwen-image-2.0` 与 `-pro` 能力完全一致,只是注册名不同(上游对两者的 QPS 配额不同,
按需用 `RECA_RENDER_BACKEND_*` 切)。

### 4.3 ltx-2.3(本地)

唯一的非 API 后端,跑本地扩散权重,`provider="local"`,不进 KeyPool。
权重根目录由 `LTX23_ROOT` 指定,并发 `LTX23_WORKERS`(默认 1)。
segment 走 `ti2vid_two_stages`,bridge 走关键帧插值;输出按 64 对齐的高度生成
(704 / 1088),再用 ffmpeg pad/crop 回请求的精确尺寸,保证与其它后端能拼。

---

## 5. 加一个新后端

视频后端只需实现 2 个方法 + 1 个 spec:

```python
from videorlm.backends.media.interface.segment_backend import (
    ProviderSpec, VideoSegmentBackendBase, auto_register,
)

class MyBackend(VideoSegmentBackendBase):
    PROVIDER_SPEC = ProviderSpec(
        name="myvendor-1.0-i2v",
        family="myvendor",
        supports_resolutions=("1280x720",),
        segment_duration_range=(3.0, 15.0),
        bridge_duration_range=(3.0, 8.0),
        max_reference_images=4,
        concurrency_env="RECA_MYVENDOR_WORKERS",
        default_concurrency=8,
        per_key_concurrency=8,          # >0 时按 key 数自动扩容
        provider="myvendor",
        supports_i2v=True, supports_r2v=True,
        cost_per_call_usd=0.0, cost_per_second_usd=0.0,
    )
    def render_segment(self, req): ...
    def render_bridge(self, req): ...

auto_register(MyBackend())
```

`VideoSegmentBackendBase` 会从 `PROVIDER_SPEC` 自动生成 `capabilities()`。
再在 `media/impl/<vendor>/__init__.py` 里 import 该模块,并在
`registry._autoload_default_backends()` 加一行 import 即可。
图像后端不走这个 Protocol —— 只要有 `NAME` / `capabilities()` / `render(ImageRequest)`
三样,然后 `register_backend(NAME, instance)`。

新后端要落地产物到公网,统一调 `_common/oss_publisher.upload_file`,不要自己 `import oss2`。

---

## 6. BackendCapabilities

`capabilities()` 返回的字段(dispatch 在调用前就读它做计划期判断):

- 标识:`backend_name / model_family / model_id / provider / supports_kinds`
- 能力位:`supports_t2v / i2v / r2v / first_image / last_image / mid_reference /
  multi_shot_single_call / t2i / i2i`,以及观测用的 `flf2v_mode` 标签
- 约束:`min_duration_s / max_duration_s / duration_granularity_s /
  supported_resolutions / max_prompt_chars / max_reference_images`
- 吞吐:`max_concurrency / requests_per_minute`
- 成本:`estimated_cost_per_call_usd / estimated_cost_per_second_usd`
- 重试:`max_retries`(默认 3)/ `retry_backoff_base_s`(1.5)/ `retry_backoff_max_s`(60)/
  `retryable_error_names`(默认含 `BackendRenderError` 和一批 httpx 超时/连接异常名)

两个错误类型:`RenderPlanError`(计划期不匹配,调用前抛,不烧配额)、
`BackendRenderError`(上游 API 报错)。

---

## 7. 鉴权、限流、重试

### 7.1 Provider

每个上游用一个 `Provider` 对象封装:key 环境变量列表、多 key CSV env、base_url、
限流表路径、响应分类函数。已注册 5 个:

| provider | 单 key env | 多 key CSV env | 限流表 |
|---|---|---|---|
| `dashscope` | `DASHSCOPE_API_KEY` | `DASHSCOPE_API_KEYS` | `<repo>/configs/dashscope_rate_limits.json` |
| `openai` | `OPENAI_API_KEY` | `OPENAI_API_KEYS` | `<repo>/configs/openai_rate_limits.json` |
| `kling` / `pixverse` / `vidu` | 共用 dashscope 的 key env | 同左 | 无 |

`Provider.api_key()` 按 `api_key_envs` 顺序找第一个非空,都没有就取 CSV 的第一个。
限流表可以用 `DASHSCOPE_RATE_LIMITS_CONFIG` / `OPENAI_RATE_LIMITS_CONFIG` **整份换掉**
(是文件路径选择器,不是单值覆盖 —— 要改数值就直接编辑 JSON)。

新增 provider:在 `_common/providers/<name>.py` 里 `register_provider(Provider(...))`,
并在 `providers/__init__.py` 的循环里加一行模块名。

### 7.2 with_key + KeyPool

```python
from videorlm.backends._common.platforms import with_key

rsp = with_key("dashscope", lambda api_key: client.chat.completions.create(...))
rsp = with_key("openai", _do_call, model="gpt-image-2")   # 传 model 才过限流闸
```

`model=` 非空时,先过 `RateLimiter`(该 provider 的 JSON 里那一行的
`interval_s` + `max_parallel`),再进 KeyPool。不传就只走 KeyPool。
`classify_response=` 不传时自动用该 provider 的分类函数;传了可以透传 `Retry-After`
(返回 `(kind, cooldown_s)` 元组)。

**KeyPool 选 key**:`score = w_load × (in_flight / cap) + w_err × error_rate_ewma`,
取最低分,并列随机。健康度下滑的 key 在触发 cooldown 之前就已经被自动降权。

**释放时按错误类型分级 cooldown**:

| kind | 起始 | 上限 | 指数递增 |
|---|---|---|---|
| `ok` | 0 | 0 | — |
| `rate_limit` | 60s | 300s | 是 |
| `tps_throttle` | 5s | 15s | 否 |
| `overload_503` | 10s | 60s | 否 |
| `daily_quota` | ∞ | ∞ | — |
| `auth_invalid` | ∞ | ∞ | — |
| `network` / `other` | 0 | 0 | — |

`∞` 表示该 key 在本进程内永久停用(重启或人工换 key 才恢复)。
`network` / `other` 不 cooldown,交给外层 `retry_until_exhausted` 决定。

旋钮:`RECA_KEYPOOL_PER_KEY_CAP_<PROVIDER>` / `RECA_KEYPOOL_PER_KEY_CAP`(默认 8)、
`RECA_KEYPOOL_W_LOAD` / `RECA_KEYPOOL_W_ERR`(默认 1.0)、`RECA_KEYPOOL_EWMA_ALPHA`(0.2)、
`RECA_KEYPOOL_COOLDOWN_S`、健康度打印 `RECA_KEYPOOL_HEALTH_DUMP`(默认开)+
`RECA_KEYPOOL_HEALTH_DUMP_EVERY_S`(默认 60)。

### 7.3 内容审核 fallback 链

`qwen-image-2.0-pro` / `qwen-image-2.0` 是全仓唯一会**换后端**的地方:当异常文本命中
内容审核关键词(`DataInspectionFailed` / `content_filter` / `GreenNet` / `敏感` / `审核未通过` …)
时,按 `RECA_IMAGE_FALLBACK_CHAIN`(默认 `wan2.7-image-pro,gpt-image-2`)依次重试同一个请求,
每跳都打日志;整条链都被拒就抛最后一个错误。设成空串即关闭,退化成"第一次被拒就抛错"。
非审核类失败(网络 / 配额 / 超时)不走这条链,仍然是同后端重试。

### 7.4 重试与超时

```python
from videorlm.backends._common.retry import RetryPolicy, retry_until_exhausted
from videorlm.backends._common.timeout import call_with_timeout, BackendCallTimeout
```

`retry_until_exhausted(backend_name, call_fn, policy=...)` 只在**同一个后端**上重试。
后端实现里**不要再写自己的 retry 循环** —— 声明 `capabilities().max_retries` 即可。

### 7.5 trace

`trace_event(component, event, *, log_dir=None, **fields)` 往 `<log_dir>/trace.jsonl`
追加一行 JSON;`RECA_BACKEND_TRACE_LOG_PATH` 可以指定一个全局路径。
软降级(忽略末帧、截断参考图等)会写 `video.degrade` 事件,事后可审计哪些请求被降级执行了。

### 7.6 OSS

```python
from videorlm.backends._common.oss_publisher import upload_file, upload_bytes
```

对象 key 形如 `<base_prefix>/<prefix>/YYYYMMDD/<request_id><后缀>`,上传后设为公开读,
返回公网 URL。配置来自 `oss_AccessKey_ID` / `oss_AccessKey_Secret` / `oss_bucket`(或
`os_path` 里的 `oss://bucket/prefix`)/ `region` 或 `OSS_ENDPOINT`。没配全时返回 `None`,
调用方自行退回本地路径。**这是全仓上传 OSS 的唯一出口。**

---

## 8. 不变量

1. `dispatch_*` 只用一个后端,失败只在它上面重试;跨后端切换由上层负责
   (`qwen-image` 的审核 fallback 是唯一例外,且必须打日志)。
2. `SegmentRequest.mode` 由上层显式给定,后端不得按 `len(refs)` 自行推断。
3. 计划期不匹配(时长越界、后端不支持该 mode)抛 `RenderPlanError`,**在真正调用之前**,
   不浪费并发槽和配额。
4. 软降级必须写 `video.degrade` trace 并注明原因。
5. 顶层 `from videorlm.backends.media import ...` 的 re-export 是公共 API;
   `interface/` 与 `impl/` 内部布局可以改,re-export 必须稳定。
6. 所有 DashScope SDK 调用走 `with_key("dashscope", ...)`;所有 OSS 上传走 `oss_publisher`。
