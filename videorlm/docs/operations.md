# operations — 怎么跑

## 1. 运行环境

**解释器**需要这些包:`dashscope` / `openai` / `pydantic` / `av` / `PIL` / `oss2` / `httpx` /
`typing_extensions`(`numpy` / `cv2` / `skimage` / `jinja2` / `pytest` 只有测试和
`jinja2` 模板风格用得到)。先自检:

```bash
/usr/local/bin/python3 -c "import dashscope, openai, pydantic, av, PIL, oss2, httpx; print('deps OK')"
```

外部命令依赖 `ffmpeg` / `ffprobe`(拼接与探测)。

**工作目录固定在 repo 根**(`unirlm-02/`),所有命令都以 `python3 -m videorlm...` 的模块形式跑。

**密钥**放在 `<repo>/.env`,`_smoke.py::load_env()` 启动时逐行读进 `os.environ`
(已存在的环境变量不会被覆盖)。用到的键:

```
DASHSCOPE_API_KEY / DASHSCOPE_API_KEYS      # 后者是逗号分隔多 key,启用轮转
OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_API_KEYS
oss_AccessKey_ID / oss_AccessKey_Secret / oss_bucket / region / OSS_ENDPOINT / os_path
```

> `backends._common.env.env_value()` 自己也有一层 `.env` 文件兜底,但它搜的是
> `videorlm/` 往上第 4/5/6 层目录和 `~/.env`,**不包含 `<repo>/.env`**。
> 实际生效的路径是 `_smoke.py` 先把 `<repo>/.env` 注入 `os.environ`,
> `env_value` 优先读 `os.environ`。不经过 `_smoke.py` 的入口需要自己把 key 导进环境。

**代理**:`load_env()` 默认会把 `HTTP_PROXY` / `HTTPS_PROXY` 设成
`RECA_HTTP_PROXY_URL`(默认 `http://127.0.0.1:20172`)。不想走代理就设
`RECA_DISABLE_HTTP_PROXY=1`,并在外面 `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY`。

---

## 2. 跑一条

```bash
cd /mnt/workspace/akide/code/unirlm-02
/usr/local/bin/python3 -m videorlm.framework._scripts._smoke \
    --story input_examples/true_input/example_01.txt \
    --out-dir videorlm/outputs/demo/ex01 \
    --label ex01 \
    --segments --render --validate --validate-segments \
    --backend happyhorse --video-resolution 1920x1080 --seed 0
```

| 参数 | 默认 | 作用 |
|---|---|---|
| `--story` | `input_examples/example_01.txt` | 剧情 txt |
| `--out-dir` | `videorlm/outputs/version_2.2/ex01` | 产物目录 |
| `--label` | `smoke` | 日志行前缀 |
| `--segments` | off | 规划到 segment 粒度并生成 `render_plan.json`;不给就只跑 `plan_skeleton` |
| `--render` | off | 跑渲染(需要 `--segments`) |
| `--resume` | off | `render_plan.json` 已存在就跳过规划,直接渲染 |
| `--validate` | off | 开 anchor validator |
| `--validate-segments` | off | 开 segment validator + repair router |
| `--max-repair-attempts` | 2 | 同时作用于两个 validator |
| `--segment-axis-threshold` | 0.6 | segment 校验的单轴通过线 |
| `--segment-overall-threshold` | 0.7 | segment 校验的总分通过线 |
| `--segment-validator-model` | `qwen3-vl-plus` | 打分用的视觉模型 |
| `--segment-validator-fps` | 10 | 判定时服务端抽帧率(1–10) |
| `--segment-repair-strategies` | 三个全允许 | 逗号分隔白名单;传空串 = 只打分不重渲 |
| `--no-segment-best-of-n` | off | 关掉 attempt 快照与择优,恒用最后一次渲染 |
| `--segment-validator-on-error` | `raise` | validator 调用失败时 `raise` 打挂 shot chain / `accept` 保留当前渲染 |
| `--seed` | 0 | 写进 `render_plan` 各 request 和 `summary.json` |
| `--backend` | `happyhorse` | 视频后端分支:`wan` / `happyhorse` / `ltx` |
| `--force-i2v` | off | 所有 segment 强制走 i2v 并丢掉参考图(设 `RECA_FORCE_I2V=1`) |
| `--video-resolution` | `1920x1080` | 视频分辨率。**`--backend wan` 必须用 `1280x720`**,wan2.7 只支持 720p |

`--backend` 展开成的实际路由(用 `setdefault`,已有的同名 env 优先):

| 分支 | segment_r2v | segment_i2v | bridge |
|---|---|---|---|
| `happyhorse` | `happyhorse-1.0-r2v` | `happyhorse-1.0-i2v` | `wan2.7-i2v` |
| `wan` | `wan2.7-r2v` | `wan2.7-r2v` | `wan2.7-i2v` |
| `ltx` | `ltx-2.3` | `ltx-2.3` | `ltx-2.3` |

图像三种 kind(`portrait` / `anchor_image` / `image_edit`)统一 `setdefault` 到
`gpt-image-2`,想换就在外面 export `RECA_RENDER_BACKEND_*`。

**planner 端点**默认 DashScope + `qwen3.6-max-preview`。同时设
`RECA_PLANNER_API_KEY` + `RECA_PLANNER_BASE_URL` 可整体切到别的网关(此时默认模型变 `gpt-5.5`)。
`RECA_PLANNER_API_PATH` 选 ingress:`auto`(默认,`gpt-5*` 走 `/v1/messages`,其余走
chat.completions)/ `messages` / `chat`。

## 3. 批量跑

`scripts/` 下是若干批处理 wrapper,共同形状:清代理 → 设 key 和 backend env →
wan / happyhorse 两个 worker 并行、各自串行遍历样本 → 每个样本 `final.mp4` 已存在就跳过。

```bash
nohup bash videorlm/scripts/run_batch_qwen.sh > videorlm/outputs/version_2.4/_batch_logs/batch.log 2>&1 &
disown
tail -f videorlm/outputs/version_2.4/_batch_logs/ex01_wan.log
```

可用 `BATCH_WAN_RANGE` / `BATCH_HH_RANGE` 覆盖样本编号列表,`BATCH_RESUME=1` 让每个样本带
`--resume`。

其它脚本:`run_batch_18.sh` / `run_batch_45_61.sh` / `run_batch_62_68.sh` / `resume_batch.sh`
(不同批次的样本范围),`run_ex01_qwen.sh`(单样本双后端),
`cancel_stale_happyhorse_tasks.py`(取消上游残留任务),
`prepare_qwen_bundle.py` / `append_to_v24_bundle.py` / `align_qwen_sections.py`(产物打包),
`probe_messages_api.py` / `probe_responses_api.py` / `test_structured_output*.py`(端点探针)。

> 仓库中的运行脚本只读取环境变量；API key 不应写入脚本或提交到仓库。

## 4. 产物与断点续跑

```
<out-dir>/
├── _frozen_prompts/{shot_planner,segment_planner,segment_replanner,
│                    validator_anchor,validator_segment}.py + freeze_meta.json
├── skeleton.json / planner.json / render_plan.json
└── run/
    ├── portraits|locations|props|anchors/  <id>.png + <id>.png.url
    ├── anchors/repairs/                    <anchor>.repair_<n>.png
    ├── segments/                           <id>.mp4 + <id>.mp4.url + <id>.tail.png
    ├── bridges/                            <id>.mp4 + <id>.mp4.url
    ├── logs/trace.jsonl
    ├── final.mp4 / summary.json / _backend_info.json
```

**跳过规则只有一条**:产物文件和 `<产物>.url` sidecar 同时存在且 sidecar 非空 → 跳过。
所以

- 想重渲某个 cell:删掉它的 `.url`(或连产物一起删)
- 想重跑整轮渲染但保留规划:留着 `render_plan.json`,加 `--resume`
- 想重新规划:删 `render_plan.json`(或不加 `--resume`)

`--resume` 只影响"要不要重新调 planner",不影响渲染层的跳过判断。

失败的 cell 不会留任何标记文件,也不会用占位视频顶替 —— 失败就是抛异常。

`summary.json` 记录:总耗时、各阶段耗时、四类计数、当前 backend 路由与并发、
`seed`、`RECA_IMAGE_BACKEND_TAG`。同样的后端信息另存一份 `_backend_info.json` 方便下游 grep。

---

## 5. 环境变量总表

### 后端路由

| 变量 | 说明 |
|---|---|
| `RECA_RENDER_BACKEND_SEGMENT_I2V` / `..._R2V` | 按 mode 指定 segment 后端(最高优先) |
| `RECA_RENDER_BACKEND_SEGMENT` | 两个 mode 的共同兜底 |
| `RECA_RENDER_BACKEND_BRIDGE` | bridge 后端 |
| `RECA_RENDER_BACKEND_PORTRAIT` / `..._ANCHOR_IMAGE` / `..._IMAGE_EDIT` | 各图像 kind 的后端 |
| `RECA_RENDER_BACKEND_LOCATION` / `..._PROP` | dispatch 层支持,但 pipeline **不会**用到 —— 它把 location / prop 当 `anchor_image` 派发,改这两个没有效果 |
| `RECA_FORCE_I2V=1` | 所有 segment 强制 i2v 并丢参考图 |
| `RECA_IMAGE_FALLBACK_CHAIN` | qwen-image 被内容审核拒绝时的后备后端链,默认 `wan2.7-image-pro,gpt-image-2`,空串关闭 |
| `RECA_IMAGE_BACKEND_TAG` | 写进 `summary.json` 的自由文本标签 |

### 模型与 planner

| 变量 | 默认 | 说明 |
|---|---|---|
| `RECA_PLANNER_MODEL` | `qwen3.6-max-preview` | shot_planner 模型 |
| `RECA_SP_MODEL` | `qwen3.6-max-preview` | segment_planner 模型 |
| `RECA_PLANNER_API_KEY` / `RECA_PLANNER_BASE_URL` | — | 两个都设才生效,把 planner 切到别的网关 |
| `RECA_PLANNER_API_PATH` | `auto` | `auto` / `messages` / `chat` |
| `RECA_PLANNER_THINKING_BUDGET` | 16000 | Anthropic Messages 的 thinking 预算 |
| `RECA_QWEN_THINKING` | 1 | qwen 系是否开 `enable_thinking` |
| `RECA_PLAN_SKELETON_MAX_RETRIES` | 10 | skeleton 解析/校验失败的重问次数 |

### 并发

| 变量 | 默认 | 说明 |
|---|---|---|
| `RECA_PLANNER_POOL_SIZE` | 8 | planner 角色池(同时也是 `plan_segments_all` 的线程数) |
| `RECA_ANCHOR_VALIDATOR_POOL_SIZE` | 4 | anchor validator 池 |
| `RECA_SEGMENT_VALIDATOR_POOL_SIZE` | 2 | segment validator 池 |
| `RECA_RENDER_POOL_SIZE` | 8 | shot-chain 线程池 |
| `RECA_BACKEND_CONCURRENCY_<NAME>` | — | 直接覆盖某后端的并发上限(名字大写,`-`/`.`→`_`) |
| `RECA_WAN27_I2V_WORKERS` / `RECA_WAN27_R2V_WORKERS` / `RECA_HAPPYHORSE_I2V_WORKERS` / `RECA_HAPPYHORSE_R2V_WORKERS` / `RECA_WAN27_IMAGE_WORKERS` / `RECA_WAN27_IMAGE_PRO_WORKERS` / `RECA_WAN26_IMAGE_WORKERS` / `RECA_QWEN_IMAGE_WORKERS` / `RECA_GPT_IMAGE_2_WORKERS` / `LTX23_WORKERS` | 见 [backends.md §4](backends.md) | 各后端并发 |
| `RECA_FORK_INHERIT_PAIRS` | 2 | `inherit_policy="last_n_pairs"` 时继承几对消息 |

### KeyPool / 限流 / 重试 / 超时

| 变量 | 默认 | 说明 |
|---|---|---|
| `RECA_KEYPOOL_PER_KEY_CAP_<PROVIDER>` / `RECA_KEYPOOL_PER_KEY_CAP` | 8 | 单 key 并发上限 |
| `RECA_KEYPOOL_W_LOAD` / `RECA_KEYPOOL_W_ERR` | 1.0 | 选 key 评分权重 |
| `RECA_KEYPOOL_EWMA_ALPHA` | 0.2 | 错误率 EWMA 平滑系数 |
| `RECA_KEYPOOL_COOLDOWN_S` | — | 覆盖默认 cooldown |
| `RECA_KEYPOOL_HEALTH_DUMP` / `RECA_KEYPOOL_HEALTH_DUMP_EVERY_S` | 1 / 60 | 周期性打印各 key 健康度 |
| `DASHSCOPE_RATE_LIMITS_CONFIG` / `OPENAI_RATE_LIMITS_CONFIG` | — | **整份换掉**限流 JSON 的路径(不是单值覆盖) |
| `RECA_VIDEO_CALL_TIMEOUT_S` | 3700 | 单次 dispatch 的 wall-clock 超时 |
| `RECA_BACKEND_TIMEOUT_WORKERS` / `RECA_BACKEND_TIMEOUT_SLOT_WAIT_S` | — | 超时执行器的线程数 / 等槽超时 |
| `RECA_GPT_IMAGE_2_MAX_RETRIES` / `RECA_GPT_IMAGE_2_BACKOFF_MAX_S` / `RECA_GPT_IMAGE_2_REF_CAP` | 16 / 300 / 4 | gpt-image-2 专属 |
| `RECA_HAPPYHORSE_POLL_INTERVAL_S` / `RECA_HAPPYHORSE_POLL_MAX_WAIT_S` | 5 / 3600 | happyhorse 轮询 |
| `RECA_HAPPYHORSE_R2V_RATIO` / `RECA_HAPPYHORSE_I2V_RATIO` | — | 透传给 API 的 `ratio` 参数 |
| `RECA_WAN27_I2V_SEGMENT_REUSE_FIRST_AS_LAST` | — | 设 1 才允许 `wan2.7-i2v` 渲 segment(拿首帧当末帧) |
| `M12_SEGMENT_VALIDATOR_TIMEOUT_S` | 900 | segment validator 单次调用超时 |
| `RECA_SEGMENT_AXIS_THRESHOLD` / `RECA_SEGMENT_OVERALL_THRESHOLD` | 0.6 / 0.7 | segment 校验通过线(CLI 同名参数优先) |
| `RECA_SEGMENT_VALIDATOR_MODEL` / `RECA_SEGMENT_VALIDATOR_FPS` | `qwen3-vl-plus` / 10 | 打分模型与抽帧率 |
| `RECA_SEGMENT_REPAIR_STRATEGIES` | 三个全允许 | 修复策略白名单;空串 = 只打分不重渲 |
| `RECA_SEGMENT_BEST_OF_N` | 1 | 设 0 关掉 best-of-N |
| `RECA_SEGMENT_VALIDATOR_ON_ERROR` | `raise` | `raise` / `accept` |

### 其它

| 变量 | 默认 | 说明 |
|---|---|---|
| `RECA_DISABLE_HTTP_PROXY` / `RECA_HTTP_PROXY_URL` | — / `http://127.0.0.1:20172` | 代理开关与地址 |
| `RECA_BACKEND_TRACE_LOG_PATH` | — | 全局 trace 文件(否则写各 request 的 `log_dir/trace.jsonl`) |
| `RECA_XSEM_ENABLE` / `RECA_XSEM_BASE_DIR` / `RECA_XSEM_TTL_S` / `RECA_XSEM_POLL_S` / `RECA_XSEM_<BACKEND>` | 见下 | 跨进程信号量 |
| `LTX23_ROOT` | — | ltx-2.3 权重根目录 |

---

## 6. 跨进程信号量

同一台机器同时跑多个 `_smoke.py` 时,进程内的信号量管不到彼此。设
`RECA_XSEM_ENABLE=1`,`_smoke.py` 会在导入 pipeline 之前启用一层**文件锁信号量**
(`fcntl.flock` + 每槽一个 marker 文件),把 `dispatch_segment` / `dispatch_bridge` /
`dispatch_image` 和 `OpenAICompatibleAgent.prompt` 都纳入跨进程配额。
超过 TTL 的槽会被自动回收,所以 `kill -9` 不会永久占坑。

默认上限(每个都能用 `RECA_XSEM_<NAME>` 覆盖):

```
happyhorse-1.0-r2v 18   happyhorse-1.0-i2v 8    happyhorse-1.0-t2v 8
wan2.7-i2v          8   wan2.7-r2v        12    wan2.7-t2v        16
wan2.7-image        8   wan2.7-image-pro   8    wan2.6-image       8
wanx-v1             2   qwen-image-2.0-pro 8
gpt-image-2        12   gpt-image-2-pro   12
gpt-5.5            12   qwen3.6-max-preview 12  qwen3.6-plus 12   qwen3.6-flash 12
```

槽目录默认在 `/tmp/videorlm_xsem/`,可用 `RECA_XSEM_BASE_DIR` 换到别处。

---

## 7. 本地 web server

给"自带 key 跑一条"的页面用的极简任务运行器,纯标准库,只监听回环:

```bash
/usr/local/bin/python3 -m videorlm.framework._scripts._web_server --port 18800 --host 127.0.0.1
```

| 端点 | 作用 |
|---|---|
| `GET /health` | `{ok: true}` |
| `POST /api/run` | 提交一次运行,返回 `{job_id}` |
| `GET /api/jobs/<id>` | 状态 + 各阶段进度 + 日志尾部 |
| `GET /api/jobs/<id>/manifest` | 当前磁盘上所有产物的 URL(前端渐进渲染用) |
| `GET /api/jobs/<id>/final.mp4` | 成片 |
| `GET /api/jobs/<id>/asset/<path>` | 任务目录下任意文件(内联预览用) |

每个任务是一个 `_smoke.py` 子进程,工作目录 `<repo>/byok_jobs/<job_id>/`;
阶段进度靠正则匹配日志里的 `[stage] X START` / `dt=` 推断。
表单字段会被翻译成 env:`planner_model` / `planner_api_key` / `planner_base_url` /
`image_backend` / `video_backend` / `resolution` / `seed` / `validate` /
`validate_segments` / `image_concurrency` / `image_fallback_chain` / `resume_job_id`。

---

## 8. 测试

```bash
cd /mnt/workspace/akide/code/unirlm-02
/usr/local/bin/python3 -m pytest videorlm/backends/tests -q
```

`backends/tests/` 覆盖:并发信号量、KeyPool 评分与 cooldown 策略、provider 注册、
`with_key` 中间件、限流表加载、r2v 首帧 SSIM。
需要真实 API 的用例由 `RECA_E2E_LIVE` 控制,默认跳过。

`framework/_scripts/_test_*.py` 是三个独立脚本(不是 pytest 用例):
`_test_web_server.py` / `_test_manifest_scan.py` / `_test_demo_e2e.py`。

---

## 9. 错误对照

| 报错 | 含义 / 处理 |
|---|---|
| `Backend '<name>' ... is not registered. Available: []` | `media/impl/` 没能 import(多半是 `dashscope` 没装,或换了解释器) |
| `RenderPlanError: backend=X does not implement render_segment` | 路由把 segment 指到了纯图像/纯 bridge 后端 |
| `BackendRenderError: wan2.7-i2v ... pure i2v segment generation is not its native shape` | `segment_i2v` 指到了 `wan2.7-i2v`;改指 `happyhorse-1.0-i2v` / `wan2.7-r2v`,或设 `RECA_WAN27_I2V_SEGMENT_REUSE_FIRST_AS_LAST=1` |
| `BackendRenderError: <backend> supports mode='i2v' only` | 给只支持 i2v 的后端发了 `mode="r2v"` |
| `_dispatch_one_image[...]: refusing to cache non-URL` | 图像后端没能产出公网 URL(通常是 OSS 没配全) |
| `[segment-trace] <id>: NO first_url` / `first_url is local path` | 上游 anchor 缺失或没上 OSS |
| `_extract_last_frame: PyAV decoded 0 frames` | 下载到的 mp4 是坏文件 |
| `plan_skeleton: exhausted N retries` | planner 连续 N 次输出不合 schema;看日志里的 `reply head` |
| `shot X: Σ segment duration (a) != shot.duration_s (b)` | segment_planner 拆分时长对不上,会自动重问,持续失败才抛 |
| `SegmentValidatorError: ...` | validator 调用或输出格式失败。**不会被吞** —— 该 shot chain 会失败(mp4 已进缓存,`--resume` 不会重渲) |
| `[router] <id>: strategy=replan but sp_replan_cfg=None` | 开了 router 但没给 replan 配置,按设计直接报错不静默降级 |
| `render_segments: N shot chain(s) failed` | 汇总错误;其它 shot 的结果已缓存,`--resume` 续跑 |
| `concat_final: ffmpeg exit <rc>` | 上面会打印 ffmpeg stderr 末尾 30 行 |
