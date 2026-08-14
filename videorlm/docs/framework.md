# framework — 编排层

`videorlm/framework/` 负责:用 LLM 把剧情拆成可渲染的计划,调 `videorlm.backends` 渲染,
用 VLM 质检并修复,最后拼成 mp4。它是**唯一**决定"渲什么、渲几次、按什么顺序渲"的地方;
"用哪个模型、怎么重试、怎么限流"全在 [backends.md](backends.md)。

顶层三个函数(`videorlm.framework` 直接导出):

```python
from videorlm.framework import run_planner, to_render_plan, run_render

planner     = run_planner(story, sp_cfg, out_dir=out_dir)   # 规划
render_plan = to_render_plan(planner, run_dir, seed=0)      # 转换(纯 Python)
final_mp4   = run_render(render_plan, out_path)             # 渲染
```

`_smoke.py` 走的是拆开的等价路径(`plan_skeleton` → `plan_segments_all` →
`merge_into_planner_output` → `to_render_plan` → `run_render`),这样每步都能落盘。

---

## 1. 六个 LLM 角色

| 角色 | 模块 | system prompt | 默认模型 | 角色池 | 输入 → 输出 |
|---|---|---|---|---|---|
| shot_planner(parent) | `shot_planner/plan.py` | `prompts/shot_planner/system.md` | `qwen3.6-max-preview` | planner | story → skeleton |
| segment_planner(SP) | `segment_planner/plan.py` | `prompts/segment_planner/system.md` | `qwen3.6-max-preview` | planner | shot + anchor → 该 shot 的 segments |
| segment_replanner | `segment_replanner/plan.py` | `prompts/segment_replanner/system_replan.md`<br>`.../system_micro_adjust.md` | `qwen3.6-max-preview` | planner | 失败判定 → 改写整 shot / 单 segment |
| anchor validator | `validator/anchor/validate.py` | `prompts/validator_anchor/system.md` | `gpt-5.5` | anchor_validator | anchor PNG + 设计规格 → 判定 |
| segment validator | `validator/segment/validate.py` | `prompts/validator_segment/system.md` | `qwen3-vl-plus` | segment_validator | segment mp4 + portraits → 3 轴打分 |
| repair router | `validator/segment/router.py` | `prompts/validator_segment/router.md` | `qwen3.6-plus` | (随 cfg) | 判定 + 历史 → 修复策略 |

模型不写死在代码里,由调用方构造 config 传入(见 `_scripts/_smoke.py`)。
角色池见 [agents.md §5](agents.md)。

### prompt 的存放与冻结

所有 system prompt 的正文都是 `framework/prompts/<角色>/*.md`,Python 侧只有一行装载:

```python
from videorlm.framework.templates import ChatPromptTemplate
PARENT_SYSTEM_PROMPT = ChatPromptTemplate.from_files(
    system=_PROMPTS_ROOT / "shot_planner" / "system.md",
).format_system()
```

`ChatPromptTemplate` 支持三种插值风格:`dollar`(默认,`$var` / `${var}`,未知占位符原样保留)、
`format`(`{var}`)、`jinja2`(惰性 import,支持循环/条件)。

`_scripts/_freeze_prompts.py::freeze_prompts(out_dir)` 在每次运行开始时把
`shot_planner / segment_planner / segment_replanner / validator_anchor / validator_segment`
五个 prompt 模块的源码快照写进 `<out_dir>/_frozen_prompts/`,附 `freeze_meta.json`
(冻结时间 + git sha)。`run_planner(out_dir=...)` 和 `_smoke.py` 都会调它。

---

## 2. 规划侧

### 2.1 plan_skeleton

```python
plan_skeleton(parent: Agent, story: str, *, max_retries=RECA_PLAN_SKELETON_MAX_RETRIES) -> dict
```

把剧情整段丢给 parent agent,要求回一个 JSON。解析 + 校验 + 重试统一走
`parsers.JSONOutputParser(Skeleton).parse_with_retry`:解析失败或 schema 不过就
**丢弃这一轮的 user+assistant 消息对**再重问,指数退避 2s→60s,耗尽抛 `RuntimeError`。
默认 `max_retries=10`(env `RECA_PLAN_SKELETON_MAX_RETRIES`)。

`Skeleton`(`schemas.py`)的结构与硬约束:

| 字段 | 类型 | 约束 |
|---|---|---|
| `source` | `{input_path, n_shots≥1, per_shot_duration_s[], seed}` | — |
| `portrait_plan` | `list[Portrait]` | `id` / `name` / `reference_role∈{portrait, supporting_portrait}` / `prompt` |
| `location_plan` | `dict[loc_id, Location]` | `Location.props` 是 `dict[prop_id, Prop]`,`Prop.owner` 可指向 portrait_id 或 loc_id |
| `boundarys.boundary_anchors` | `list[BoundaryAnchor]` | `id` 匹配 `^a\d{2}_\w+`;`reference_inputs = {portrait, place, prop}`(逗号分隔多值) |
| `shots` | `list[Shot]` | `id` 匹配 `^shot\d{2}_\w+`;`duration_s ∈ [3,240]`;`story_goal/start_state/end_state/visual_intent` 必填 |
| `transitions` | `list[Transition]` | `mode ∈ {cut, bridge}`;`bridge` 时 `duration_s/first_frame/last_frame/prompt` 必须齐 |

跨字段不变量(`@model_validator`):

- `len(shots) == len(boundary_anchors)` —— **shots[i] 与 anchors[i] 一一对应**,`_anchor_for_shot` 靠下标取
- `len(transitions) == len(shots) - 1`

所有模型都是 `extra="allow"`:planner 多写的字段原样穿过 `model_dump()`,不会被丢。

### 2.2 plan_segments_all

```python
plan_segments_all(parent_state, skeleton, sp_cfg, *, max_workers=PLANNER_POOL_SIZE) -> dict
```

每个 shot 起一个**独立的 SP agent**,线程池并行。SP 不是"接着 parent 聊",而是
`_common/fork.py::sub_conversation_with_system_swap` 造出来的新会话:

```
parent.state = [system(SHOT_PLANNER), user(story), assistant(skeleton)]
        │  inherit_policy="first_pair_only"  → 只带走 [user(story), assistant(skeleton)]
        ▼
sp.state     = [system(SEGMENT_PLANNER), user(story), assistant(skeleton)]
        │  sp.prompt(该 shot 的 user 消息)
        ▼
sp.state     = [..., user(per-shot), assistant(segments_json)]
```

`inherit_policy` 四种:`all`(除 system 全继承)/ `first_pair_only`(默认)/
`last_n_pairs`(条数由 `RECA_FORK_INHERIT_PAIRS` 定,默认 2 对)/ `none`。
也可以用 `inherited_msg_count=N` 直接取尾部 N 条(segment validator 用 4)。
新 state 一定拿到新的 `thread_id`(`sub-<uuid12>`)。

每个 shot 的输出用 `ShotSegments.validate_for_shot(raw, shot, anchor)` 校验,不过就重问
(`retry_agent_call`,默认 3 次):

- `Σ segment.duration_s == shot.duration_s`
- `segment_index_in_shot` 从 0 连续无空洞
- `end_state` 非空
- `seg0.start_anchor == anchor.id` 且 `seg0.first_frame_path == "shot_start_anchor"`
- `seg_k>0 .first_frame_path == "previous_segment_last_frame"`
- `id` 匹配 `^seg_shot\d{2}_\w+`

`merge_into_planner_output(skeleton, segments)` 就是 `{**skeleton, "segments": segments}`。

### 2.3 离线重建 agent 状态

SP 在自己的线程里跑完就 close 了,后面的 validator / replanner 需要"当时那个 SP"的记忆,
于是有两个纯函数按同样形状重新拼出 state:

```python
reconstruct_segment_planner_state(planner, story, shot_id)
# → [system(SP), user(story), assistant(skeleton), user(per-shot), assistant(该 shot segments)]

reconstruct_parent_state(story, skeleton)
# → [system(SHOT_PLANNER), user(story), assistant(skeleton)]
```

`--resume` 场景下 planner agent 早就没了,这两个函数是唯一的记忆来源。

---

## 3. 转换:to_render_plan

```python
to_render_plan(planner, run_dir, *, seed=0,
               resolution="1280x720",         # 图像
               video_resolution="1920x1080")  # 视频
```

纯 Python,不调任何 API。把 planner 的每个实体包成一个 request dict,填好落盘路径,
URL 全部留空(渲染时才解析)。**图像和视频用两个独立的分辨率参数**。

| planner 来源 | render_plan 位置 | request 字段 | 分辨率 | 落盘 |
|---|---|---|---|---|
| `portrait_plan[]` | `portrait_plan{id}` | `image_request` | `resolution` | `run/portraits/<id>.png` |
| `location_plan{}` | `location_plan{id}` | `image_request` | `resolution` | `run/locations/<id>.png` |
| `location_plan{}.props{}` | `prop_plan{id}`(提升到顶层,保留 `owner`) | `image_request` | `resolution` | `run/props/<id>.png` |
| `boundarys.boundary_anchors[]` | `boundary_anchors[]` | `image_request` | `resolution` | `run/anchors/<id>.png` |
| `segments{}` | `segments{seg_id}` | `segment_request` | `video_resolution` | `run/segments/<id>.mp4` |
| `transitions[] where mode=="bridge"` | `boundary_policies[]` | `bridge_request` | `video_resolution` | `run/bridges/<id>.mp4` |

`shots` / `transitions` 原样搬过去。anchor 的 `image_request.references` 是
`[{role, url:"", asset_id}]`,`role` 由 `reference_inputs` 映射:
`portrait→portrait`、`place→scene`、`prop→reference`。

---

## 4. 渲染侧:run_render 五阶段

```python
run_render(render_plan, out_path, *, validator=None, segment_validator=None, seed=None) -> str
```

每阶段打一行 `[stage] <name> dt=<秒>`,最后写 `summary.json` + `_backend_info.json`。

### 4.1 阶段一:_render_image_dag(并发 16)

portrait / location / prop / anchor 在**同一张依赖图**里跑,谁的依赖齐了谁先发:

| 节点 | 依赖 |
|---|---|
| portrait / location | 无 |
| prop | 它的 `owner`(若 owner 在图里) |
| anchor | `image_request.references[].asset_id` 里所有在图内的资产 |

> **派发用的 kind 不等于 render_plan 里的 kind 字段**:DAG 只给 portrait 打 `"portrait"`,
> location / prop / anchor **一律按 `"anchor_image"` 派发**。也就是说
> `RECA_RENDER_BACKEND_LOCATION` / `RECA_RENDER_BACKEND_PROP` 在这条路径上不起作用,
> 要改 location / prop 的后端得改 `RECA_RENDER_BACKEND_ANCHOR_IMAGE`。

anchor 提交前才从当前已完成的 URL 池里解析引用,并按目标后端的
`capabilities().max_reference_images` 截断;截断顺序按角色优先级
`portrait > scene > reference > start > end > source`,同级保持原顺序,丢弃的会打
`[ref-truncate]`。

单个节点的规则(`_dispatch_one_image`):

1. 有 `.url` sidecar 且是 `http` 开头 → 直接复用,打 `[render-skip]`
2. 有 `.url` 但内容是本地路径 → 重新上传 OSS 换成公网 URL(`[republish]`),失败则报错
3. 否则调 `dispatch_image`;**结果不是 `http` 开头就拒绝写缓存并抛错**(下游 segment 必须拿到公网 URL)

某个节点失败不会立刻中断整张图,其它独立节点继续跑;全部结束后如果有失败,汇总抛
`RuntimeError` 并提示 `--resume`。

### 4.2 阶段二:anchor validator(可选,`--validate`)

`ValidatorParams(planner, story, validator_cfg, max_repair_attempts=2, repair_dir=None)`。
每个 anchor 一个任务,线程池 `min(4, n_anchors)`。

判定走 `validate_anchor_via_agent`,agent 由
`make_validator_from_segment_planner_state(sp_state, cfg)` 从**重建出来的 SP 状态**再 fork 一层
(继承尾部 4 条)。喂给它的是:anchor 图 URL(优先公网 URL,否则本地路径)+
`reference_inputs` 里每个 portrait / prop / place 的**文字设计规格**(`{id, name, prompt}`)。
参考图本身不进视觉通道,只有 anchor 图进。

输出必须含 7 个字段:`anchor_id / score / reason / pass_or_not / reference_inputs /
negative_prompt / edit_prompt`,`pass_or_not ∈ {pass, fail}`。

`reason` 的前缀 tag 决定修复分支:

| tag | 分支 | 做法 |
|---|---|---|
| `[design_bug]` | `_repair_anchor_re_render` | 从 `edit_prompt` 里用正则抠出 `改成:"<新 prompt>"`,抠不出来直接抛错;带原 refs 重渲 |
| `[low_quality]` | `_repair_anchor_re_render` | prompt 不变,带原 refs 重渲 |
| `[render_drift]` | `_repair_anchor_image_edit` | 以当前图为 `source` + 原 refs,走 `kind="image_edit"`,prompt 用 `edit_prompt` |
| 其它 / 无 tag | — | 接受当前图,不修 |

每次修复的渲染本身重试 3 次(退避 60s × 次数),产物写
`<run>/anchors/repairs/<anchor_id>.repair_<n>.png`。修满 `max_repair_attempts` 或
validator 调用抛错都按"接受当前 URL"处理。

### 4.3 阶段三:render_segments

shot 之间并行(线程池 `RENDER_POOL_SIZE`,默认 8),**shot 内严格串行** ——
上一段 mp4 的尾帧就是下一段的首帧。

每个 segment(`_render_shot_chain`):

1. **缓存**:有 `.url` 就跳过,但仍会抽一次尾帧供下一段用
2. **首帧**:`first_frame_path == "shot_start_anchor"` 用 `anchor_urls[start_anchor]`,否则用上一段尾帧。
   首帧为空或不是 `http` 开头 → 直接抛错
3. **引用**:按角色 `portrait / supporting_portrait / place / prop` 从资产池解析 URL;
   `_BACKEND_EXCLUDE_ROLES` 允许按后端名排除某些角色
4. **mode 决策**:`RECA_FORCE_I2V=1` → `i2v` 并丢掉所有 refs;否则 `len(refs)==0 → i2v`,`else → r2v`
5. **截断 + ref hint**:按目标后端 `max_reference_images` 截断,再按后端拼一句参考图说明追加到 prompt 末尾
   (`happyhorse-1.0-r2v` → `[Image i]名字`;`wan2.7-r2v` / `wan2.7-i2v` → `图i=名字`)
6. `dispatch_segment` → 抽尾帧(`_extract_last_frame`,PyAV 解到最后一帧存 `<mp4>.tail.png` 并上传 OSS)
7. **立刻写 `.url` 缓存**(在 validator 之前),这样 validator 出问题时重跑不会重烧一次视频 API
8. 进 segment validator 循环(见下)
9. 循环结束后再写一次 `.url`(best-of-N 可能换了 URL)

**尾帧抽取失败**(PyAV 解出 0 帧)会抛 `RuntimeError`,整条 shot chain 失败。
任一 shot chain 失败 → `render_segments` 汇总抛 `RuntimeError`,其它 shot 的结果已进缓存。

### 4.4 segment validator 循环(可选,`--validate-segments`)

配置:

```python
SegmentValidatorParams(
    planner, story,
    enabled=True,
    max_repair_attempts=2,     # 最多 max+1 次判定、max 次重渲
    router_cfg=None,           # None ⇒ 每次都用 seed_reroll
    sp_micro_adjust_cfg=None,
    sp_replan_cfg=None,
    # ── 可插拔旋钮(全部有默认值,不动就是历史行为) ──────────────
    axis_pass_threshold=None,      # None ⇒ 0.6
    overall_pass_threshold=None,   # None ⇒ 0.7
    validator_model=None,          # None ⇒ qwen3-vl-plus
    video_sample_fps=None,         # None ⇒ 10
    allowed_strategies=("seed_reroll", "micro_adjust", "replan"),
    best_of_n=True,
    on_error="raise",              # "raise" | "accept"
)
```

| 旋钮 | 默认 | 作用 | env |
|---|---|---|---|
| `axis_pass_threshold` | 0.6 | 单轴通过线 | `RECA_SEGMENT_AXIS_THRESHOLD` |
| `overall_pass_threshold` | 0.7 | 加权总分通过线 | `RECA_SEGMENT_OVERALL_THRESHOLD` |
| `validator_model` | `qwen3-vl-plus` | 打分用的视觉模型 | `RECA_SEGMENT_VALIDATOR_MODEL` |
| `video_sample_fps` | 10 | 服务端抽帧率 | `RECA_SEGMENT_VALIDATOR_FPS` |
| `allowed_strategies` | 三个都允许 | router 可选策略的白名单;不在名单里的降级成第一个并打日志。**设成 `()` 就是只打分不重渲** | `RECA_SEGMENT_REPAIR_STRATEGIES` |
| `best_of_n` | `True` | 是否保存每次 attempt 快照并最后取最高分那次;`False` 时不产生快照,恒用最后一次 | `RECA_SEGMENT_BEST_OF_N=0` |
| `on_error` | `"raise"` | validator **调用本身**失败(API 4xx/5xx、JSON 解析炸)时的行为 | `RECA_SEGMENT_VALIDATOR_ON_ERROR` |

关闭整段校验只需 `segment_validator=None`(CLI 不加 `--validate-segments`),
`_render_shot_chain` 会整块跳过,零额外调用。

前置条件:`reconstruct_segment_planner_state` 能重建出该 shot 的 SP 状态;
重建失败则**该 shot 整体跳过 segment 校验**(打日志,不报错)。

**判定** `run_segment_validator(segment_spec, shot_start_state, segment_mp4_url, portrait_urls,
sp_state, shot_story_goal=, shot_visual_intent=)`:

- 走 DashScope 原生 `MultiModalConversation`(不是 OpenAI-compat),因为只有它接受 `file://`
  本地路径 —— 有的视频后端只落本地不给公网 URL
- 消息 = `system(校验 prompt)` + SP 尾部 4 条(纯文本)+ 当前轮
  `[{video: <url|file://>, fps:10}, {image: portrait}..., {text: user_prompt}]`
- 看的是**整段 mp4**,模型自己在时间轴上采样,不是单张尾帧
- 并发受 `SEGMENT_VALIDATOR_POOL_SIZE`(默认 2)限制,超时 `M12_SEGMENT_VALIDATOR_TIMEOUT_S`(默认 900s)

输出 9 字段:`segment_id / aesthetic_score / global_alignment_score /
action_consistency_score / overall_score / pass_or_not / reason / repair_action / edit_prompt`。
三个 axis 分和 overall 必须是 `[0,1]` 浮点,`repair_action ∈ {"", RegenerateUnit,
RepackPrompt, ReanchorState}`,否则抛 `SegmentValidatorError`。

| axis | 权重 | 看什么 |
|---|---|---|
| aesthetic | 0.25 | 构图 / 光照 / 清晰度 / artifact |
| global_alignment | 0.40 | end_state 兑现 / story_goal / visual_intent |
| action_consistency | 0.35 | motif 守恒 / identity 守恒 / 时间与物理合理 |

`overall_score = 0.25·A + 0.40·B + 0.35·C`。

**pass 判定只看数字**:`A ≥ 0.6 且 B ≥ 0.6 且 C ≥ 0.6 且 overall ≥ 0.7`。
模型自报的 `pass_or_not` 只做参考,代码不采信(`validate.py::_to_judgment`)。
阈值是模块常量 `_AXIS_PASS_THRESHOLD` / `_OVERALL_PASS_THRESHOLD`。

**不过时选修复策略**。`router_cfg` 为 `None` 就恒用 `seed_reroll`;否则问
`route_repair_strategy`(纯文本 LLM,看当前 prompt + 本次判定 + 历史判定):

| 策略 | 做法 | 成本 |
|---|---|---|
| `seed_reroll` | prompt 不动,`seed = (seed + (attempt+1)×7919) % 2147483647`,重渲 | 只多一次视频 API |
| `micro_adjust` | `micro_adjust_single_segment(sp_state, seg, judgment, cfg)` 只改这一段的 prompt(可选 end_state) | 一次 LLM + 一次视频 API |
| `replan` | `replan_segments_for_shot(...)` 重写整个 shot 的 segments,取回本段的新 prompt / end_state | 一次较贵 LLM + 一次视频 API |

配置缺失时**不静默降级**:router 返回 `replan` 而 `sp_replan_cfg is None` → 抛 `RuntimeError`;
返回 `micro_adjust` 而两个 cfg 都是 `None` → 抛 `RuntimeError`。router 自己调用失败
(`RouterError`)则退回 `seed_reroll`。运行期失败的退路:`replan` 抛错或返回结果里没有本段
→ 转 `micro_adjust`;`micro_adjust` 抛错 → 跳出循环,保留上一次成功的渲染。

**best-of-N**:每次非终局的 attempt 都把当前 mp4 复制成 `<mp4>.attempt<i>.mp4` 快照并记下分数。
循环结束时若 attempt 数 > 1,把 `overall_score` 最高的那次拷回正式路径,打
`[segment-best-of-N] restored attempt <i>`,再删掉其余快照。

**耗尽后不 fail-hard**:用满 `max_repair_attempts` 仍不过就接受 best-of-N 结果,打日志继续。

**validator 调用本身失败**(区别于"判定为 fail")由 `on_error` 决定:
`"raise"`(默认)向上冒,导致该 shot chain 失败 —— 此时 mp4 已经进了缓存,`--resume` 不会重渲;
`"accept"` 打一行 `[segment-validate] ... on_error=accept` 后保留当前渲染继续走。

> `validator/segment/validate.py::apply_segment_repair` 和
> `validator/anchor/validate.py::validate_and_repair` 是可独立调用的库函数,
> `pipeline.py` 走的是上面描述的 router / `_validate_and_repair_anchors` 路径,不调它们。

### 4.5 bridge 的提前分发

`transitions` 里 `mode=="bridge"` 的才渲。bridge 不等所有 shot 跑完:
某个 shot 的 chain 一结束,如果它有出向 bridge 且对端 shot 的 anchor 已就绪,
就立刻把任务提交给一个独立的 4 线程 executor(`_maybe_submit_outgoing_bridge`)。
`first_url` = 本 shot 末段的尾帧,`last_url` = 对端 shot 的 anchor。
`run_render` 在 `render_segments` 返回后统一收 future(`bridges-wait` 阶段)。

### 4.6 concat_final

按 shot 顺序排 clip,bridge 插在两个 shot 之间;每个 clip 优先用本地文件,没有才用 URL。

**丢首帧规则**(`trim`):首个 clip 不丢;shot 内非首段丢 1 帧;
一个 shot 的首段在其入向 transition 是 `bridge` 时丢 1 帧;bridge clip 自身丢 1 帧。
去掉的是接缝处重复的那一帧(`select='gte(n\,1)'` / `aselect='gte(n\,1)'`)。

**归一**:以第一个 clip 的 `(width, height, r_frame_rate)` 为准,每个 clip 加
`scale→pad→setsar=1→fps`;只有当**所有** clip 都在本地且都探到音轨时才走带音频的分支,
统一 `aresample=<第一个 clip 的采样率>`,否则整条链 `-an` 无音频。

输出编码固定 `libx264 -preset veryfast -crf 18 -pix_fmt yuv420p`,有音频时 `aac 192k`。
ffmpeg 非零退出会打印 stderr 末尾 30 行再抛 `RuntimeError`。

---

## 5. 缓存与 resume

唯一的缓存单位是 **`<output_path>.url` sidecar**:文本文件,内容是该 cell 的最终 URL。
判定"命中"要求 **产物文件和 sidecar 同时存在**且 sidecar 非空。

- 图像:命中直接复用;内容是本地路径时重传 OSS 后改写 sidecar
- segment:命中跳过渲染,但仍重新抽一次尾帧(下一段要用)
- bridge:命中直接复用

`_smoke.py --resume` 的语义是**跳过规划**:`render_plan.json` 已存在就不再调 planner,
直接读盘进 `run_render`;渲染层面的跳过完全由 sidecar 决定,跟 `--resume` 无关。

没有失败标记文件,也没有占位片段 —— 任何 cell 失败都是抛异常,不会用黑屏顶替。

---

## 6. 公共 API

```python
from videorlm.framework import (
    # 规划
    PARENT_SYSTEM_PROMPT, plan_skeleton,
    SEGMENT_PLANNER_SYSTEM_PROMPT, format_segment_planner_user_prompt,
    make_segment_planner_from_parent_state,
    plan_segments_for_shot, plan_segments_all,
    reconstruct_parent_state, reconstruct_segment_planner_state,
    # 编排
    ValidatorParams, SegmentValidatorParams,
    merge_into_planner_output, to_render_plan,
    render_segments, concat_final,
    run_planner, run_render,
    parse_json_block,
)
```

其余按需从子包取:

```python
from videorlm.framework.schemas   import Skeleton, RenderPlan, Shot, Segment, ShotSegments
from videorlm.framework.parsers   import JSONOutputParser, retry_agent_call, strip_markdown_fence, ParseError
from videorlm.framework.templates import ChatPromptTemplate
from videorlm.framework._common.fork  import sub_conversation_with_system_swap
from videorlm.framework._common.pools import (
    PLANNER_POOL_SIZE, ANCHOR_VALIDATOR_POOL_SIZE,
    SEGMENT_VALIDATOR_POOL_SIZE, RENDER_POOL_SIZE, pool_size_for_role,
)
from videorlm.framework.validator.segment import run_segment_validator, SegmentJudgment
from videorlm.framework.validator.segment.router import route_repair_strategy, RouterDecision
from videorlm.framework.segment_replanner import replan_segments_for_shot, micro_adjust_single_segment
```

## 7. 文件地图

```
framework/
├── pipeline.py                to_render_plan / run_planner / run_render / render_segments /
│                              concat_final / _render_image_dag / _validate_and_repair_anchors
├── schemas.py                 Skeleton / RenderPlan / Shot / Segment / ShotSegments / ...
├── parsers.py                 strip_markdown_fence / retry_agent_call / JSONOutputParser / ParseError
├── templates.py               ChatPromptTemplate(dollar / format / jinja2)
├── run_context.py             RunContext 数据类(当前无调用方)
├── prompts/
│   ├── shot_planner/system.md
│   ├── segment_planner/system.md
│   ├── segment_replanner/{system_replan.md, system_micro_adjust.md}
│   ├── validator_anchor/system.md
│   └── validator_segment/{system.md, router.md}
├── shot_planner/              plan.py(plan_skeleton) + prompts.py(装载 md)
├── segment_planner/           plan.py(fork / 并行 / 状态重建) + prompts.py
├── segment_replanner/         plan.py(replan / micro_adjust) + prompts.py
├── validator/
│   ├── anchor/                validate.py + prompts.py + _codex_vision.py
│   └── segment/               validate.py + router.py + prompts.py
├── _common/                   fork.py / pools.py
└── _scripts/                  _smoke.py / _web_server.py / _freeze_prompts.py /
                               _xprocess_semaphore.py / _test_*.py
```

> `run_context.py` 定义了 `RunContext`,但它 import 的 `framework.callbacks` 模块不存在,
> 直接 `import videorlm.framework.run_context` 会 `ModuleNotFoundError`。当前没有任何代码引用它。
