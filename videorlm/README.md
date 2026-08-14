# videorlm

把**一段中文剧情**端到端跑成**一条成片 mp4**。

规划由 LLM 完成(拆镜头 → 拆片段),渲染由外部图像/视频生成 API 完成,质检由 VLM 完成,
最后 ffmpeg 拼接。整个包不训练模型,也不需要 GPU(`ltx-2.3` 本地后端除外)。

```
story.txt  ──▶  planner.json  ──▶  render_plan.json  ──▶  final.mp4
             (LLM 规划)         (纯 Python 转换)      (渲染 + 质检 + 拼接)
```

---

## 端到端链路

```
story (中文剧情 txt)
  │
  │ ── plan_skeleton(parent_agent, story)
  ▼
skeleton  = portrait_plan / location_plan(含 props) / boundarys.boundary_anchors
            / shots / transitions
  │
  │ ── plan_segments_all(parent.state, skeleton, sp_cfg)    每个 shot 一个 fork,并行
  ▼
planner   = skeleton + segments{seg_id: {prompt, end_state, duration_s, ...}}
  │
  │ ── to_render_plan(planner, run_dir, seed=, video_resolution=)
  ▼
render_plan = 每个实体挂一个 image_request / segment_request / bridge_request
  │
  │ ── run_render(render_plan, out_path, validator=, segment_validator=)
  │
  ├─ ① _render_image_dag         portrait / location ▸ prop ▸ anchor,单 DAG 并发 16
  ├─ ② anchor validator          (可选) 逐 anchor 判定 → 重渲 or image_edit 修
  ├─ ③ render_segments           shot 之间并行,shot 内串行:上一段尾帧 = 下一段首帧
  │     └ segment validator      (可选) 每段渲完看整段 mp4 → router → 重渲 → best-of-N
  │     └ bridge inline dispatch shot 末段一渲完就提交它的 bridge,不等其它 shot
  ├─ ④ bridges-wait              收 bridge future
  └─ ⑤ concat_final              分辨率/帧率/音频采样率归一后 ffmpeg 拼接
  ▼
final.mp4 + summary.json
```

## 跑起来

```bash
cd /mnt/workspace/akide/code/unirlm-02

# 解释器:需要 dashscope / openai / pydantic / av / PIL / oss2 / httpx
/usr/local/bin/python3 -c "import dashscope, openai, pydantic, av, PIL, oss2, httpx; print('deps OK')"

# 密钥:放在 <repo>/.env,_smoke.py 启动时读进 os.environ
#   DASHSCOPE_API_KEY / DASHSCOPE_API_KEYS(逗号分隔多 key)
#   OPENAI_API_KEY / OPENAI_BASE_URL
#   oss_AccessKey_ID / oss_AccessKey_Secret / oss_bucket / region

/usr/local/bin/python3 -m videorlm.framework._scripts._smoke \
    --story input_examples/true_input/example_01.txt \
    --out-dir videorlm/outputs/demo/ex01 \
    --label ex01 \
    --segments --render --validate --validate-segments \
    --backend happyhorse --video-resolution 1920x1080 --seed 0
```

产物落在 `--out-dir` 下:

```
<out-dir>/
├── _frozen_prompts/       本次运行的 prompt 源码快照 + freeze_meta.json
├── skeleton.json          plan_skeleton 输出
├── planner.json           skeleton + segments
├── render_plan.json       run_render 的输入
└── run/
    ├── portraits/ locations/ props/ anchors/    *.png + *.png.url
    ├── segments/                                *.mp4 + *.mp4.url + *.tail.png
    ├── bridges/                                 *.mp4 + *.mp4.url
    ├── logs/trace.jsonl
    ├── final.mp4
    ├── summary.json
    └── _backend_info.json
```

`*.url` sidecar 是断点续跑的依据:带 `--resume` 再跑时,有 `.url` 的 cell 直接跳过。

## 目录

```
videorlm/
├── framework/                 编排层 — 规划 / 转换 / 渲染 / 质检
│   ├── pipeline.py            唯一 orchestrator:to_render_plan / run_planner / run_render
│   ├── schemas.py             Pydantic 单一真源(Skeleton / RenderPlan / Segment / ...)
│   ├── parsers.py             JSON 解析 + 带重试的 agent 调用
│   ├── templates.py           ChatPromptTemplate — 从 .md 文件读 prompt
│   ├── prompts/               所有 system prompt 的正文(按角色分目录)
│   ├── shot_planner/          plan_skeleton
│   ├── segment_planner/       plan_segments_for_shot / plan_segments_all / 状态重建
│   ├── segment_replanner/     replan_segments_for_shot / micro_adjust_single_segment
│   ├── validator/anchor/      anchor 图像质检 + 两种修复
│   ├── validator/segment/     segment 视频质检(3 轴) + 修复策略 router
│   ├── _common/               fork.py(子会话) / pools.py(4 个角色池)
│   └── _scripts/              _smoke.py(CLI) / _web_server.py / _freeze_prompts.py /
│                              _xprocess_semaphore.py
├── backends/                  后端层 — 与外部 API 的唯一接触面
│   ├── media/interface/       3 个 Request + 3 个 dispatch + 注册表 + Protocol
│   ├── media/impl/            dashscope / openai / local(ltx-2.3)
│   ├── llm/agents/            4 个 Agent 类(codex / openai_compat / openai_messages / qwen)
│   ├── _common/               KeyPool / RateLimiter / providers / retry / timeout /
│   │                          trace / oss_publisher / env / concurrency
│   └── tests/                 pytest
├── scripts/                   批量跑批脚本 + 各类探针
├── outputs/                   历次运行产物(按 version_* 分)
└── docs/                      见下
```

## 文档

| 文档 | 内容 |
|---|---|
| [docs/framework.md](docs/framework.md) | 编排层:三个 planner 角色、两个 validator、两个 JSON schema、run_render 五阶段、缓存与拼接规则 |
| [docs/backends.md](docs/backends.md) | 后端层:3 个 Request × 3 个 dispatch、已注册后端能力表、KeyPool / 限流 / 重试 / OSS |
| [docs/agents.md](docs/agents.md) | LLM agent 层:4 个 Agent 类、配置字段、fork 继承策略、角色池 |
| [docs/operations.md](docs/operations.md) | 运行:CLI 参数、批量跑、env 全表、resume 语义、web server、错误对照 |
