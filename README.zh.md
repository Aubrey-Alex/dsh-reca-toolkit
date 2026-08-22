# ReCA Director

ReCA Director 是运行在 DeepSeek Harness 上的长视频创作 Skill。用户用自然语言描述故事，DSH 负责交互和任务调用，ReCA 负责规划、素材生成、Wan3.0 渲染、视觉审计、修复、恢复和最终交付。

## 0.4.0 更新

- GPT Image 2 默认负责人物、场景、anchor 和图片修复。
- Wan3.0 使用与实际接口媒体组合兼容的纯 R2V 连续生成路由。
- GPT Responses 审计支持跨 Gateway 子进程限流、重试和紧凑上下文。
- DSH 插件会按需拉起本仓库的本地 runtime，对话里不需要用户再开 Gateway。
- 新增由真实运行产物构建、不会公开付费 API 的静态回放 Demo 模板。

## 运行结构

```text
DSH Web / CLI
    -> ReCA Director Skill
    -> Gateway :8787
    -> ReCA Core
    -> final.mp4 + plan + audit + artifact manifest
```

DSH 不负责拆分镜头或直接调用 provider。Gateway 只管理进程、队列、恢复和 HTTP；ReCA 是视频业务状态和产物 manifest 的唯一来源。

## 安装

在仓库根目录：

```bash
bash scripts/install.sh
# 编辑 .env，填入 planner / 图像 / 视频密钥
bash scripts/doctor.sh
dsh plugin --profile web add "file:$PWD/dsh-plugin"
dsh web
```

之后只在 DeepSeek Harness 里描述故事。Skill 会调用 `reca_create_video`、
轮询 `reca_get_status`，再用 `reca_get_artifact` 取回成片。插件自己拉起本地
runtime。`bash scripts/start-gateway.sh` 只给不走 DSH 插件的 HTTP 客户端用。

DSH 对话模型配置可直接复制
[`configs/dsh-settings.example.yaml`](configs/dsh-settings.example.yaml) 到
`$DSH_HOME/settings.yaml`，并在启动 DSH 的进程中导出
`RECA_DSH_DEEPSEEK_API_KEY`。示例使用 DSH 的 `llm-pi-ai` OpenAI-compatible
路由（团队网关支持 `/v1/chat/completions`）；ReCA 内部 planner 仍独立使用
原有的 Messages adapter。

## DSH 工具

新接口为 `reca_create_video`、`reca_get_status`、`reca_cancel`、`reca_resume`、`reca_list_runs` 和 `reca_get_artifact`。旧的 `reca_start`、`reca_status` 仍保留兼容。

每次运行分别返回 Gateway 状态、ReCA 阶段、`video_state`、`audit_state` 和 artifact manifest。生成成功不代表审计成功，审计状态会明确返回 `audited`、`audit_skipped`、`audit_failed` 或 `audit_repaired`。

`reca_create_video` 接受故事文本以及 `duration`、`resolution`、`style`、
`aspect_ratio`、`backend`、`enable_audit` 和 `seed`。默认分辨率是 `1280x720`。
Wan3.0 只适配 provider 输入，不改变 ReCA 的 planner 和串行尾帧链：I2V 将当前
帧作为唯一参考图；R2V 将当前帧放在 `reference_image[0]`，后面最多附加三张
planner 选择的人物、场景或道具参考图，并用 R2V 前缀明确要求从第一张图开始。
Bridge 仍使用真实首尾帧。由于 Wan3.0 不支持把硬首帧和额外参考图组合提交，
R2V 的开始约束属于软约束。

## 真实运行回放 Demo

`demo/` 是静态产品回放页面，不是公开的视频生成接口。可从任意已经完成的真实任务生成回放数据：

```bash
python3 scripts/build_replay_manifest.py .dsh_runs/<run_id>
python3 scripts/build_demo_bundle.py .dsh_runs/<run_id>
python3 -m http.server 8080 --directory demo
```

页面使用真实的用户请求、Planner、Render Plan、事件、审计和产物清单。成片、运行日志以及针对某次任务生成的 replay manifest 都不进入 Git，应发布到独立 Demo 部署或对象存储。`scripts/generate_first_frames.py` 和 `scripts/monitor_batch.py` 可用于准备、监控精选的多任务 Demo 批次，密钥仍然只从进程环境读取。

## 安全和来源

真实 key 只放在 `.env`，不进入工具参数、模型上下文、事件日志或 Git。ReCA 快照来源和集成改动见 [RECA_INTEGRATION_PATCHES.md](RECA_INTEGRATION_PATCHES.md)。
