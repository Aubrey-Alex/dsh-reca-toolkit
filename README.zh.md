# ReCA Director

ReCA Director 是运行在 DeepSeek Harness 上的长视频创作 Skill。用户用自然语言描述故事，DSH 负责交互和任务调用，ReCA 负责规划、素材生成、Wan3.0 渲染、视觉审计、修复、恢复和最终交付。

## 0.4.0 更新

- DSH、Gateway 和 ReCA 全链路支持可选首帧与具名参考图。
- GPT Image 2 默认负责人物、场景、anchor 和图片修复。
- Wan3.0 使用与实际接口媒体组合兼容的纯 R2V 连续生成路由。
- GPT Responses 审计支持跨 Gateway 子进程限流、重试和紧凑上下文。
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

```bash
bash scripts/install.sh
# 编辑 .env，填入自己的 provider 配置
bash scripts/doctor.sh
bash scripts/start-gateway.sh
```

在安装了 DSH 的环境中：

```bash
dsh plugin --profile web add "file:$PWD/dsh-plugin"
dsh web
```

DSH 对话模型配置可直接复制
[`configs/dsh-settings.example.yaml`](configs/dsh-settings.example.yaml) 到
`$DSH_HOME/settings.yaml`，并在启动 DSH 的进程中导出
`RECA_DSH_DEEPSEEK_API_KEY`。示例使用 DSH 的 `llm-pi-ai` OpenAI-compatible
路由（团队网关支持 `/v1/chat/completions`）；ReCA 内部 planner 仍独立使用
原有的 Messages adapter。

## DSH 工具

新接口为 `reca_create_video`、`reca_get_status`、`reca_cancel`、`reca_resume`、`reca_list_runs` 和 `reca_get_artifact`。旧的 `reca_start`、`reca_status` 仍保留兼容。

每次运行分别返回 Gateway 状态、ReCA 阶段、`video_state`、`audit_state` 和 artifact manifest。生成成功不代表审计成功，审计状态会明确返回 `audited`、`audit_skipped`、`audit_failed` 或 `audit_repaired`。

`reca_create_video` 除文字故事外还支持可选的 `first_frame` 和 `reference_images`。提供首帧时，它会直接作为第一个镜头的起始 anchor；参考图会进入 anchor 规划，并按照 ReCA 的 segment contract 转发到视频段。Wan3.0 只适配 provider 输入，不改变 ReCA 的 planner 和串行尾帧链：I2V 将当前帧作为唯一参考图；R2V 将当前帧放在 `reference_image[0]`，后面最多附加三张 planner 选择的人物、场景或道具参考图，并用 R2V 前缀明确要求从第一张图开始。Bridge 仍使用真实首尾帧。由于 Wan3.0 不支持把硬首帧和额外参考图组合提交，R2V 的开始约束属于软约束。没有提供图片时，ReCA 继续自动生成角色、场景和 anchor。

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
