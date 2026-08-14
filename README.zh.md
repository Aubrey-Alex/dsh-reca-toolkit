# ReCA Director

ReCA Director 是运行在 DeepSeek Harness 上的长视频创作 Skill。用户用自然语言描述故事，DSH 负责交互和任务调用，ReCA 负责规划、素材生成、Wan3.0 渲染、视觉审计、修复、恢复和最终交付。

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

## DSH 工具

新接口为 `reca_create_video`、`reca_get_status`、`reca_cancel`、`reca_resume`、`reca_list_runs` 和 `reca_get_artifact`。旧的 `reca_start`、`reca_status` 仍保留兼容。

每次运行分别返回 Gateway 状态、ReCA 阶段、`video_state`、`audit_state` 和 artifact manifest。生成成功不代表审计成功，审计状态会明确返回 `audited`、`audit_skipped`、`audit_failed` 或 `audit_repaired`。

## 安全和来源

真实 key 只放在 `.env`，不进入工具参数、模型上下文、事件日志或 Git。ReCA 快照来源和集成改动见 [RECA_INTEGRATION_PATCHES.md](RECA_INTEGRATION_PATCHES.md)。
