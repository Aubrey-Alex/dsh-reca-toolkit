# agents — LLM 会话层

`videorlm/backends/llm/agents/` 提供**有状态、可分叉**的对话句柄。framework 里所有
planner / validator / router 都是这层的调用方。

```python
from videorlm.backends.llm.agents import (
    Agent, AgentMessage, AgentState, AgentCapabilities, AgentError, slot,
    OpenAICompatibleAgent, OpenAICompatibleConfig,
    DashScopeQwenAgent, QwenAgent, QwenConfig,
    OpenAIMessagesAgent,
    CodexAgent, CodexConfig,
)
```

---

## 1. 四个实现

| 类 | 传输 | PROVIDER_NAME | 视觉 | 说明 |
|---|---|---|---|---|
| `OpenAICompatibleAgent` | openai SDK `chat.completions.create(stream=True)` | `openai` | 图 + 视频 | 通用 OpenAI 兼容端点 |
| `DashScopeQwenAgent` | 同上 | `dashscope` | 图 + 视频 | 上者的子类,只把 provider 换成 dashscope(从而走 DashScope 的多 key 轮转) |
| `OpenAIMessagesAgent` | httpx 直接 `POST {base_url}/v1/messages`,解析 SSE | `openai` | 文本 | Anthropic Messages 形状,支持 extended thinking |
| `CodexAgent` | `codex-acp` 子进程(NDJSON JSON-RPC) | — | 文本 | 本地 codex 会话,`cwd` 下的 `AGENTS.md` / `CLAUDE.md` 会被原生读取 |

**`QwenAgent` 不是类,是工厂函数**:

```python
QwenAgent(config)   # base_url 含 dashscope / aliyuncs.com,或为空 → DashScopeQwenAgent
                    # 其它任何 base_url                          → OpenAICompatibleAgent
```

所以 `with QwenAgent(cfg) as agent:` 里拿到的可能是两个类中的任一个;报错信息和并发槽
都会体现真正的那个类。要做 `isinstance` 判断请用两个具体类。

---

## 2. 五方法契约

所有 agent 都实现同一套接口,可以互换:

| 方法 | 行为 |
|---|---|
| `start()` | 打开底层会话。幂等。有 `system_prompt` 时把 `AgentMessage(role="system", ...)` 放进 `state["messages"]` 头部 |
| `close()` | 关闭。幂等 |
| `prompt(text) -> str` | 一轮对话。先取并发槽,再发。user 与 assistant 两条消息都会追加进 state |
| `load_session(thread_id)` | 绑定到已有会话 id。HTTP 类后端是无状态的,这里只是改标签;codex 会真的 `session/load` |
| `fork() -> Agent` | 深拷贝 state 造一个新 agent。两支后续互不污染 |

支持 `with agent: ...`(`__enter__`/`__exit__` 即 `start`/`close`)。

三种输入形态:

```python
agent.prompt(text)                              # 纯文本
agent.prompt_with_images(text, image_urls)      # 文本 + N 张图
agent.prompt_with_video(text, video_url, image_urls=[])  # 文本 + 1 段视频 + N 张图
```

后两个的默认实现**直接抛 `AgentError`** —— 给纯文本后端传图会立刻报错,而不是静默丢掉。
`OpenAICompatibleAgent` / `DashScopeQwenAgent` 实现了这两个;
`OpenAIMessagesAgent` 与 `CodexAgent` 没实现。
`image_urls` 传空列表时 `prompt_with_images` 自动退回 `prompt`;
`video_url` 为空时 `prompt_with_video` 退回 `prompt_with_images`。

> framework 的 segment validator **不走这条路径** —— 它需要传本地 `file://` 视频,
> 只有 DashScope 原生 SDK 接受,所以那里直接调 `MultiModalConversation`,
> 见 [framework.md §4.4](framework.md)。

---

## 3. 配置

三层继承:`AgentConfig`(通用)→ `QwenConfig` / `OpenAIMessagesConfig`。全部是 frozen
dataclass,改用 `cfg.with_overrides(**kw)` 生成新实例。

### AgentConfig

| 字段 | 默认 | 说明 |
|---|---|---|
| `model` | `""` | |
| `api_key` / `base_url` | `""` | 留空时 `start()` 从绑定的 provider 里解析 |
| `provider` | `""` | 非空则覆盖类上的 `PROVIDER_NAME`,决定用哪个 KeyPool 和分类器 |
| `system_prompt` | `None` | |
| `temperature` | 0.5 | |
| `max_tokens` | `None` | |
| `request_timeout_s` | 180.0 | |
| `sdk_max_retries` | 0 | 固定 0:SDK 自带重试会绑死同一个 key,和 KeyPool 轮转打架 |
| `role` | `""` | 角色池 key,见 §5 |
| `max_concurrency` | 8 | 该槽的容量 |
| `inline_images` | False | True 时把图片 URL 在本机抓下来转成 base64 `data:` URI 内联,适用于上游网关取不到内网 OSS 的场景 |
| `video_sample_fps` | 10 | `prompt_with_video` 的服务端抽帧率,DashScope 接受 `[0.1, 10]` |

### QwenConfig(DashScope)

`model` 默认改成 `"qwen3.6-max-preview"`;新增 `enable_thinking: bool = False` ——
True 时 `chat.completions` 带 `extra_body={"enable_thinking": True}`,
流里的 `reasoning_content` 会被静默丢弃,只保留最终 content。

### OpenAIMessagesConfig(Anthropic Messages)

新增 `thinking_budget_tokens: int = 0`,>0 时请求体带
`"thinking": {"type": "enabled", "budget_tokens": n}`。

### CodexConfig

独立一套(不继承 `AgentConfig`):`binary_path`(默认 `codex-acp`)、`cwd`、`env`、
`request_timeout_s=600`、`startup_timeout_s=30`、`max_concurrency=8`、`system_prompt`。

---

## 4. 状态

```python
class AgentState(TypedDict, total=False):
    messages: list[AgentMessage]   # 每轮 prompt 追加 user + assistant
    thread_id: NotRequired[str]
```

`AgentMessage(role, content)` 的 role 只接受 `system` / `user` / `assistant`,
其它值(含 `tool`)抛 `ValueError`。frozen dataclass,深拷贝安全。

HTTP 类后端服务端无状态,**每轮把完整消息列表重发一遍** —— 所以 state 就是全部真相,
可以直接序列化落盘再恢复:

```python
snapshot = {"thread_id": agent.state.get("thread_id"),
            "messages": [asdict(m) for m in agent.state["messages"]]}
...
state = {"thread_id": d["thread_id"],
         "messages": [AgentMessage(**m) for m in d["messages"]]}
agent = QwenAgent(cfg, state=state); agent.start()
```

### fork() 与 sub_conversation_with_system_swap 的区别

两件不同的事,别混:

| | `agent.fork()` | `sub_conversation_with_system_swap(state, new_system, ...)` |
|---|---|---|
| 在哪 | `agents/base.py` | `framework/_common/fork.py` |
| system prompt | 原样保留 | **换成新的** |
| 历史 | 全量深拷贝 | 按策略切片(`all` / `first_pair_only` / `last_n_pairs` / `none`,或 `inherited_msg_count=N`) |
| thread_id | 沿用 | 一定新生成 `sub-<uuid12>` |
| 典型用途 | 同一个角色开两条分支 | parent → segment_planner / validator 换角色 |

framework 里换角色一律用后者,细节见
[framework.md §2.2](framework.md)。

---

## 5. 并发:角色池

`prompt()` 外面套一个按名字索引的 `BoundedSemaphore`,名字和容量来自
`capabilities()`:

```python
slot_key = f"role:{role}" if role else model      # OpenAICompatibleAgent / DashScopeQwenAgent
slot_key = "codex"                                # CodexAgent
cap      = config.max_concurrency
```

也就是说 **`role` 非空时按角色共享槽位,与用什么模型无关**;`role` 为空则退回按模型名分槽。

framework 定义了四个角色和它们的默认容量(`framework/_common/pools.py`):

| 常量 | role 字符串 | env | 默认 | 覆盖范围 |
|---|---|---|---|---|
| `PLANNER_POOL_SIZE` | `planner` | `RECA_PLANNER_POOL_SIZE` | 8 | shot_planner + segment_planner + replanner |
| `ANCHOR_VALIDATOR_POOL_SIZE` | `anchor_validator` | `RECA_ANCHOR_VALIDATOR_POOL_SIZE` | 4 | anchor 校验 |
| `SEGMENT_VALIDATOR_POOL_SIZE` | `segment_validator` | `RECA_SEGMENT_VALIDATOR_POOL_SIZE` | 2 | segment 校验(VL 调用慢且贵) |
| `RENDER_POOL_SIZE` | `render` | `RECA_RENDER_POOL_SIZE` | 8 | `render_segments` 的 shot-chain 线程池 |

`pool_size_for_role(role)` 做映射,未知 role 返回 8。
构造 config 时把 `role=` 和 `max_concurrency=pool_size_for_role(role)` 一起传,
否则同角色的不同实例会对容量有分歧 —— **同名槽的容量由第一个创建者定死**,后来者的
`max_concurrency` 会被忽略。

四个常量在 import 时读 env,所以要覆盖就在启动进程前 export。

想给非 `prompt()` 的代码路径也占同一份预算,直接用 `slot`:

```python
from videorlm.backends.llm.agents import slot
with slot("role:planner", 8):
    ...
```

跨进程再加一层限流见 [operations.md §6](operations.md)。

---

## 6. 鉴权

`start()` 时按 `config.provider or 类的 PROVIDER_NAME` 找 provider,
`api_key` / `base_url` 为空就从该 provider 解析(env 列表见
[backends.md §7.1](backends.md))。找不到已注册的 provider 会抛 `AgentError`。

真正发请求时,`_chat_create` 把 SDK 调用包进 `with_key(PROVIDER_NAME, ...)`,
于是拿到多 key 轮转 + 分级 cooldown + EWMA 健康度。该 provider 一个 key 都没配时
退回直接用 `config.api_key` 调(本地 / 单 key 场景)。

---

## 7. 常见错误

| 症状 | 含义 |
|---|---|
| `AgentError: agent is closed` | 对已 `close()` 的 agent 调 `prompt()` / `fork()` |
| `AgentError: ... _prompt_with_images not implemented` | 给纯文本后端(`OpenAIMessagesAgent` / `CodexAgent`)传了图 |
| `AgentError: ... _prompt_with_video not implemented` | 同上,传了视频 |
| `ValueError: AgentMessage.role='tool' invalid` | 只接受 system / user / assistant |
| provider 未注册的 `AgentError` | `config.provider` 或 `PROVIDER_NAME` 拼错,或对应模块没被 import |
| 并发上不去 | 多个 agent 用了同一个 `role`,但第一个创建者的 `max_concurrency` 偏小 |

## 8. 文件地图

```
backends/llm/agents/
├── base.py                 Agent ABC / AgentMessage / AgentState / AgentCapabilities /
│                           AgentError / slot()
├── config.py               AgentConfig
├── openai_compat/agent.py  OpenAICompatibleAgent(流式 chat.completions + 多模态)
├── openai_messages/        OpenAIMessagesAgent + OpenAIMessagesConfig(/v1/messages)
├── qwen/                   DashScopeQwenAgent + QwenConfig + QwenAgent 工厂
└── codex/                  CodexAgent + CodexConfig + acp.py(子进程 JSON-RPC)
```
