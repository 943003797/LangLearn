# LangChain / LangGraph（Python）学习工作区

按「原语 → 编排 → 工程化 → 部署」四阶段推进。当前进度：**第 1 阶段**。

## 环境

```bash
cp .env.example .env      # 填入 DEEPSEEK_API_KEY
uv sync                   # 以 editable 方式安装本项目

uv run core/l1_chat.py    # 直接跑也可以
uv run python -m core.l1_chat   # 等价写法
```

> 本项目在 `pyproject.toml` 里声明了 `[build-system]` + `[tool.setuptools] packages`，
> `uv sync` 后 `common` / `core` 已装进 `.venv`，所以两种写法都能 import。
> 若报 `ModuleNotFoundError: No module named 'common'`，说明没跑过 `uv sync`，
> 或者用了系统 python 而不是 `uv run`。

## 第 1 阶段：LangChain 原语

| 文件 | 内容 | 学完应该能回答 |
|---|---|---|
| `core/l1_chat.py` | 单轮 / 多轮 / 流式 | 模型的"记忆"是谁维护的？`invoke` 和 `stream` 返回什么？ |
| `core/l2_tools.py` | `@tool` + `bind_tools` + 手动执行 `tool_calls` | 模型到底执行了工具吗？为什么每个 `tool_call` 都要配一条 `ToolMessage`？ |
| `core/l3_structured.py` | Pydantic 结构化输出 | `with_structured_output` 返回的是字符串吗？ |
| `core/l4_rag.py` | 切分 / 向量化 / 检索 / 拼提示词 | RAG 效果的瓶颈通常在哪一步？ |

**阶段验收**：自己加一个工具（例如查汇率），并让 `l2_tools.py` 里的模型在一次对话里调用它。

## 第 2 阶段：LangGraph 编排（下一步）

1. `StateGraph` + `TypedDict` 定义 state，节点是纯函数
2. 条件边 —— 手写 ReAct 循环（不用 `create_agent`）
3. `MessagesState` + `MemorySaver` + `thread_id` → 多轮记忆
4. `interrupt()` + `Command(resume=...)` → 人工审批
5. `stream_mode="messages" | "updates"` → 流式输出
6. 最后用 `create_agent` 对照自己手写的版本

## 第 3 / 4 阶段

- 工程化：`SqliteSaver`、`store`（跨线程记忆）、LangSmith tracing、pytest + `FakeListChatModel`
- 部署：`langgraph.json` + `langgraph dev`，前端可接你熟悉的 Vue `useStream`

## 已踩过的坑（提前避开）

- `MODEL` 里的 `provider:` 前缀只解析一次，后续换模型要传**裸模型名**
- DeepSeek 无 embedding 接口，RAG 需换 OpenAI / DashScope 等
- 视觉模型只有 `deepseek-flash`；`deepseek-v4-pro` 会**静默丢图**
