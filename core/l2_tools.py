"""第 2 课：工具调用 —— @tool / bind_tools / 手动执行 tool_calls。

这一课是整个 Agent 的地基，务必手写一遍，不要用 create_agent 跳过。

核心认知：
    1. 模型**不会执行**工具，它只返回 tool_calls（名字 + 参数 + id）。
    2. 执行工具、把结果包成 ToolMessage 回传，都是你的代码干的。
    3. 每个 tool_call 必须配一条同 tool_call_id 的 ToolMessage，
       否则下一轮请求会被 API 直接拒掉（这是最常见的 400 原因）。

运行：uv run python -m core.l2_tools
"""

from typing import Annotated

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

from common.config import get_model

# 假数据，避免依赖真实天气 API
_FAKE_WEATHER = {"北京": "晴，18°C", "上海": "多云，22°C"}


@tool
def get_weather(city: Annotated[str, "城市名，例如 北京"]) -> str:
    """查询指定城市的当前天气。"""
    return _FAKE_WEATHER.get(city, f"{city}：暂无数据")


@tool
def add(a: int, b: int) -> int:
    """两个整数相加。"""
    return a + b


TOOLS = {t.name: t for t in (get_weather, add)}


def main() -> None:
    model = get_model().bind_tools([get_weather, add])

    messages: list = [HumanMessage("北京天气怎么样？顺便算一下 17+25")]

    # 第 1 轮：模型返回工具调用请求
    ai = model.invoke(messages)
    print("模型请求的工具调用：")
    for tc in ai.tool_calls:
        print(f"  - {tc['name']}({tc['args']})  id={tc['id']}")
    print()

    messages.append(ai)

    # 我们自己执行工具，并把结果回传
    for tc in ai.tool_calls:
        result = TOOLS[tc["name"]].invoke(tc["args"])
        print(f"执行 {tc['name']} -> {result!r}")
        messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    print()

    # 第 2 轮：模型基于工具结果生成最终回答
    final = model.invoke(messages)
    print("最终回答：", final.content)


if __name__ == "__main__":
    main()
