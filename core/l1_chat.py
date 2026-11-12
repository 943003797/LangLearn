"""第 1 课：最小对话 —— 单轮 / 多轮 / 流式。

核心认知：
    1. ChatModel 是**无状态**的，所谓"记忆"就是你自己把历史消息一起传进去。
    2. 三种消息：SystemMessage / HumanMessage / AIMessage。
    3. stream() 返回 chunk，chunk.text 是增量文本。

运行：uv run python -m core.l1_chat
"""

from langchain_core.messages import HumanMessage, SystemMessage

from common.config import get_model


def main() -> None:
    model = get_model()

    # # ---- 1) 单轮 ----
    # reply = model.invoke("用一句话解释 LangGraph 是什么")
    # print("[单轮]", reply.content)
    # print("       token 用量:", reply.usage_metadata)
    # print()

    # # ---- 2) 多轮：手动维护消息列表 ----
    # history = [SystemMessage("你是一个简洁的中文助手，回答不超过 40 字。")]
    # for question in ["LangChain 是什么？", "那它和 LangGraph 什么关系？"]:
    #     history.append(HumanMessage(question))
    #     ai = model.invoke(history)  # 把全部历史传进去
    #     history.append(ai)          # 关键：模型回复也要进历史
    #     print(f"[多轮] Q: {question}")
    #     print(f"       A: {ai.content}")
    # print(f"\n       历史条数：{len(history)}")
    # print()

    # ---- 3) 流式 ----
    print("[流式] ", end="", flush=True)
    for chunk in model.stream("默写湖心亭看雪"):
        print(chunk.text, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
