import os, json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from dotenv import load_dotenv
from langchain.agents import create_agent

load_dotenv()

MODEL = os.getenv("MODEL", "deepseek:deepseek-chat")

@tool
def get_time()->str:
    """Return the current time."""
    return '2026-11-18 00:00:00';

# model = init_chat_model(MODEL)
# model = model.bind_tools([get_time])

agent = create_agent(
    model=init_chat_model(MODEL),
    tools=[get_time],
    system_prompt="需要查询时间时，请使用 get_time 工具。",
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "请调用 get_time 工具查询时间"}]
})
print(result)

# ai = model.invoke("请调用 get_time 工具查询时间")
# print(json.dumps(ai.model_dump(), ensure_ascii=False, indent=2))