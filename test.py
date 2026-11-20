from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from common.config import MODEL

from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel

class TimeResponse(BaseModel):
    Time: str

@tool
def get_time() -> str:
    """获取当前时间，格式为 YYYY-MM-DD HH:MM:SS"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

agent = create_agent(
    # DeepSeek 思考模式不支持 ToolStrategy 使用的强制工具调用。
    model=init_chat_model(MODEL, extra_body={"thinking": {"type": "disabled"}}),
    system_prompt="""
    用中文回答。
    """,
    tools=[get_time]

)

messages = [HumanMessage("你是什么模型?")]
messages.append(HumanMessage("你能告诉我当前时间吗？"))

result = agent.invoke({"messages": messages})
history = result["messages"]

history.append(HumanMessage("默写1-9。"))
result = agent.invoke({"messages": history})

print(result["messages"])
