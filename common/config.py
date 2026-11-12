"""公共配置：加载 .env、创建 ChatModel。

注意一个 v1 的经典坑：
    MODEL 里的 "provider:model" 前缀只在 init_chat_model 调用时解析一次。
    之后如果通过 configurable 或别的途径换模型，必须传裸模型名
    （例如 deepseek-reasoner，而不是 deepseek:deepseek-reasoner），
    否则整串会被当成模型名原样发给 API。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODEL = os.getenv("MODEL", "deepseek:deepseek-chat")


def require_api_key() -> None:
    """没有 key 时给出可直接照做的提示，而不是让 SDK 抛一个看不懂的错。"""
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise SystemExit(
            "缺少 DEEPSEEK_API_KEY。\n"
            f"  1) 复制配置：cp {ROOT}/.env.example {ROOT}/.env\n"
            "  2) 填入你的 key 后重跑。"
        )


def get_model(*, temperature: float = 0.0):
    """按 .env 里 MODEL 的配置创建 ChatModel。"""
    require_api_key()
    from langchain.chat_models import init_chat_model

    return init_chat_model(MODEL, temperature=temperature)
