# -*- coding: utf-8 -*-
"""DeepSeek 对话模型唯一入口。

为什么独立成文件：全工程所有"生成"（RAG 回答、Agent 思考）都共用这一个 LLM，
集中管理 model / temperature，改一处全工程生效。

用法：
    from app.llm import get_llm
    llm = get_llm()
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

_llm = None


def get_llm() -> ChatDeepSeek:
    """模块级单例：重复调用不重建，省去重复初始化的开销。"""
    global _llm
    if _llm is None:
        _llm = ChatDeepSeek(
            model="deepseek-chat",          # Agent 主线用 chat；reasoner 工具调用受限
            temperature=0,                  # 科普问答求稳，0 减少幻觉
            api_key=os.getenv("DEEPSEEK_API_KEY"),
        )
    return _llm
