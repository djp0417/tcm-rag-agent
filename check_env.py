# -*- coding: utf-8 -*-
"""Step 0 连通性测试：确认 langchain/langgraph 版本 + DeepSeek + SiliconFlow 都可用。

用法：
    conda activate rag-agent
    python check_env.py

预期输出：
    langchain 1.4.x / langgraph 1.2.x / langchain-core 1.6.x
    DeepSeek 回复：……（一句话自我介绍）
    embedding 维度: 1024
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")          # 从工程根目录读 .env

# langchain/langgraph 等包未必导出 __version__ 属性，
# 统一用 importlib.metadata 从「已安装发行版」读版本号，最可靠。
from importlib.metadata import version   # noqa: E402

import langchain                        # noqa: E402


def main() -> None:
    print(f"langchain: {version('langchain')} | "
          f"langgraph: {version('langgraph')} | "
          f"core: {version('langchain-core')}")

    if not os.getenv("DEEPSEEK_API_KEY"):
        raise SystemExit("缺少 DEEPSEEK_API_KEY，请先按 .env.example 配置 .env")

    from langchain_deepseek import ChatDeepSeek
    llm = ChatDeepSeek(model="deepseek-chat",
                       temperature=0,
                       api_key=os.getenv("DEEPSEEK_API_KEY"))
    print("DeepSeek 回复:", llm.invoke("用一句话介绍你自己").content)

    if not os.getenv("SILICONFLOW_API_KEY"):
        raise SystemExit("缺少 SILICONFLOW_API_KEY，请先按 .env.example 配置 .env")

    from langchain_openai import OpenAIEmbeddings
    emb = OpenAIEmbeddings(model="BAAI/bge-m3",
                           base_url="https://api.siliconflow.cn/v1",
                           api_key=os.getenv("SILICONFLOW_API_KEY"))
    print("embedding 维度:", len(emb.embed_query("你好，测试一下")))


if __name__ == "__main__":
    main()
