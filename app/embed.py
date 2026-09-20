# -*- coding: utf-8 -*-
"""Embedding 唯一入口：SiliconFlow 云端 BAAI/bge-m3（OpenAI 兼容）。

关键点：
- DeepSeek 官方 API 没有 embedding 接口，故向量化走 SiliconFlow；
- 必须用模块级单例：建库与查询若各 new 一个实例，可能因连接/配置不同
  导致"维度不一致"或"查询不到"的隐性 bug。
- bge-m3：1024 维，中文效果好，单条最长 8192 token。

用法：
    from app.embed import get_embeddings
    emb = get_embeddings()
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

_embeddings = None


def get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model="BAAI/bge-m3",
            base_url="https://api.siliconflow.cn/v1",   # OpenAI 兼容端点
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            # ⚠️ 关键参数：langchain-openai 默认 check_embedding_ctx_length=True，
            # 会先用 tiktoken 把文本切成 token ID 数组再发给 API。OpenAI 官方
            # 端点能正确处理 token 数组，但 SiliconFlow 不行——返回的向量与
            # 直接发字符串完全不同（实测同一文本余弦相似度仅 0.31，应为 1.0），
            # 表现为"召回全乱"：明明库里有的内容排到 300 名开外。
            # 设为 False 后原样发送字符串，实测与裸调 API 的向量完全一致。
            # （2026-09-09 排查：这是本项目此前"embedding 语义错乱"的真正根因）
            check_embedding_ctx_length=False,
        )
    return _embeddings
