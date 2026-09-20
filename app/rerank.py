# -*- coding: utf-8 -*-
"""重排（Rerank）封装：SiliconFlow 云端 BAAI/bge-reranker-v2-m3 交叉编码器。

为什么需要重排——两段式检索架构：
    第一段（召回）：embedding 粗排，快但糙 —— 从几百块里捞回 k_recall 个候选；
    第二段（重排）：交叉编码器精排，慢但准 —— 逐对打分(query, doc)后取 top_k。
交叉编码器把 query 和 doc 拼在一起过模型，能看到词级交互，
区分度远高于双塔 embedding（只各自独立编码再算余弦）。

本工程实测（2026-09）：查询"阳虚体质有哪些表现？饮食上应该注意什么？"
    - embedding 排序：阳虚块垫底（cos 0.294，排第 6）
    - rerank 排序：  阳虚块得分 0.953，断层第一
即 rerank 不仅解决"候选无优先级"，还修复了 embedding 排序不准的问题。

接口（Cohere 兼容格式，无官方 SDK，直接 HTTP）：
    POST https://api.siliconflow.cn/v1/rerank
    {"model": "BAAI/bge-reranker-v2-m3", "query": ..., "documents": [...], "top_n": ...}

用法：
    from app.rerank import rerank
    ranked = rerank("阳虚体质表现", ["块1", "块2", ...], top_n=3)
    # → [(原始下标, 相关度分数), ...] 已按分数降序
"""
import json
import os
import urllib.request
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

API_URL = "https://api.siliconflow.cn/v1/rerank"
MODEL = "BAAI/bge-reranker-v2-m3"


def rerank(query: str, documents: list[str],
           top_n: int | None = None) -> list[tuple[int, float]]:
    """对候选文档按与 query 的相关度重排。

    Args:
        query:     查询文本（建议用"改写后的独立问题"效果更好）
        documents: 候选文档原文列表
        top_n:     只保留前 N 条，None 表示全部返回
    Returns:
        [(原始 documents 中的下标, relevance_score), ...] 按分数降序。
        score 是 0~1 的相关度（不是 logits，无需 sigmoid）。
    """
    if not documents:
        return []
    payload = {
        "model": MODEL,
        "query": query,
        "documents": documents,
        "top_n": top_n or len(documents),
        "return_documents": False,       # 不回传原文，省带宽
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {os.getenv('SILICONFLOW_API_KEY')}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        results = json.loads(resp.read()).get("results", [])
    # results 已按 relevance_score 降序；index 指向原始 documents 下标
    return [(r["index"], r["relevance_score"]) for r in results]
