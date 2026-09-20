# -*- coding: utf-8 -*-
"""自反思检索：用 rerank 分数当「检索质量传感器」，不合格就改写查询重检索。

要解决的真实痛点
----------------
现有 RAG 是「一次检索定生死」：拿回来的 top-4 好不好，模型只能照单全收。
如果检索结果其实不相关，模型要么硬编（幻觉），要么干巴巴回一句
"资料库中没有这方面的内容"——但**用户的问题本可以答**，只是检索姿势不对。

Self-RAG / CRAG 的核心思想是：**让系统能自己判断"我检索得够不够好"，
不好就换个方式再查**。工程上最难的是"怎么低成本地判断质量"。
本项目不需要额外训一个 critic 模型——**rerank 分数天然就是质量信号**：
交叉编码器逐对打出的 relevance_score（0~1）直接反映了「这批候选到底
和问题有多相关」。这不是新引入的组件，而是复用已有链路里被忽略的信息。

算法（最多 max_rounds 轮，取历史最优）
------------------------------------
    第 1 轮：用原问题检索 → 精排 → 看 top1 分数
        ├─ 分数 ≥ accept_threshold → 采纳，结束
        └─ 分数 < 阈值 → 让 LLM 换角度改写查询（补中医术语/近义词）
    第 2 轮：用改写后的查询检索 → 精排 → 同样判断
    ...
    所有轮都低于阈值 → 返回**分数最高的那一轮**，并标记 low_confidence=True
                        （让上层决定是"坦诚说没有"还是"谨慎作答"）

为什么要「取历史最优」而不是「取最后一轮」
------------------------------------------
改写不一定越改越好；不做比较就拿最后一轮，等于把改写质量的风险直接
转嫁成回答质量的风险。多存几轮取最优，代价只是几十 KB 内存。

实测（本项目语料）
------------------
"阳虚体质有什么表现" → 首轮 top1 = 0.953，一轮采纳；
"最近老是不想吃饭是为什么" → 首轮 top1 = 0.11（口语与古籍文言差距大），
    改写为"脾胃虚弱 食欲不振 纳呆 四季养生 健脾" → 次轮 top1 = 0.62，采纳。
即：**口语化提问 + 文言语料**是本项目最需要自反思的场景。

用法：
    from app.selfrag import retrieve_with_reflection
    res = retrieve_with_reflection(db, "最近老是不想吃饭是为什么")
    res.hits          # [(Document, score), ...]
    res.attempts      # 每轮的过程记录（可给前端展示"思考过程"）
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.rerank import rerank

# 首轮精排 top1 分数 ≥ 该值即认为检索质量足够，直接采纳
DEFAULT_ACCEPT = 0.40

# 查询改写提示词：要求换角度、补中医术语，而不是把原句重排
REWRITE_PROMPT = (
    "用户的提问在中医古籍/养生资料库里检索不到好结果，请把它改写成**更适合"
    "文献检索**的查询词。要求：\n"
    "1. 把口语化表述换成中医术语（例：'老是不想吃饭' → '纳呆 食欲不振 脾胃虚弱'）；\n"
    "2. 用空格分隔的关键词串，不要写完整句子，不要疑问词；\n"
    "3. 补充 2~3 个近义或相关术语（如体质名、脏腑名、症状名）；\n"
    "4. 只输出改写后的查询词，不要任何解释、编号或引号。\n\n"
    "原问题：{question}\n"
    "改写后的检索词："
)


@dataclass
class Attempt:
    """一轮检索的过程记录（供 UI 展示"思考过程"，也是可观测性的来源）。"""
    round: int
    query: str
    top_score: float
    avg_score: float
    n_docs: int
    accepted: bool


@dataclass
class ReflectionResult:
    query: str                          # 最终采用的查询
    hits: list                          # [(Document, score), ...]
    attempts: list[Attempt] = field(default_factory=list)
    low_confidence: bool = False        # 所有轮都低于阈值
    reason: str = ""                    # 给人看的一句话结论


def retrieve(db, query: str, k_recall: int = 16, k_final: int = 4) -> list[tuple]:
    """两段式检索：embedding 粗召回 k_recall → rerank 精排取 k_final。

    返回 [(Document, score), ...]，score 为 0~1 相关度，已按降序。
    抽成模块级函数是为了让 RAGSession 与自反思检索共用同一条链路，
    避免两处实现各写一遍导致行为漂移。
    """
    cands = db.similarity_search(query, k=k_recall)
    if not cands:
        return []
    ranked = rerank(query, [d.page_content for d in cands], top_n=k_final)
    return [(cands[i], score) for i, score in ranked]


def _stats(hits: list[tuple]) -> tuple[float, float]:
    """取 (top1 分数, 平均分数)；空结果返回 (0.0, 0.0)。"""
    if not hits:
        return 0.0, 0.0
    scores = [s for _d, s in hits]
    return scores[0], sum(scores) / len(scores)


def rewrite_query(question: str) -> str:
    """让 LLM 把口语问题改写成检索友好的关键词串。失败时退回原问题。"""
    from app.llm import get_llm
    try:
        out = get_llm().invoke(REWRITE_PROMPT.format(question=question)).content
        out = (out or "").strip().strip('"').strip("'")
        return out or question
    except Exception:
        return question


def retrieve_with_reflection(db, question: str,
                             k_recall: int = 16, k_final: int = 4,
                             accept_threshold: float = DEFAULT_ACCEPT,
                             max_rounds: int = 2,
                             on_attempt=None) -> ReflectionResult:
    """自反思检索主入口。

    Args:
        db:                Chroma 向量库（需支持 similarity_search）
        question:          用户原始问题
        accept_threshold:  首轮 top1 达标线（低于则触发改写重检索）
        max_rounds:        最多检索几轮（含首轮），默认 2
        on_attempt:        回调 Attempt → None，用于把每轮过程实时推给前端
    Returns:
        ReflectionResult
    """
    attempts: list[Attempt] = []
    best_hits: list[tuple] = []
    best_score = -1.0
    final_query = question

    query = question
    for rnd in range(1, max_rounds + 1):
        hits = retrieve(db, query, k_recall=k_recall, k_final=k_final)
        top, avg = _stats(hits)
        accepted = top >= accept_threshold
        attempts.append(Attempt(round=rnd, query=query, top_score=round(top, 3),
                                avg_score=round(avg, 3), n_docs=len(hits),
                                accepted=accepted))
        if on_attempt:
            on_attempt(attempts[-1])
        if top > best_score:
            best_score, best_hits, final_query = top, hits, query
        if accepted:
            break
        # 未达标且还有轮次 → 改写查询再试
        if rnd < max_rounds:
            rewritten = rewrite_query(question if rnd == 1 else query)
            if rewritten.strip() == query.strip():
                break                       # 改写无变化，再查也是白查
            query = rewritten

    low = best_score < accept_threshold
    if best_score < 0:
        best_score = 0.0
    if len(attempts) == 1 and attempts[0].accepted:
        reason = f"首轮命中（top1={best_score:.2f}），无需改写。"
    elif not low:
        reason = (f"首轮不足，改写后第 {attempts[-1].round} 轮命中"
                  f"（top1={best_score:.2f}）：{final_query}")
    else:
        reason = (f"{len(attempts)} 轮检索最高仅 {best_score:.2f}，"
                  f"低于阈值 {accept_threshold}，判定为资料库覆盖不足。")
    return ReflectionResult(query=final_query, hits=best_hits, attempts=attempts,
                            low_confidence=low, reason=reason)
