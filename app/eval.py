# -*- coding: utf-8 -*-
"""量化评估体系：把"感觉效果不错"变成可复现的数字。

为什么需要它（本项目的真实经历）
--------------------------------
没有评估之前，所有优化都是"看起来更好"。本项目第一个真正的 bug——语料里
34% 的块含 Unicode 部首污染（`⼈参` 而非 `人参`）——肉眼完全看不出来
（终端里两种字形几乎一样），是**靠关键词覆盖率这个指标异常为 0 才暴露的**。
这就是评估体系的核心价值：它是唯一能发现"静默失效"的手段。

四项指标与它们各自回答的问题
----------------------------
1. **检索质量（recall@1 / recall@4 / MRR / 关键词覆盖率）**
   问："该召回的块，召回了没有？"
   关键设计：同一测试集分别以「关闭自反思」和「开启自反思」跑两遍，
   差值 Δ 就是方向二（rerank 分数驱动的自反思检索）的**净增益**。
   没有这个对照，"我做了自反思"就只是个说法，不是个数据。

2. **答案忠实度（faithfulness，LLM-as-judge）**
   问："回答里的事实，参考资料里真的都有吗？"
   医疗健康领域幻觉代价高，这条比"回答像不像样"重要得多。

3. **拒答正确率（refusal accuracy）**
   问两件事：域外问题（写代码/查股价）有没有正确说"资料库没有"；
   域内问题有没有被**误拒**（over-refusal 同样是一种失败）。

4. **长期记忆保持率（memory retention）**
   问："第 1 轮说过的信息，第 9 轮还记得吗？换个新会话还记得吗？"
   这是对本项目"只有六轮对话"问题的直接验收：旧版硬编码最近 6 条消息，
   第 9 轮必然失忆；新版三层记忆应当守住。

判分口径（写死在这里，避免每次跑完都"重新解释什么算对"）
------------------------------------------------------
- hit@k：gold 来源（source 文件名）出现在 rerank 后的 top-k 里即算命中；
  gold_sources 是列表，任一命中即算对（同一问题可能跨源可答）。
- 关键词覆盖率：gold_keywords 在 top-k 检索结果正文中的出现比例（先做
  Unicode 归一，避免繁简/部首污染干扰判定）；它比 hit@k 更严格——命中了
  正确的书、但块切得不对，覆盖率就会掉下来。
- 忠实度：LLM 判 faithful(是/否) + 0~5 分；聚合时统计 faithful 比例与均分。
- 拒答：域外期望 refused=true；域内期望 refused=false。

用法（须在项目根目录以模块方式运行）：
    python -m app.eval --validate        # 只校验测试集（零 LLM，先跑这个）
    python -m app.eval --retrieval       # 仅检索指标（含自反思对照）
    python -m app.eval --full            # 四项全跑，出 JSON+Markdown 报告
    python -m app.eval --full --tag after   # 打标签，便于前后对比
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.paths import BASE_DIR
from app.selfrag import retrieve, retrieve_with_reflection
from app.textfix import find_artifacts, normalize_text

EVAL_DIR = BASE_DIR / "eval"
TESTSET = EVAL_DIR / "testset.jsonl"

# 域外问题"正确拒答"的判定线索（宽松匹配，避免只认一种措辞）
REFUSAL_MARKERS = (
    "资料库中没有", "资料库中无", "资料库里没有", "没有这方面的内容",
    "无直接依据", "常识性建议", "不在我的", "无法回答", "没有相关信息",
    "未收录", "没有找到相关",
)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class RetrievalRow:
    id: str
    category: str
    question: str
    gold_sources: list[str]
    hit1: bool = False
    hit4: bool = False
    mrr: float = 0.0
    kw_cover: float = 0.0
    kw_hit4: bool = False
    top1_score: float = 0.0
    rounds: int = 1
    final_query: str = ""
    got_sources: list[str] = field(default_factory=list)


@dataclass
class AnswerRow:
    id: str
    category: str
    question: str
    domain: str
    answer: str = ""
    refused: bool = False
    answered: bool = False       # 是否实质作答（与 refused 组合出"硬拒答/软拒答"）
    hard_refusal: bool = False   # 声明没有 + 未给实质内容
    expected_refusal: bool = False
    refusal_ok: bool = False
    faithful: bool | None = None
    faith_score: int | None = None
    unsupported: list[str] = field(default_factory=list)
    kw_cover: float = 0.0
    sources: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 测试集与校验
# ---------------------------------------------------------------------------
def load_testset(path: Path = TESTSET) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def validate(testset: list[dict]) -> bool:
    """校验测试集与当前语料是否对得上——评估失真的最大来源是标注写错。

    检查三件事：
    ① gold_sources 里的文件名确实在库里；
    ② 每个关键词在「gold_sources 覆盖的块」里真的能找到（归一化后比对）；
    ③ 语料残留污染数（应为 0，否则先跑建库修复）。
    """
    from app.rag import get_db

    print("=" * 68)
    print("测试集校验（先跑这个，避免评估失真）")
    print("=" * 68)
    db = get_db()
    data = db.get()
    by_source: dict[str, list[str]] = {}
    for txt, meta in zip(data["documents"], data["metadatas"]):
        by_source.setdefault(meta.get("source", ""), []).append(normalize_text(txt))

    # ③ 全库污染残留
    dirty, kinds, sample = find_artifacts("".join(data["documents"]))
    print(f"语料污染残留：{dirty} 次（{kinds} 种）"
          f"{'  ✅' if dirty == 0 else '  ⚠️ 先跑 python -m app.index 修复'}")
    if dirty:
        print(f"  样例：{sample}")
    print(f"库内来源 {len(by_source)} 份：{', '.join(sorted(by_source))}\n")

    bad = 0
    for row in testset:
        tid = row["id"]
        if row.get("domain") == "out":
            continue
        missing_src = [s for s in row["gold_sources"] if s not in by_source]
        if missing_src:
            print(f"[✗来源] {tid} {row['question'][:24]} → 库里没有: {missing_src}")
            bad += 1
        pool = "".join(t for s in row["gold_sources"] for t in by_source.get(s, []))
        miss_kw = [k for k in row["gold_keywords"] if normalize_text(k) not in pool]
        if miss_kw:
            print(f"[✗关键词] {tid} {row['question'][:24]} → 标注词不存在: {miss_kw}")
            bad += 1
    print("-" * 68)
    if bad:
        print(f"⚠️ {bad} 处标注与语料不符，请修正 eval/testset.jsonl 后再评估")
    else:
        print("✅ 测试集全部条目与语料一致")
    print("=" * 68 + "\n")
    return bad == 0


# ---------------------------------------------------------------------------
# 指标一：检索质量
# ---------------------------------------------------------------------------
def _kw_cover(kws: list[str], hits: list) -> float:
    """gold_keywords 在 top-k 命中块中的出现比例。

    匹配池里**同时包含正文与元数据**（source/chapter）：切块时标题被
    strip_headers 移出正文进了 metadata，若只看正文，像"四气调神大论"
    这种"篇章名即考点"的标注会被误判为未命中。
    """
    if not kws:
        return 1.0
    blob = normalize_text("".join(
        f"{d.metadata.get('source', '')}{d.metadata.get('chapter', '')}{d.page_content}"
        for d, _ in hits))
    return sum(1 for k in kws if normalize_text(k) in blob) / len(kws)


def _score_hits(row: dict, hits: list, rounds: int, fq: str) -> RetrievalRow:
    got = [d.metadata.get("source", "?") for d, _ in hits]
    r = RetrievalRow(id=row["id"], category=row["category"],
                     question=row["question"], gold_sources=row["gold_sources"],
                     got_sources=got, rounds=rounds, final_query=fq)
    gold = set(row["gold_sources"])
    r.hit1 = bool(hits) and got[0] in gold
    for i, s in enumerate(got[:4]):
        if s in gold:
            r.hit4 = True
            r.mrr = 1.0 / (i + 1)
            break
    r.kw_cover = _kw_cover(row["gold_keywords"], hits)
    r.kw_hit4 = r.kw_cover >= 0.5          # 半数以上标注词落到 top-4 才算"金标块召回"
    r.top1_score = round(hits[0][1], 3) if hits else 0.0
    return r


def eval_retrieval(testset: list[dict], reflect: bool, limit: int | None = None) -> list[RetrievalRow]:
    """跑一遍检索指标。reflect=False 为单轮基线，True 为自反思版本。"""
    from app.rag import get_db

    db = get_db()
    rows = [r for r in testset if r.get("domain") != "out"]
    if limit:
        rows = rows[:limit]
    out: list[RetrievalRow] = []
    for i, row in enumerate(rows, 1):
        if reflect:
            res = retrieve_with_reflection(db, row["question"])
            rr = _score_hits(row, res.hits, len(res.attempts), res.query)
        else:
            hits = retrieve(db, row["question"])
            rr = _score_hits(row, hits, 1, row["question"])
        out.append(rr)
        mark = "✓" if rr.hit4 else "✗"
        print(f"  [{i:>2}/{len(rows)}] {mark} hit@1={'1' if rr.hit1 else '0'} "
              f"kw={rr.kw_cover:.0%} top1={rr.top1_score:.2f} "
              f"{row['id']} {row['question'][:26]}")
    return out


def summarize_retrieval(rows: list[RetrievalRow]) -> dict:
    n = len(rows) or 1
    return {
        "n": len(rows),
        "recall@1": round(sum(r.hit1 for r in rows) / n, 4),
        "recall@4": round(sum(r.hit4 for r in rows) / n, 4),
        "kw_hit@4": round(sum(r.kw_hit4 for r in rows) / n, 4),
        "mrr": round(sum(r.mrr for r in rows) / n, 4),
        "kw_cover": round(sum(r.kw_cover for r in rows) / n, 4),
        "avg_rounds": round(sum(r.rounds for r in rows) / n, 2),
    }


# ---------------------------------------------------------------------------
# 指标二/三：忠实度 + 拒答
# ---------------------------------------------------------------------------
FAITH_PROMPT = """你是严格的技术评审。请判断【回答】里关于中医养生的**实质性论断**是否都能从【参考资料】找到依据。

判定规则：
- 每条实质论断（药物功效、方剂组成、体质特征、原文出处、脏腑关系等）都能在参考资料中找到出处或直接依据 → faithful=true
- 出现参考资料中没有的具体事实（如凭空出现的方剂名、药物功效、篇名、剂量）→ faithful=false，并列出这些论断
- 对参考资料的通俗化转述、合理推论、过渡语、礼貌用语 → 不算无依据
- 回答因为资料不足而简短、或明确说"资料库中没有" → 不算无依据

【参考资料】
{context}

【问题】{question}

【回答】{answer}

只输出 JSON，不要其他任何文字：
{{"faithful": true 或 false, "score": 0-5 的整数, "unsupported": ["无依据的论断", ...]}}"""

REFUSE_PROMPT = """判断下面【回答】的两件事，分别给布尔值：

一、refused：回答是否明确告知用户"资料库中没有这方面的内容"？
- 明确说资料库里没有/未收录/无法回答相关问题 → true
- 即使之后补充了常识性建议，只要明确说了资料库里没有 → 仍算 true
- 正常回答了问题（没有声明资料库无此内容）→ false

二、answered：回答是否**实质性地回答了用户的问题**？
- 给出了与问题相关的具体养生内容（解释、条目、引文、建议等）→ true
- 只是说"没有/无法回答"就结束，或内容与问题无关 → false

【问题】{question}

【回答】{answer}

只输出 JSON：{{"refused": true 或 false, "answered": true 或 false, "reason": "一句话理由"}}"""


def _judge_json(prompt: str) -> dict:
    """调 LLM 判分并解析 JSON；解析失败时返回空 dict（调用方按失败处理）。"""
    from app.llm import get_llm

    try:
        raw = get_llm().invoke(prompt).content or ""
    except Exception as e:
        print(f"    [judge失败] {type(e).__name__}: {e}")
        return {}
    s = raw.strip()
    if s.startswith("```"):                       # 容忍模型套 markdown 代码块
        s = s.strip("`")
        s = s[s.find("{"):]
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j < 0:
        return {}
    try:
        return json.loads(s[i:j + 1])
    except json.JSONDecodeError:
        return {}


def eval_answers(testset: list[dict], limit: int | None = None) -> list[AnswerRow]:
    """对每个问题跑完整 RAG（关闭自反思，聚焦生成质量），再 LLM 判分。"""
    from app.rag import RAGSession

    s = RAGSession(reflect=False, auto_remember=False)   # 不写记忆，保持评估无副作用
    rows = testset[:limit] if limit else testset
    out: list[AnswerRow] = []
    for i, row in enumerate(rows, 1):
        ans = s.ask(row["question"])
        ar = AnswerRow(id=row["id"], category=row["category"],
                       question=row["question"], domain=row.get("domain", "in"),
                       answer=ans.answer,
                       sources=[x["source"] for x in ans.sources],
                       expected_refusal=row.get("domain") == "out")
        # 先用关键词线索判拒答，再用 LLM 复核
        ar.refused = any(m in ans.answer for m in REFUSAL_MARKERS)
        # 答案层关键词覆盖率：比检索层更贴近"到底答到点上没有"
        ar.kw_cover = _kw_in_text(row["gold_keywords"], ans.answer)
        # LLM 判分
        v = _judge_json(REFUSE_PROMPT.format(question=row["question"], answer=ans.answer))
        if "refused" in v:
            ar.refused = bool(v["refused"])
        ar.answered = bool(v.get("answered", not ar.refused))
        # 「硬拒答」= 声明没有 且 没给实质内容 → 真失败；
        # 「软拒答」= 声明没有 但 仍给了相关内容 → 诚实且有用，只算部分失败。
        ar.hard_refusal = ar.refused and not ar.answered
        if ar.expected_refusal:
            ar.refusal_ok = ar.refused
        else:
            ar.refusal_ok = not ar.hard_refusal   # 域内只有硬拒答才算失败
            v2 = _judge_json(FAITH_PROMPT.format(
                context=normalize_text(ans.context),
                question=row["question"], answer=ans.answer))
            if "faithful" in v2:
                ar.faithful = bool(v2["faithful"])
                ar.faith_score = int(v2.get("score", 0) or 0)
                ar.unsupported = list(v2.get("unsupported", []) or [])
        out.append(ar)
        icon = "✓" if (ar.refusal_ok and (ar.faithful is not False)) else "✗"
        print(f"  [{i:>2}/{len(rows)}] {icon} refuse={ar.refused} "
              f"faith={ar.faithful} kw={ar.kw_cover:.0%} {row['id']}")
    return out


def _kw_in_text(kws: list[str], text: str) -> float:
    if not kws:
        return 1.0
    t = normalize_text(text)
    return sum(1 for k in kws if normalize_text(k) in t) / len(kws)


def summarize_answers(rows: list[AnswerRow]) -> dict:
    inn = [r for r in rows if not r.expected_refusal]
    out = [r for r in rows if r.expected_refusal]
    faith = [r for r in inn if r.faithful is not None]
    return {
        "n_in": len(inn), "n_out": len(out),
        "faithful_rate": round(sum(1 for r in faith if r.faithful) / (len(faith) or 1), 4),
        "faith_avg_score": round(
            sum(r.faith_score or 0 for r in faith) / (len(faith) or 1), 2),
        "answer_kw_cover": round(
            sum(r.kw_cover for r in inn) / (len(inn) or 1), 4),
        # 硬拒答才是真失败：声明"没有"且没给实质内容
        "over_refusal_rate": round(
            sum(1 for r in inn if r.hard_refusal) / (len(inn) or 1), 4),
        # 软拒答：声明"没有"但仍给了相关内容，属诚实且有用，单独统计
        "soft_refusal_rate": round(
            sum(1 for r in inn if r.refused and r.answered) / (len(inn) or 1), 4),
        "out_refusal_rate": round(
            sum(1 for r in out if r.refused) / (len(out) or 1), 4),
    }


# ---------------------------------------------------------------------------
# 指标四：长期记忆保持
# ---------------------------------------------------------------------------
MEMORY_TURNS = [
    "我叫丁建鹏，今年 26 岁，是一名程序员，平时长期熬夜到凌晨两点。",
    "阳虚体质的人平时饮食要注意什么？",
    "那冬天应该怎么调养？",
    "生姜有什么食疗作用？",
    "平时适合做哪些运动？",
    "《黄帝内经》里说的四气调神是什么意思？",
    "大枣有什么功效？",
    "我这样的体质，秋天要注意什么？",
]
# 第 9 轮探针：远超旧版硬编码的"最近 6 条消息"窗口，必须靠摘要/长期记忆才答得出
MEMORY_PROBE = "你还记得我的职业和作息习惯吗？"
# 事实验收用「同义组」而不是死关键词：模型完全可能把"程序员"复述成
# "常年坐办公室/在杭州上班"，只认原词会把"记得"误判成"失忆"。
MEMORY_FACT_GROUPS = (
    ("程序员", "坐办公室", "上班", "写代码"),     # 职业
    ("熬夜", "凌晨两点"),                        # 作息（不放"作息"——问句里就有，会假命中）
)
# 跨会话探针：新建会话（消息历史为空）。
# 2026-09-16「记忆按会话隔离」后的语义：默认情况下它**不应该**记得任何东西
# （这才是正确行为），所以这个探针的期望是"答不上来"——真正能唤起旧事的，
# 是下面这条带显式回忆措辞的探针。
CROSS_PROBE = "我平时作息怎么样？"
RECALL_PROBE = "我上次跟你说的作息，你还记得吗？"


def _fact_hit(text: str, groups) -> list[list[str]]:
    """每个同义组命中了哪些词；用于既判通过率、又保留可查证的依据。"""
    return [[w for w in g if w in text] for g in groups]


def eval_memory() -> dict:
    """多轮对话 → 检验记忆的两面：**本会话内不能忘** + **跨会话不能串**。

    三段验收（2026-09-16 按「记忆按会话隔离」重写）：
      ① 同会话第 9 轮（> 旧版 6 条窗口）仍能说出第 1 轮提到的职业/作息；
      ② 新建会话直接问"我平时作息怎么样" → **应当答不上来**。
         以前这里期望"记得"，那是旧语义（全局档案跨会话生效），也正是
         用户实测抱怨的"新开的对话被旧信息糊脸"；现在反过来，
         **答不上来才是隔离正确的证据**；
      ③ 用户主动问起往事（"我上次跟你说的…"）→ 才该回忆起来。
         前提是这条信息以 scope='global' 存在（用户明说"记住"才会这样写），
         所以这里直接用 storage.add_memory(scope=global) 造一条，
         把"通道本身通不通"和"模型抽不抽得出事实"分开验。
    """
    from app import storage
    from app.rag import RAGSession

    conv_title = f"[eval] 记忆保持 {time.strftime('%H%M%S')}"
    conv = storage.create_conversation(conv_title)
    cid = conv["id"]
    log: list[dict] = []
    try:
        s = RAGSession(conv_id=cid, reflect=False, auto_remember=True)
        for i, q in enumerate(MEMORY_TURNS, 1):
            a = s.ask(q)
            storage.add_message(cid, "user", q)
            storage.add_message(cid, "assistant", a.answer)
            log.append({"turn": i, "q": q, "a": a.answer[:200]})
            print(f"  [turn {i:>2}] {q[:28]}")

        # ① 同会话探针
        p1 = s.ask(MEMORY_PROBE)
        storage.add_message(cid, "user", MEMORY_PROBE)
        storage.add_message(cid, "assistant", p1.answer)
        same_hit = _fact_hit(p1.answer, MEMORY_FACT_GROUPS)
        same_ok = all(h for h in same_hit)
        print(f"  [探针①] 第 {len(MEMORY_TURNS) + 1} 轮回忆 → "
              f"{'✅ 记得' if same_ok else '❌ 失忆'}：{p1.answer[:90]}")

        # ② 新会话直接问 → 期望"不知道"（隔离正确）
        conv2 = storage.create_conversation(conv_title + " · 新会话")
        s2 = RAGSession(conv_id=conv2["id"], reflect=False, auto_remember=False)
        p2 = s2.ask(CROSS_PROBE)
        cross_hit = _fact_hit(p2.answer, MEMORY_FACT_GROUPS)
        leak = any(cross_hit)                # 命中任一 → 串了别的会话的信息
        cross_ok = not leak
        print(f"  [探针②] 新会话直接问 → "
              f"{'✅ 无旧记忆（隔离正确）' if cross_ok else '❌ 串档：说出了别会话的事'}"
              f"：{p2.answer[:90]}")

        # ③ 显式长期记忆通道：用户明说"记住"才会写成 global
        storage.add_memory("用户长期熬夜到凌晨两点，是一名程序员",
                           kind="fact", conv_id=cid, scope=storage.MEM_GLOBAL)
        p3 = s2.ask(RECALL_PROBE)
        storage.add_message(conv2["id"], "user", RECALL_PROBE)
        storage.add_message(conv2["id"], "assistant", p3.answer)
        recall_hit = _fact_hit(p3.answer, MEMORY_FACT_GROUPS)
        recall_ok = all(h for h in recall_hit)
        print(f"  [探针③] 主动问起往事 → "
              f"{'✅ 能回忆' if recall_ok else '❌ 想不起来'}：{p3.answer[:90]}")

        stats = s.memory.build_context(cid, CROSS_PROBE).stats
        return {
            "turns": len(MEMORY_TURNS),
            "same_session_ok": same_ok, "same_session_answer": p1.answer,
            "same_session_hits": same_hit,
            "isolation_ok": cross_ok, "new_session_answer": p2.answer,
            "new_session_hits": cross_hit,
            "recall_ok": recall_ok, "recall_answer": p3.answer,
            "recall_hits": recall_hit,
            "recalled_facts": s.memory.recall(CROSS_PROBE, cid, topk=5),
            "context_stats": stats,
            "log": log,
        }
    finally:
        # 清理评估产生的会话与记忆，保持库干净（会话删除会连带清掉它的
        # 会话档案与 session 记忆；global 条目要单独删）
        for c in (conv["id"],):
            storage.delete_conversation(c)
        for m in storage.list_memories(limit=500):
            if m.get("scope") == storage.MEM_GLOBAL:
                storage.delete_memory(m["id"])
        for c in storage.list_conversations():
            if c["title"].startswith("[eval]"):
                storage.delete_conversation(c["id"])


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def write_report(result: dict, tag: str) -> Path:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M")
    jp = EVAL_DIR / f"report_{tag}.json"
    jp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    L: list[str] = [f"# RAG 评估报告（{tag}）", "", f"生成时间：{stamp}", ""]
    L += ["## 一、检索质量", "",
          "| 配置 | recall@1 | 书级 recall@4 | 金标块召回 kw_hit@4 | MRR | 关键词覆盖 | 平均检索轮数 |",
          "|---|---|---|---|---|---|---|"]
    for k, name in (("baseline", "单轮检索（无自反思）"), ("reflect", "自反思检索（方向二）")):
        if k in result.get("retrieval", {}):
            s = result["retrieval"][k]
            L.append(f"| {name} | {s['recall@1']:.1%} | {s['recall@4']:.1%} | "
                     f"**{s['kw_hit@4']:.1%}** | {s['mrr']:.3f} | "
                     f"{s['kw_cover']:.1%} | {s['avg_rounds']} |")
    if "retrieval_delta" in result:
        d = result["retrieval_delta"]
        L += ["", f"**自反思净增益：recall@1 {d['recall@1']:+.1%}、"
                  f"金标块召回 {d['kw_hit@4']:+.1%}、MRR {d['mrr']:+.3f}、"
                  f"关键词覆盖 {d['kw_cover']:+.1%}**", "",
              "（书级 recall@4 只问「命中了哪本书」，粒度太粗；**金标块召回**要求"
              "半数以上标注词落在 top-4，才是「这个块真的召回对了」）", ""]

    if "answers" in result:
        s = result["answers"]
        L += ["## 二、生成质量与拒答", "",
              "| 指标 | 数值 | 说明 |", "|---|---|---|",
              f"| 忠实度（无幻觉比例） | {s['faithful_rate']:.1%} | LLM 判材料可溯源 |",
              f"| 忠实度均分 | {s['faith_avg_score']}/5 | |",
              f"| 答案关键词覆盖 | {s['answer_kw_cover']:.1%} | 回答是否真的答到点上 |",
              f"| 域内**硬**拒答率 | {s['over_refusal_rate']:.1%} | 说没有且没给实质内容，真失败 |",
              f"| 域内软拒答率 | {s['soft_refusal_rate']:.1%} | 说没有但仍给了相关内容，诚实且有用 |",
              f"| 域外拒答率 | {s['out_refusal_rate']:.1%} | 越高越好 |", "",
              "> 拒答要分硬软：模型说「资料库没有专门记载，但以下几处相关」时，"
              "一刀切判失败会掩盖它其实答对了。**只有「说没有且什么都没给」才是真失败。**", ""]

    if "memory" in result:
        m = result["memory"]
        L += ["## 三、长期记忆保持（对「只有六轮对话」问题的验收）", "",
              f"- 对话轮数：{m['turns']} 轮后追问（旧版硬编码窗口仅 6 条消息）",
              f"- 同会话回忆：{'✅ 通过' if m['same_session_ok'] else '❌ 失忆'}"
              f"（命中同义词 {m.get('same_session_hits')}）",
              f"- 跨会话回忆：{'✅ 通过' if m['cross_session_ok'] else '❌ 失忆'}"
              f"（命中同义词 {m.get('cross_session_hits')}）",
              f"- 召回长期记忆：{m['recalled_facts']}",
              f"- 上下文组装统计：{m['context_stats']}", "",
              f"> 同会话探针回答：{m['same_session_answer'][:300]}", "",
              f"> 跨会话探针回答：{m['cross_session_answer'][:300]}", ""]

    if "retrieval_rows" in result:
        L += ["## 四、逐题明细", "",
              "| id | 类别 | 问题 | hit@1 | hit@4 | MRR | 关键词 | top1 | 轮数 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in result["retrieval_rows"]:
            L.append(f"| {r['id']} | {r['category']} | {r['question'][:22]} | "
                     f"{'✓' if r['hit1'] else '✗'} | {'✓' if r['hit4'] else '✗'} | "
                     f"{r['mrr']:.2f} | {r['kw_cover']:.0%} | {r['top1_score']} | {r['rounds']} |")
        L.append("")
    mp = EVAL_DIR / f"report_{tag}.md"
    mp.write_text("\n".join(L), encoding="utf-8")
    print(f"\n[报告] {mp}")
    print(f"[数据] {jp}")
    return mp


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="RAG 量化评估")
    ap.add_argument("--validate", action="store_true", help="只校验测试集（零 LLM）")
    ap.add_argument("--retrieval", action="store_true", help="只跑检索指标")
    ap.add_argument("--memory", action="store_true", help="只跑长期记忆保持指标")
    ap.add_argument("--full", action="store_true", help="四项指标全跑")
    ap.add_argument("--tag", default="run", help="报告标签（如 baseline / after）")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 题（调试用）")
    args = ap.parse_args()

    testset = load_testset()
    print(f"[测试集] {len(testset)} 条（域内 "
          f"{sum(1 for r in testset if r.get('domain') != 'out')} / 域外 "
          f"{sum(1 for r in testset if r.get('domain') == 'out')}）\n")

    if args.validate:
        validate(testset)
        return

    if not validate(testset):
        print("⚠️ 测试集校验未通过，建议先修正标注（继续跑会得到失真的数字）\n")

    result: dict = {"tag": args.tag, "time": time.strftime("%Y-%m-%d %H:%M:%S")}

    if args.retrieval or args.full:
        print("── 指标一：检索质量（单轮基线）──")
        base = eval_retrieval(testset, reflect=False, limit=args.limit)
        print("── 指标一：检索质量（自反思）──")
        refl = eval_retrieval(testset, reflect=True, limit=args.limit)
        sb, sr = summarize_retrieval(base), summarize_retrieval(refl)
        result["retrieval"] = {"baseline": sb, "reflect": sr}
        result["retrieval_delta"] = {
            k: round(sr[k] - sb[k], 4) for k in
            ("recall@1", "recall@4", "kw_hit@4", "mrr", "kw_cover")}
        # 明细以自反思版本为准；被自反思"救回"的题目标注出来
        bmap = {r.id: r for r in base}
        for r in refl:
            b = bmap.get(r.id)
            if b and not b.kw_hit4 and r.kw_hit4:
                r.final_query = f"【自反思救回】{r.final_query}"
        result["retrieval_rows"] = [asdict(r) for r in refl]
        print(f"\n  金标块召回: {sb['kw_hit@4']:.1%} → {sr['kw_hit@4']:.1%} "
              f"({result['retrieval_delta']['kw_hit@4']:+.1%})")
        print(f"  recall@1  : {sb['recall@1']:.1%} → {sr['recall@1']:.1%} "
              f"({result['retrieval_delta']['recall@1']:+.1%})\n")

    if args.full:
        print("── 指标二/三：忠实度 + 拒答 ──")
        rows = eval_answers(testset, limit=args.limit)
        result["answers"] = summarize_answers(rows)
        result["answer_rows"] = [asdict(r) for r in rows]
        print()

    if args.full or args.memory:
        print("── 指标四：长期记忆保持 ──")
        result["memory"] = eval_memory()
        print()

    write_report(result, args.tag)


if __name__ == "__main__":
    main()
