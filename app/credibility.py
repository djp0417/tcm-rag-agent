# -*- coding: utf-8 -*-
"""引用可采信性筛选：**引文要看"可不可信"，不只是"相不相关"**（架构层问题 P0-3）。

要治的毛病
----------
实测反馈：系统把典籍里的**迷信 / 经验性记载**当作严肃建议的一部分输出——
例如鹿茸条目里的「中有小白虫，入人鼻必为虫颡」这类记载，被原样引进回答。
根因：引用筛选只判断"**是否相关**"，没判断"**是否可采信**"。

原则（一句话）
--------------
> 药性、功效、宜忌、配伍属**可采信**；
> 鬼神致病、符咒祈禳、成仙长生、虫入鼻成虫这类**传说性记载**，
> 要么剔除，要么只作**文化背景**说明一句。

为什么在"取用侧"做而不是重建索引
--------------------------------
重建要重嵌入上万块，代价大而无必要——和 `strip_meta_info` 的取舍一致。
这里做**句级**处理：一个 500 字的块里通常只有一两句是传说，
整块丢掉会把真知识一起丢掉。若剔除后剩不下什么内容，则整块丢弃。

判据分三级
----------
    ok          正常知识 → 原样进入参考资料
    background  带传说色彩但可能有文化价值 → 保留，但标注"仅作文化背景"
    reject      明确的迷信/巫术/长生之说 → 从参考资料中剔除
"""
from __future__ import annotations

import re

OK = "ok"
BACKGROUND = "background"
REJECT = "reject"

_LEVEL_LABEL = {OK: "可采信", BACKGROUND: "仅作文化背景", REJECT: "不可采信"}

# ---------------------------------------------------------------------------
# 一、判据词表
# ---------------------------------------------------------------------------
# reject：**不能作为建议依据**的记载。宁可少判，所以每一类都限定在
# "一旦出现就说明这条记载谈的不是药性"的表述上，避免误伤正常医理
# （如"胃为水谷之海"里的"海"、五行里的"神"都不是这里要抓的）。
_REJECT_PATTERNS = (
    r"鬼神", r"鬼(?:怪|祟|魅|魂)", r"邪祟", r"祟(?:病|鬼)", r"妖(?:邪|术)",
    r"符咒", r"符水", r"咒(?:语|术|禁)", r"祈禳", r"祈(?:祷|神)", r"祭(?:祀|拜)",
    r"请神", r"驱邪", r"辟邪(?:符|物)", r"作法", r"巫(?:术|者|婆)", r"蛊(?:毒|术)",
    r"成仙", r"长生(?:不老|不死|之药)", r"羽化", r"登仙", r"仙(?:方|丹|术)",
    r"炼丹(?:成仙|服食)", r"服(?:食|饵)(?:金|玉|丹砂)(?:求|以)(?:仙|寿|长生)",
    r"人中(?:白虫|虫)", r"入人鼻", r"必为虫", r"虫颡",
    r"报应", r"因果病", r"前世", r"转世", r"命中注定", r"上天(?:惩罚|降罪)",
    r"梦中(?:神仙|神人|有人授|遇仙)", r"神人(?:授|传|告)", r"梦授", r"神授",
    r"尸(?:注|疰)相传", r"传染(?:是|因)(?:鬼|神)",
)

# background：有文化价值、但**不足以支撑建议**。保留并标注，不用于论证疗效。
_BACKGROUND_PATTERNS = (
    r"相传", r"传说", r"民间(?:流|传)传", r"古人(?:云|说)", r"野史", r"笔记(?:云|载)",
    r"祝由", r"祝祷", r"禁咒", r"禳(?:灾|解)", r"以(?:镇|避)(?:邪|恶)",
    r"不祥", r"灾祸", r"凶兆", r"兆(?:验|头)",
)

_REJECT_RE = re.compile("|".join(_REJECT_PATTERNS))
_BACKGROUND_RE = re.compile("|".join(_BACKGROUND_PATTERNS))

# 句级切分（保留分隔符），与 scan.strip_meta_info 同一套断句口径
_SENT_RE = re.compile(r"(?<=[。！？；\n])")


# ---------------------------------------------------------------------------
# 二、判定
# ---------------------------------------------------------------------------
def classify(text: str) -> str:
    """给一段文字判可采信级别（reject > background > ok）。"""
    t = text or ""
    if _REJECT_RE.search(t):
        return REJECT
    if _BACKGROUND_RE.search(t):
        return BACKGROUND
    return OK


def classify_lines(text: str) -> list[tuple[str, str]]:
    """逐句判定：[(句子, 级别)]。"""
    return [(p, classify(p)) for p in _SENT_RE.split(text or "") if p.strip()]


def level_label(level: str) -> str:
    return _LEVEL_LABEL.get(level, level)


# 自己补的两句注释。**幂等的正确做法**是：处理前先把上次补的注释摘掉（否则
# 注释里"传说""文化背景"这些词会被再次判定），**并记住它曾经存在** ——
# 若只是摘掉不记，第二遍会因为"本轮没有命中 reject 句"而不再补回，
# 文本就又变了（第一版两处都错过：先是重复补，改完变成第二遍丢注释）。
NOTE_DROPPED = ("\n（注：本段中涉及传说/巫术性的记载已略去，"
                "它们不作为建议依据。）")
NOTE_BACKGROUND = ("\n（注：其中「相传/传说」类内容仅供文化背景了解，"
                   "**不得**作为「能不能吃」的依据。）")

_NOTE_DROPPED_RE = re.compile(r"\n?（注：本段中涉及传说/巫术性的记载已略去[^）]*）")
_NOTE_BACKGROUND_RE = re.compile(r"\n?（注：其中「相传/传说」类内容仅供文化背景了解[^）]*）")
_NOTE_RE = re.compile(r"\n?（注：(?:本段中涉及传说|其中「相传)[^）]*）")


def filter_chunk(text: str, min_keep: int = 40) -> tuple[str, dict]:
    """清洗一个检索块：剔除 reject 句，标注 background 句。

    Returns: (清洗后文本, 审计信息)

    审计信息：
      {"level": 整块的级别, "dropped": [被剔除的句子…], "background": [背景句…],
       "rejected": 是否整块丢弃}
    整块丢弃的判据：剔除后剩余内容过短（< min_keep 字符），说明这块几乎都是
    传说性的，留下来只会误导。**幂等**：同一块处理两次结果一致。
    """
    raw = text or ""
    had_dropped_note = bool(_NOTE_DROPPED_RE.search(raw))
    had_bg_note = bool(_NOTE_BACKGROUND_RE.search(raw))
    t = _NOTE_RE.sub("", raw)
    kept: list[str] = []
    dropped: list[str] = []
    bg: list[str] = []
    for piece, lv in classify_lines(t):
        if lv == REJECT:
            dropped.append(piece.strip())
            continue
        if lv == BACKGROUND:
            bg.append(piece.strip())
        kept.append(piece)
    cleaned = "".join(kept).strip()
    info = {"level": classify(t), "dropped": dropped, "background": bg,
            "rejected": bool(dropped) and len(cleaned) < min_keep}
    if info["rejected"]:
        return "", info
    # 幂等：同一块被处理两次（生成前一次、审计一次）不能再补一遍注释；
    # 反之，上一遍补过的注释摘掉后必须原样补回，否则第二遍会"丢注释"。
    if (dropped or had_dropped_note) and "已略去" not in cleaned:
        # 明确告诉模型"这里少了一句"——否则它可能把上下文接错。
        cleaned += NOTE_DROPPED
    if (bg or had_bg_note) and "仅供文化背景" not in cleaned:
        cleaned += NOTE_BACKGROUND
    return cleaned, info


def filter_sources(items) -> tuple[list, list[dict]]:
    """批量清洗 [(doc, score)]：返回（保留的，审计列表）。"""
    out, audits = [], []
    for doc, score in (items or []):
        cleaned, info = filter_chunk(getattr(doc, "page_content", "") or "")
        meta = getattr(doc, "metadata", {}) or {}
        if info["rejected"]:
            # 整块丢弃：只留审计，不喂给模型（并保留来源名便于解释）
            audits.append({"source": meta.get("source", "?"),
                           "chapter": meta.get("chapter", "?"),
                           "level": info["level"],
                           "label": level_label(info["level"]),
                           "dropped": len(info["dropped"]),
                           "rejected": True})
            continue
        try:
            doc.page_content = cleaned          # 就地替换（doc 是本轮临时对象）
        except Exception:
            pass
        audits.append({"source": meta.get("source", "?"),
                       "chapter": meta.get("chapter", "?"),
                       "level": info["level"],
                       "label": level_label(info["level"]),
                       "dropped": len(info["dropped"]),
                       "rejected": False})
        out.append((doc, score))
    return out, audits


# 注入提示词的约束块（与 scripts.py 里的话术块同一层，放在这里便于就近维护）
CREDIBILITY_BLOCK = """【引用的可采信性筛选（引用前先过这一道）】
典籍里夹着**迷信 / 传说 / 巫术性记载**（例如「中有小白虫，入人鼻必为虫颡」
这类，以及鬼神致病、符咒祈禳、成仙长生、报应之说）。引用前必须判断：
  1) **可采信**：药性、功效、主治、宜忌、配伍、炮制 → 可以用作依据；
  2) **不可采信**：鬼神致病、巫术咒符、成仙长生、怪诞传闻 →
     **不得**作为"能不能吃 / 有没有效"的依据；确要提到，只能作
     「古人有这样一种说法（属传说）」的文化背景说明，一句带过、不加论证；
  3) **存疑经验之谈**：古人个人的经验记载、无出处传闻 → 不用于支撑结论，
     要引用必须注明"这是古人的经验记载，未经验证"。
**每条引用之后紧接着写出「所以对你而言意味着什么」**——只引不判比不引更危险；
同一条原文用在**不同人**身上，判读必须不同（同一段绿豆原文，寒证与热证
要得出相反结论）。"""
