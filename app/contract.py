# -*- coding: utf-8 -*-
"""输出契约：每一轮回答必须包含固定模块，**缺哪个补哪个，不由模型决定**。

要治的毛病（架构层问题五）
--------------------------
输出模块的出现是随机的：有时有追问有时没有，有时有就医阈值有时没有，
有时有穴位有时没有，有时有替代方案有时没有——取决于模型"这一轮想没想到"，
而不是流程规定。实测同一能力项在不同轮次得分在 0 和 9 之间跳动，
用户无法建立稳定预期。

契约
----
    ① 分层判断（本 / 标）+ 判断依据
    ② 针对提问对象的逐条结论 + 每条的档位 + 理由（绑定此人条件）
    ③ 可执行部分（食疗 / 经络 / 起居 / 运动）
    ④ 风险管理（原处方药保护 / 监测指标 / 停止信号）
    ⑤ 就医与排查引导（含症状阈值）
    ⑥ 基于信息缺口的追问（说明为什么问）

流程上是**输出前的完整性自检**：哪个模块缺失就补，不允许直接输出。

三层保障（与项目其它地方同一套思路）
------------------------------------
1. **提示词侧**：契约以"必须包含"的形式注入（`render_contract`）；
2. **程序化自检**：生成后逐模块检测（`audit`），检测用的是**宽口径关键词**，
   宁可判"已有"也不要误补（追加内容越多越像八股文）；
3. **确定性补全**：缺 ②④⑤ 时用**已有的事实**补（分级判读文本、西药纪律原文、
   排查项清单）——这些都来自硬规则层，不依赖模型；缺 ① 只在信息足够时补，
   信息不足时**不补**（不能为了凑模块而下证型结论）。

为什么不把补全做成"全部都补"：一轮回答末尾堆六段模板会变成八股文，
用户的体感是"答非所问"。所以只补"缺了会有实质损失"的模块，
且每段都尽量短。
"""
from __future__ import annotations

import re

# 模块定义：(key, 人话名, 是否"缺了必须有动作")
MODULES: tuple[tuple[str, str], ...] = (
    ("layer", "分层判断（本 / 标）+ 依据"),
    ("verdicts", "逐条结论 + 档位 + 理由"),
    ("actions", "可执行部分（食疗/经络/起居/运动）"),
    ("risk", "风险管理（原处方药纪律/监测/停止信号）"),
    ("referral", "就医与排查引导（含阈值）"),
    ("questions", "基于信息缺口的追问（说明为什么问）"),
)

# 检测信号：宽口径（宁可判"已有"）。命中任一条即视为该模块存在。
_SIGNALS: dict[str, tuple[str, ...]] = {
    "layer": (r"本虚", r"标实", r"底子", r"壅滞", r"根在", r"本在", r"根源",
              r"层(?:面)?看", r"往上一步", r"说白了"),
    "verdicts": (r"不建议", r"可以用", r"可以吃", r"可以喝", r"能用", r"不能吃",
                 r"不能喝", r"不宜", r"暂缓", r"先(?:别|停)", r"适合", r"不适合",
                 r"需(?:专业)?确认", r"建议先", r"别急着"),
    "actions": (r"食疗", r"穴位", r"经络", r"按揉", r"艾灸", r"足三里", r"关元",
                r"起居", r"作息", r"运动", r"八段锦", r"散步", r"泡脚", r"导引"),
    "risk": (r"不能自行停", r"不能停", r"不要自行(?:停|减)", r"不能自行减",
             r"监测", r"记录", r"复查", r"立即就医", r"就诊", r"就停", r"即停",
             r"停止信号", r"出现.{0,8}(?:就|要)停"),
    "referral": (r"建议(?:去)?(?:查|做|检)", r"排查", r"查一下", r"检查",
                 r"去医院", r"看看医生", r"mmHg", r"阈值", r"超出.{0,6}立即"),
    "questions": (r"[?？]",),
}
_SIGNAL_RE = {k: re.compile("|".join(v)) for k, v in _SIGNALS.items()}

# 补全时最多补几段（避免末尾堆成八股文）
MAX_APPEND = 3


def audit(answer: str) -> dict[str, bool]:
    """逐模块检测回答覆盖情况。"""
    a = answer or ""
    return {k: bool(_SIGNAL_RE[k].search(a)) for k, _ in MODULES}


def missing(answer: str) -> list[str]:
    return [k for k, ok in audit(answer).items() if not ok]


# ---------------------------------------------------------------------------
# 确定性补全（用的都是硬规则层已有的事实，不依赖模型）
# ---------------------------------------------------------------------------
# 2026-09-16 第五轮（版本退化修复）：此前这里是
# `**再补充几点（前面没有展开，但和你的情况直接相关）：**` —— 这是典型的
# **流程语言**（在跟用户解释"我在补模块"）。反馈实测它**重复出现两遍**，
# 且后面直接倒出内部规则原文。现在补漏段改用**内容性小标题**直接承接
# （每段自带「用药纪律 / 建议先做的排查 / 逐条结论」），不加任何元叙述。
_HEAD = "\n\n"

# 上一版遗留的流程语言引子：历史回答里可能还带着它，被模型复述出来。
# 输出前统一剥掉（这样"再补充几点…"不会再在界面上出现第二遍）。
_FLOW_RE = re.compile(
    r"\n*-{3,}\n*\*\*再补充几点（前面没有展开[^）]*）[：:]?\*\*\n*"
    r"|\n*\*\*再补充几点（前面没有展开[^）]*）[：:]?\*\*\n*")


def _strip_flow(answer: str) -> str:
    """剥掉**旧版**遗留的流程语言引子（幂等，无则原样返回）。"""
    a = _FLOW_RE.sub("\n\n", answer or "")
    return a.lstrip("\n") if a.strip() else a


def _clip(text: str, n: int = 120) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip())
    return t if len(t) <= n else t[:n].rstrip("，、；") + "…"


def _fill_verdicts(ctx: dict) -> str:
    """逐条结论：用药材/食材的**分级判读**（硬规则层生成的文本）。

    西药类命中**排除在外**——它们的判读文本就是"用药纪律"，
    由 `_fill_risk` 负责；两个都补会在回答末尾出现两遍一模一样的纪律段。
    """
    from app.safety import tiers as T
    hits = [h for h in (ctx.get("hits") or [])
            if h.get("origin", "message") == "message" and h.get("kind") != "drug"]
    if not hits:
        return ""
    lines = ["**逐条结论**："]
    for h in hits:
        lines.append(f"- **{h.get('name', '?')}**（{h.get('tier_label', '')}）："
                     f"{_clip(T.render(h), 160)}")
    return "\n".join(lines)


def _fill_risk(ctx: dict) -> str:
    """风险管理：状态约束 + 用西药类命中项自带的强制话术（硬规则原文，允许长）。"""
    from app import constraints as CN

    parts: list[str] = []
    states = ctx.get("states") or []
    if states:
        # P1-3：状态类约束（备孕/妊娠/哺乳/高龄…）优先于任何药名触发的规则。
        # 它出现在这一段的**最前面**，因为"先识别状态、再谈其它"就是它的效力。
        #
        # 2026-09-16 第五轮：此前这里用 `s["note"]`（**给模型看的机制说明**），
        # 结果把内部规则原文当输出倒给用户——备孕那条 note 里含红花/桃仁/
        # 益母草/水蛭/巴豆/甘遂/附子/川乌/朱砂/雄黄/麝香等十几样**用户根本
        # 没提过**的药名。现在改用 `user_note`（人话版、无具体药名）。
        parts.append(f"**先按你的状态定边界（{'、'.join(CN.labels(states))}）**：")
        for s in states:
            parts.append(f"- {s.get('user_note') or s.get('note')}")
    hits = ctx.get("hits") or []
    drugs = [h for h in hits if h.get("kind") == "drug"]
    if drugs:
        parts.append("**用药纪律（不能省的几条）**：")
        for h in drugs:
            txt = (h.get("verdict") or "").strip()
            if txt:
                parts.append(f"- **{h.get('name', '?')}**：{txt}")
        return "\n".join(parts)
    if ctx.get("stop") or (ctx.get("tags") and ctx.get("hits")):
        parts.append("**用药纪律**：正在服用的西药（降压/降糖/抗凝等）"
                     "**不要因为讨论中药而自行停药或减量**；要不要加用中药、"
                     "怎么加，先告知主治医生和药师；出现症状加重或异常出血"
                     "（黑便、瘀斑、牙龈出血不止）立即就诊。")
    elif any(h.get("kind") == "herb" for h in hits):
        # P0-2「凡涉及药食建议，必输出原处方药保护」——即使本轮没有西药命中，
        # 也要把这条通用边界说清（用户可能没提，但漏掉它的代价不对称）。
        parts.append("**用药纪律**：如果你正在服用降压/降糖/抗凝等他汀类处方药，"
                     "**不要因为讨论中药或食疗而自行停药、减量**；"
                     "要加用任何中药前先告知主治医生和药师。")
    if not parts:
        return ""
    return "\n".join(parts)


def _fill_referral(ctx: dict) -> str:
    """就医与排查：只列**检查项**。

    刻意**不复述** `SCREENING[k]['say']`——那段话与 `must_say` 高度重叠，
    两个都补会在同一段回答里出现两遍"降压药不能停、血压<140/90"。
    """
    from app.safety import scripts as SG
    keys = ctx.get("screening") or []
    items = [SG.SCREENING[k] for k in keys if k in SG.SCREENING]
    if not items:
        return ""
    lines = ["**建议先做的排查（排掉「长得像亚健康、其实是疾病」的情况）**："]
    for it in items:
        lines.append(f"- {it['trigger']} → {'；'.join(it['tests'])}")
    return "\n".join(lines)


def _fill_actions(ctx: dict) -> str:
    """可执行部分：只在框架要求四维（症状调理类）时补，且用通用安全方向。"""
    from app import framework as FW
    plan = ctx.get("plan")
    if not plan or FW.DIM_MERIDIAN not in (plan.dimensions or []):
        return ""
    return ("**可执行的方向（通用的，不依赖具体辨证）**："
            + FW.action_dims_text() + "。")


def _fill_layer(ctx: dict) -> str:
    """分层判断：**只在信息足够时**补。

    缺信息时绝不补——为凑模块而下证型结论，与"信息不足不得下证型结论"
    是直接冲突的（这一条比"模块齐全"优先级更高）。
    """
    from app.safety import rules as R
    if ctx.get("gaps"):
        return ""
    tags = set(ctx.get("tags") or [])
    ben: list[str] = []
    biao: list[str] = []
    if R.TAG_COLD in tags:
        ben.append("阳气不足")
    if R.TAG_YIN_DEF in tags:
        ben.append("阴液不足")
    if R.TAG_DAMPNESS in tags:
        biao.append("湿浊内停（痰湿/湿困）")
    if R.TAG_DAMP_HEAT in tags:
        biao.append("湿热内蕴")
    if not (ben or biao):
        return ""
    seg = []
    if ben:
        seg.append(f"底子（本）偏**{'、'.join(ben)}**")
    if biao:
        seg.append(f"当前壅滞的（标）在**{'、'.join(biao)}**")
    return ("**分层看**：" + "，".join(seg)
            + "。这只是从你说过的线索推的方向，具体是哪一脏、要不要用药，"
              "还得靠舌脉面诊确定。")


_FILLERS = {
    "verdicts": _fill_verdicts,
    "risk": _fill_risk,
    "referral": _fill_referral,
    "actions": _fill_actions,
    "layer": _fill_layer,
    # questions 由 inquiry.ensure_questions 负责，这里不重复
}


def expected(ctx: dict) -> list[str]:
    """本轮**允许补全**的模块（哪些模块的缺失会造成实质损失）。

    判据全部来自确定性事实（命中项 / 西药 / 排查项 / 框架 / 缺口 / 状态约束），
    不看模型输出——所以补全内容不会与模型的自述打架。
    """
    from app import framework as FW

    exp: list[str] = []
    hits = ctx.get("hits") or []
    plan = ctx.get("plan")
    cat = getattr(plan, "category", "") if plan else ""
    # 知识解释类问题不补"结论/可执行/分层"——那三类只有"在给自己求方案"
    # 时才成立；给"什么是阴虚"补一段"分层看：你底子偏阴液不足"是答非所问。
    is_know = (cat == FW.CAT_KNOW)
    if not is_know and any(h.get("origin", "message") == "message"
                           and h.get("kind") != "drug" for h in hits):
        exp.append("verdicts")
    # 风险管理的触发面比"命中西药"更宽（P0-2）：只要真的给了药食建议，
    # 就必须带"原处方药保护"——不能自行停减量 + 先告知医生药师 + 预警阈值。
    # 这是"凡涉及药食建议必输出"的那一条，不再依赖是否恰好命中西药规则。
    if (any(h.get("kind") == "drug" for h in hits) or ctx.get("stop")
            or (not is_know and any(h.get("kind") == "herb" for h in hits))
            or ctx.get("states")):
        exp.append("risk")
    if ctx.get("screening"):
        exp.append("referral")
    if cat == FW.CAT_ADVICE:
        exp.append("actions")
        # 分层判断只在"信息够了 + 在求方案"时才允许补——
        # 缺信息时补它等于凭空下证型结论
        if not ctx.get("gaps"):
            exp.append("layer")
    return exp


def enforce(answer: str, ctx: dict | None = None) -> tuple[str, list[str]]:
    """输出前自检：缺哪个模块就补哪个（最多补 MAX_APPEND 段）。

    此外还做两件**不占 MAX_APPEND 名额**的确定性修正（它们不是"补模块"，
    而是"纠错"，优先级高于补全）：

      · **档位一致性**（P0-1）：回答里若出现与档位相反的口子
        （"对证但有条件"却写"实在想就少量"、"不建议"却写"会伤身体"），
        追加一句以档位为准的更正——档位是唯一判定点，不允许被正文推翻；
      · **药名外泄**（第五轮强化）：回答里出现用户**从未提及**的药食名时，
        直接把它**概化掉**（「阿司匹林、华法林」→「同类西药」），
        而不是仅补一句"那只是举例"——名字留在正文里，用户照样会记下来。

    2026-09-16 第五轮（版本退化）：① 先剥掉旧版遗留的流程语言引子
    （"再补充几点…"重复两遍的来源）；② 补漏段并入正文、不再用元叙述；
    ③ 无论有没有补模块，**都要做一次泄漏概化**（模型正文也可能外泄）。

    Returns: (最终回答, 由代码补上的模块 key 列表)。
    """
    from app import constraints as CN

    answer = _strip_flow(answer or "")
    ctx = ctx or {}
    if not ctx.get("enforce", True):
        return answer, []
    miss = missing(answer)
    expects = ctx.get("expects") or expected(ctx)
    todo = [m for m in miss if m in expects and m in _FILLERS]
    # 按重要度排序：结论 > 风险 > 排查 > 可执行 > 分层
    order = {"verdicts": 0, "risk": 1, "referral": 2, "actions": 3, "layer": 4}
    todo.sort(key=lambda m: order.get(m, 9))

    segs: list[str] = []
    added: list[str] = []
    for m in todo[:MAX_APPEND]:
        seg = _FILLERS[m](ctx).strip()
        if seg:
            segs.append(seg)
            added.append(m)
    fix = tier_fix(answer, ctx)
    if fix:
        segs.append(fix)
        added.append("tier_fix")

    out = answer if not segs else answer.rstrip() + _HEAD + "\n\n".join(segs) + "\n"

    # ---- 泄漏自检（第五轮）：把"用户没提过"的药食名概化掉 ----
    # 作用对象是**最终整段文本**：模型正文与补漏段一视同仁。
    allowed = CN.allowed_names(ctx.get("question") or "",
                               ctx.get("profile") or {},
                               ctx.get("hits") or [])
    out, leaked = CN.scrub(out, allowed)
    if leaked:
        added.append("deleak")
    return out, added


# ---------------------------------------------------------------------------
# 档位一致性自检（P0-1：档位是唯一判定点，不许被正文推翻）
# ---------------------------------------------------------------------------
# 实测缺陷：同一条命中被三个环节各自表述——标题「对证但有条件」、
# 正文「不能做」、操作指南「如果你实在想喝……」。
# 提示词侧已经用 `tiers.TIER_SPEC['never']` 把口子按档封死；
# 这里是**程序化那一道**：只要回答里仍然出现与该档相反的口子，就追加更正。
_OPENING_RE = re.compile(
    r"如果你实在(?:想|要|馋)|实在忍不住|偶尔(?:喝|吃|用)(?:一|两)?次(?:也)?"
    r"(?:没)?(?:关系|无妨|可以)|少量(?:尝|喝|吃|用)(?:一点)?(?:问题不大|也可以|无妨)|"
    r"其实也可以(?:少|偶尔)")


def tier_violations(answer: str, ctx: dict) -> list[dict]:
    """回答里与档位相反的表述（按命中项逐条检查）。"""
    from app.safety import tiers as T
    a = answer or ""
    out: list[dict] = []
    for h in (ctx.get("hits") or []):
        tier = h.get("tier")
        if tier not in (T.TIER_FORBID, T.TIER_NOT_INDICATED, T.TIER_CONFIRM):
            continue
        for phrase in T.TIER_SPEC.get(tier, {}).get("never", ()):
            if phrase and phrase in a:
                out.append({"key": h.get("key"), "name": h.get("name", "?"),
                            "tier": tier, "tier_label": h.get("tier_label", ""),
                            "phrase": phrase, "kind": "档位不符的措辞"})
        # 通用的"留口子"句式：只在"明确禁止 / 不建议"两档算违规
        if tier in (T.TIER_FORBID, T.TIER_NOT_INDICATED):
            m = _OPENING_RE.search(a)
            if m:
                out.append({"key": h.get("key"), "name": h.get("name", "?"),
                            "tier": tier, "tier_label": h.get("tier_label", ""),
                            "phrase": m.group(0), "kind": "档位之外留口子"})
    return out


def tier_fix(answer: str, ctx: dict) -> str:
    """把"与档位不符"的部分以档位为准更正（返回更正段，无违规返回空串）。"""
    from app.safety import tiers as T
    vs = tier_violations(answer, ctx)
    if not vs:
        return ""
    seen: set[str] = set()
    items: list[str] = []
    for v in vs:
        if v["name"] in seen:
            continue
        seen.add(v["name"])
        thesis = T.TIER_SPEC.get(v["tier"], {}).get("thesis", "")
        items.append(f"- **{v['name']}** 的档位是【{v['tier_label']}】：{thesis}"
                     f"前面若出现过「{v['phrase']}」这类说法，**不适用于它**。")
    return ("**口径更正（同一件事只能说一个档位）**：\n" + "\n".join(items))


# ---------------------------------------------------------------------------
# 药食名外泄守卫（P1-3：规则库内部的同类举例不得外泄成建议）
# ---------------------------------------------------------------------------
def foreign_fix(answer: str, ctx: dict) -> str:
    """回答里出现用户未提及、本轮也未命中的药食名 → 补一句澄清。

    2026-09-16 第四轮（P1）：澄清话术**不再复述外泄的药名**。实测反馈：
    守卫本意是"这些只是举例"，但把名字再列一遍（"红参、川芎、红花…"）
    等于二次强调，用户更容易记住并自行加用。改为概括表述：
    只说"只是同类举例"，不点名。
    """
    from app import constraints as CN
    hits = ctx.get("hits") or []
    if not hits:
        return ""
    allowed = CN.allowed_names(ctx.get("question") or "",
                               ctx.get("profile") or {}, hits)
    foreign = CN.foreign_herbs(answer or "", allowed)
    if not foreign:
        return ""
    return "**关于上面提到的其它药名**：" + CN.FOREIGN_GUARD_NOTE


def render_contract() -> str:
    """契约 → 注入提示词的区块（提示词侧的第一层保障）。"""
    lines = ["【输出契约（每轮都必须齐全，缺哪个补哪个，不由你决定）】",
             "本轮回答必须包含以下模块："]
    for i, (_k, label) in enumerate(MODULES, 1):
        lines.append(f"  {i}. {label}")
    lines.append("落笔前自己过一遍：**哪个模块没写到就补上**，不要直接输出。"
                 "若某项确实无从给出（如没有任何需要排查的风险），"
                 "用一句话说明「这一项不适用」也可以，但不能整块消失。")
    # 2026-09-16 第四轮（P1）：内部流程语言外泄 + 举例药名外泄
    lines.append(
        "【面向用户的文字里禁止出现的东西】\n"
        "1. **内部流程语言**：「系统完整性检查」「按提示词要求」「输出契约"
        "第 X 条」「上面缺失的模块」这类机制口径**不得**进入回答——"
        "要补内容就直接补，用「再补充几点」这类人话引出；\n"
        "2. **解释机制时的同类举例**：需要提到同类药材/食材时用概括表述"
        "（如「其他参类」「活血类」），**不要列出用户没有提到的具体药名**"
        "——用户会把举例当成推荐。")
    return "\n".join(lines)
