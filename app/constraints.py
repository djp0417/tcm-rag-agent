# -*- coding: utf-8 -*-
"""状态类约束 + 药食名外泄守卫（P1-3）。

两个问题，同一条思路：**输入侧已经确定的约束，不许在生成环节被绕过。**

一、状态类约束的优先级
----------------------
实测失效（反馈 P1-3）：用户明说「正在备孕」，系统却把「抗凝西药」作为主判断
分支，把备孕**降为背景信息**，还在结论里引入了用户从未提及的药材名。
后续测试中备孕约束能被识别——说明**不是稳定规律，而是随机**。
凡是"有时对、有时错"的规则，都必须从提示词里搬进代码：

    判断顺序强制为：① 先识别当前人的**状态约束**
                   → ② 再辨证
                   → ③ 最后匹配药食规则
    状态类约束（备孕/妊娠/哺乳/慢病/在服西药/年龄）的优先级
    **高于任何由药名触发的规则**。

二、药食名外泄
--------------
规则库为了让机制说清楚，内部会举例同类药材（附子的 why 里就有川乌、草乌、甘草）。
这些举例是**给模型看的背景**，一旦被写进结论，用户会以为"系统建议我用这个"。
所以输出前要做一次**确定性**核对：

    回答里出现的药食名 ∈ 用户提到的 + 档案里的 + 本轮命中项 + 日常食养常用品
    不在这个集合里 → 视为"外泄"，由 contract.enforce 补一句澄清。

为什么是"补一句澄清"而不是删掉那句：删句子会破坏上下文连贯性（而且文本里
那句话可能正是安全解释的一部分）。澄清的作用是**把"举例"和"建议"分开**，
用户不会被引导去自行加用。
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# 一、状态类约束
# ---------------------------------------------------------------------------
# 词表刻意取"用户可能自然说出的口语说法"，与 safety/rules.py 的口径一致。
STATES: tuple[dict, ...] = (
    {"key": "preconception", "label": "备孕",
     "re": r"备孕|准备怀孕|打算要(?:个)?(?:孩子|宝宝)|想要(?:个)?孩子|"
           r"计划要孩子|在要孩子|试试要孩子",
     # `note`：**给模型看**的机制说明（可含同类药名，帮它理解边界），只进提示词。
     # `user_note`：**给用户看**的边界声明——**不得含任何具体药名**。
     # 2026-09-16 第五轮实测缺陷（版本退化）：补全环节把 `note` 原文当输出，
     # 用户完全没提的药名（红花/桃仁/益母草/水蛭/巴豆/甘遂/附子/川乌/朱砂/
     # 雄黄/麝香…）成段倒出，且重复出现。**规则库的作用是判定，不是输出源**；
     # 需要告知用户时，必须按当前场景重新组织语言。
     "note": "备孕状态下，活血化瘀（红花、桃仁、益母草、水蛭）、"
             "峻下逐水（巴豆、甘遂）、有毒药材（附子、川乌、朱砂、雄黄）"
             "以及麝香类芳香走窜之品都应避免；任何中药/食养方先经妇产科确认。",
     "user_note": "备孕期间，**活血化瘀、峻下逐水、有毒以及芳香走窜**"
                  "这几类药材都应避开；任何中药或食养方，先经妇产科确认再谈。"},
    {"key": "pregnancy", "label": "妊娠",
     "re": r"怀孕|孕妇|妊娠|怀了|有了宝宝|孕(?:早|中|晚)期|孕\d+周|产检",
     "note": "妊娠期用药与食养必须由产科/中医师把关，不得自行加用任何中药。",
     "user_note": "孕期用药与食养要由产科 / 中医师把关，"
                  "**不要自行加用任何中药**；出现持续不适，先让产科医生看一眼。"},
    {"key": "lactation", "label": "哺乳期",
     "re": r"哺乳|喂奶|母乳|坐月子|月子(?:里|中)",
     "note": "哺乳期同样不能自行加用中药——部分成分会进入乳汁。",
     "user_note": "哺乳期同样不要自行加用中药——部分成分会进入乳汁。"},
    {"key": "age_child", "label": "儿童/婴幼儿",
     "re": r"小孩|孩子|儿童|宝宝|婴儿|婴幼儿|新生儿|我儿子|我女儿",
     "note": "儿童用药量与时机的判断标准与成人不同，必须由儿科/中医师定。",
     "user_note": "孩子的用药量与时机和成人不同，**不能按成人量减半自行试用**，"
                  "要由儿科 / 中医师来定。",
     "guard": "child_age"},     # 需年龄守卫（见 scan._child_context_ok）
    {"key": "age_elderly", "label": "高龄",
     "re": r"老年人|七十|八十|九十|(?:7\d|8\d|9\d)\s*岁|我母亲|我父亲|奶奶|爷爷|"
           r"姥姥|姥爷",
     "note": "高龄人群多重用药常见，相互作用风险高，任何调整先告知医生。",
     "user_note": "高龄人群常同时吃多种药，相互作用风险高，"
                  "任何调整先告知医生。"},
)

_HAVE_RE = {s["key"]: re.compile(s["re"]) for s in STATES}
_LABEL = {s["key"]: s["label"] for s in STATES}


def detect(text: str, profile: dict | None = None,
           conv_id: int | None = None) -> list[dict]:
    """识别本轮生效的状态类约束（含**本会话时间线 + 档案**，不只本轮）。"""
    from app import storage
    prof = profile if profile is not None else storage.get_profile(conv_id)
    blob = text or ""
    if conv_id is not None:
        try:
            from app.intake import timeline_entries
            blob += "\n" + "；".join(timeline_entries(conv_id))
        except Exception:
            pass
    if prof:
        blob += "\n" + "\n".join(str(v) for v in prof.values())

    out: list[dict] = []
    for st in STATES:
        m = _HAVE_RE[st["key"]].search(blob)
        if not m:
            continue
        if st.get("guard") == "child_age":
            from app.safety.scan import _child_context_ok
            if not _child_context_ok(blob):
                continue
        out.append({"key": st["key"], "label": st["label"],
                    "matched": m.group(0), "note": st["note"],
                    # ★ 必须一起带出来，否则 `_fill_risk` 只能回退到 note
                    # （= 内部规则原文），外泄就又会发生（2026-09-16 第五轮实测）。
                    "user_note": st.get("user_note", "")})
    return out


def labels(states) -> list[str]:
    out: list[str] = []
    for s in (states or []):
        lb = s.get("label") if isinstance(s, dict) else _LABEL.get(str(s), str(s))
        if lb and lb not in out:
            out.append(lb)
    return out


def render_block(states) -> str:
    """状态类约束 → 注入提示词的强制块（判断顺序 + 优先级）。"""
    if not states:
        return ""
    lines = [
        "【⚠️ 状态类约束优先（本轮的判断顺序由系统固定，不由你决定）】",
        f"识别到的状态约束：{'、'.join(labels(states))}",
        "**强制判断顺序**：① 先按上面这个状态确定安全边界 "
        "→ ② 再做辨证/证型判断 → ③ 最后才匹配药食规则。",
        "**优先级**：状态类约束的效力**高于任何由药名触发的规则**。",
        "**禁止**把状态类约束降级为「背景信息」或一句带过——"
        "实测缺陷：用户明说「正在备孕」，回答却把抗凝西药当主分支，"
        "备孕只在角落里提了一句（有时对、有时错，所以现在由代码固定顺序）。",
    ]
    for s in states:
        lines.append(f"  · 【{s.get('label')}】{s.get('note')}")
    lines.append("另外：**不要引入用户没有提过的药食名**——规则库里为了解释机制"
                 "举例的同类药材（如川乌、草乌、水蛭之类）不得写进建议，"
                 "只讨论「他提到的」与「你明确推荐的替代方向」。")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 二、药食名外泄守卫
# ---------------------------------------------------------------------------
# 日常食养常用品：推荐它们不构成"引入用户没提过的药"。其余药食名
# 只要用户没提到、本轮也没命中，就不该出现在回答的**建议**里。
SAFE_STAPLES: tuple[str, ...] = (
    "山药", "莲子", "茯苓", "陈皮", "生姜", "干姜", "红枣", "大枣", "小米",
    "大米", "粳米", "薏米", "赤小豆", "红豆", "百合", "银耳", "山楂", "麦芽",
    "荷叶", "冬瓜", "萝卜", "葱", "蒜", "桂圆", "枸杞", "菊花", "生姜片",
    "温水", "热粥", "白扁豆", "芡实", "南瓜", "胡萝卜", "菠菜", "白菜",
    # 2026-09-16 第五轮：把家常食材补进来，避免"建议喝梨水"这类正常
    # 食养建议被泄漏概化误伤（概化只针对**药材名**，不是食材）。
    "梨", "雪梨", "冰糖", "鸡肉", "鸡汤", "瘦肉", "鸡蛋", "牛奶", "莲藕",
    "荸荠", "枇杷", "蜂蜜", "梨水", "梨汤", "姜茶", "姜汤", "米粥", "山药粥",
)


# 规则库里以"同类举例"形式出现、但**没建独立规则**的药名。它们同样属于
# "用户没提过就不该出现在回答里"，必须一并纳入泄漏检测（第五轮实测：
# 外泄文本里既有建了规则的名字，也有只在 why/not_for 里当例子的名字）。
EXTRA_LEAK_NAMES: tuple[str, ...] = (
    "麝香", "红曲", "血脂康", "川乌", "草乌", "马兜铃", "何首乌", "雷公藤",
    "土三七", "关木通", "牵牛子", "商陆", "芫花", "大戟", "水蛭", "藏红花",
    "银杏叶", "穿山甲", "全蝎", "蜈蚣",
)

# **通用类别词**：这不是"具体药名"，而是类别 / 纪律用语，必须允许出现。
# 反例（实测风险）：把「降压药**不能自行停药**」里的"降压药"也概化掉，
# 就破坏了最重要的安全信息。类别名一律豁免。
GENERIC_MED_WORDS: tuple[str, ...] = (
    "降压药", "降糖药", "血糖药", "抗凝药", "抗血小板药", "利尿剂",
    "降脂药", "他汀类", "甲状腺用药", "处方药", "西药", "中成药", "中药",
    "活血化瘀", "活血类", "参类", "膏方", "补品",
)


def _herb_names() -> tuple[str, ...]:
    """全部药食名（规则库的 name + aliases，**含西药名**，长的优先）。

    2026-09-16 第五轮：此前只收药材/食药同源，西药名不在集合里 →
    实测外泄文本里一半是西药名（阿司匹林/华法林/氯吡格雷/红曲/血脂康…），
    却检测不到。现在把西药 aliases 与 `EXTRA_LEAK_NAMES` 一并收进来。
    """
    from app.safety import rules as R
    names: set[str] = set()
    for rule in R.ALL_HERBS:
        names.add(rule.name)
        # 展示名里的括号说明不是药名本身（"人参类（红参/生晒参/西洋参）"）
        names.add(re.split(r"[（(]", rule.name)[0])
        for a in rule.aliases:
            if len(a) >= 2:
                names.add(a)
    for rule in R.DRUG_CLASSES:
        for a in (getattr(rule, "aliases", ()) or ()):
            if len(a) >= 2:
                names.add(a)
    names |= set(EXTRA_LEAK_NAMES)
    names |= set(SAFE_STAPLES)
    return tuple(sorted((n for n in names if len(n) >= 2),
                        key=len, reverse=True))


_ALL_NAMES: tuple[str, ...] = ()


def all_names() -> tuple[str, ...]:
    global _ALL_NAMES
    if not _ALL_NAMES:
        _ALL_NAMES = _herb_names()
    return _ALL_NAMES


def allowed_names(text: str, profile: dict | None = None,
                  hits=None, extra: str = "") -> set[str]:
    """本轮**允许出现**的药食名集合。

    来源：用户本轮原话 + 本会话档案 + 本轮命中项（含命中词）+ 日常食养常用品
    + 调用方显式给出的补充文本。
    """
    blob = (text or "") + "\n" + (extra or "")
    if profile:
        blob += "\n" + "\n".join(str(v) for v in profile.values())
    allowed: set[str] = set()
    for h in (hits or []):
        d = h if isinstance(h, dict) else (
            h.to_dict() if hasattr(h, "to_dict") else {})
        for k in ("name", "matched"):
            v = (d.get(k) or "").strip()
            if v:
                allowed.add(v)
                allowed.add(re.split(r"[（(]", v)[0])
    for n in all_names():
        if n in blob:
            allowed.add(n)
    allowed |= set(SAFE_STAPLES)
    allowed |= generic_ok_names()          # 类别 / 纪律用语一律放行
    return {a for a in allowed if a}


def foreign_herbs(answer: str, allowed) -> list[str]:
    """回答里出现、但不在允许集合里的药食名（按出现顺序，已去重）。"""
    allowed = set(allowed or ())
    found: list[tuple[int, str]] = []
    for n in all_names():
        if n in allowed:
            continue
        i = (answer or "").find(n)
        if i >= 0:
            found.append((i, n))
    found.sort()
    out: list[str] = []
    for _i, n in found:
        # 已被更长的名字覆盖掉的（如"薏米"⊂"红豆薏米"）不重复报
        if any(n != o and n in o for o in out):
            continue
        out.append(n)
    return out


FOREIGN_GUARD_NOTE = (
    "（上面若出现了**你没有提到过**的药食名，那只是解释机制时的同类举例——"
    "请以你实际在问的那一样为准，**不要据此自行加用**其它药材。）")

_GENERIC_OK: frozenset[str] = frozenset()


def generic_ok_names() -> frozenset[str]:
    """**放行**的类别 / 纪律用语（不是"具体药名"）。

    判据：`name` 里带"类"的规则名（"活血化瘀类中药"）、西药类别名
    （`DrugRule.cls`）、以及 `GENERIC_MED_WORDS`。这些出现在回答里是**必要的**
    （"降压药不能自行停"），概化掉会破坏安全信息。
    """
    global _GENERIC_OK
    if not _GENERIC_OK:
        from app.safety import rules as R
        s: set[str] = set(SAFE_STAPLES) | set(GENERIC_MED_WORDS)
        for r in R.ALL_HERBS:
            if "类" in r.name:
                s.add(r.name)
                s.add(re.split(r"[（(]", r.name)[0])
        for r in R.DRUG_CLASSES:
            nm = getattr(r, "cls", "") or ""
            if nm:
                s.add(nm)
                s.add(nm.replace(" ", ""))
        _GENERIC_OK = frozenset(n for n in s if n)
    return _GENERIC_OK


_GENERIC_CACHE: dict[str, str] = {}


def generic_label(name: str) -> str:
    """把"用户没提过的具体药食名"折成**类别概括**（用于替换，不点名）。

    「阿司匹林」→「同类西药」；「丹参」→「其他活血化瘀类药材」。
    刻意**不复述名字**——上一版的澄清句把名字再列一遍，等于二次强调。
    """
    if name in _GENERIC_CACHE:
        return _GENERIC_CACHE[name]
    from app.safety import rules as R
    lab = ""
    for rule in R.ALL_HERBS:
        if (name == rule.name or name == re.split(r"[（(]", rule.name)[0]
                or name in rule.aliases):
            nm = rule.name
            lab = ("其他" + nm.split("类")[0] + "类药材") if "类" in nm \
                else "其他同类药材"
            break
    if not lab:
        for rule in R.DRUG_CLASSES:
            if name == rule.cls or name in (getattr(rule, "aliases", ()) or ()):
                lab = "同类西药"
                break
    _GENERIC_CACHE[name] = lab or "其他同类药材"
    return _GENERIC_CACHE[name]


# 相邻名字之间若只有这些字符，说明它们是一个"举例列表"→ 合并成一次概括。
# （「阿司匹林、华法林、氯吡格雷等」→「同类西药等」，而不是三个词重复）
_JOIN_RE = re.compile(r"^[\s、，,；;／/·和与或及等]+$")


def scrub(text: str, allowed) -> tuple[str, list[str]]:
    """输出前的**泄漏自检**：把回答里"用户没提过"的药食名概化掉。

    2026-09-16 第五轮（退化 2）反馈要求：**禁止输出用户未提及的药食名，
    包括"同类举例"；命中则重写**。所以这里不是"补一句澄清"（名字还在），
    而是**真的把名字换掉**：
        「（阿司匹林、华法林、氯吡格雷等）」→「（同类西药等）」
        「丹参、三七作食养」→「其他活血化瘀类药材作食养」

    Returns: (处理后文本, 被概化掉的名字列表)。
    """
    t = text or ""
    allowed = set(allowed or ())
    spans: list[list] = []
    for n in all_names():
        if n in allowed:
            continue
        i = t.find(n)
        while i >= 0:
            spans.append([i, i + len(n), n])
            i = t.find(n, i + len(n))
    if not spans:
        return t, []
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    merged: list[list] = []
    for s in spans:
        if merged and s[0] < merged[-1][1]:      # 被更长/更早的同族名覆盖
            continue
        merged.append(s)
    packed: list[list] = []
    for s in merged:
        if packed:
            gap = t[packed[-1][1]:s[0]]
            if _JOIN_RE.match(gap):              # 举例列表 → 合并
                packed[-1][1] = s[1]
                packed[-1][2] += "|" + s[2]
                continue
        packed.append(s)
    out: list[str] = []
    found: list[str] = []
    labels: set[str] = set()
    last = 0
    for st, en, names in packed:
        out.append(t[last:st])
        lab = generic_label(names.split("|")[0])
        out.append(lab)
        labels.add(lab)
        found.extend(names.split("|"))
        last = en
    out.append(t[last:])
    res = "".join(out)
    # 举例列表里夹着一个"用户提过的名字"时，两侧会各生成一个同款概括词
    # （「…A、当归、A」）→ 把相邻重复的合并掉。
    for lab in labels:
        res = re.sub(re.escape(lab) + r"(?:\s*[、，,]\s*" + re.escape(lab) + r")+",
                     lab, res)
    return res, found


# ---------------------------------------------------------------------------
# 三、状态 → 风险标签（让"状态约束"能直接参与档位判定）
# ---------------------------------------------------------------------------
# 2026-09-16 第五轮：`constraints.detect` 得到的状态此前**只进提示词**，
# 而档位判定（`tiers.classify`）读的是 `scan.tag_from_text` 出来的 tags——
# 两条链路各算各的，于是"识别到妊娠"却"档位没跟上"。
# `state_tags()` 把状态折成 tag，由调用方并入 tags，状态才真正决定档位。
_STATE_TAG: dict[str, str] = {
    "preconception": "pregnancy",
    "pregnancy": "pregnancy",
    "lactation": "lactation",
    "age_child": "child",
    "age_elderly": "elderly",
}


def state_tags(states) -> set[str]:
    """状态约束 → 风险标签集合（并入 tags 后，`classify` 才能按状态定档）。"""
    from app.safety import rules as R
    ok = {R.TAG_PREGNANCY, R.TAG_LACTATION, R.TAG_CHILD, R.TAG_ELDERLY}
    out: set[str] = set()
    for s in (states or []):
        k = s.get("key") if isinstance(s, dict) else str(s)
        t = _STATE_TAG.get(str(k))
        if t in ok:
            out.add(t)
    return out


def absolute_state_labels(states) -> list[str]:
    """本轮命中的**围产期状态**（备孕 / 妊娠 / 哺乳）人话标签。

    追问与排查项要据此切换：妊娠期不该问抗凝/出血项、不该给备孕套餐，
    而应给产科就诊引导。
    """
    out: list[str] = []
    for s in (states or []):
        k = str(s.get("key") if isinstance(s, dict) else s)
        if k in ("preconception", "pregnancy", "lactation"):
            lb = s.get("label") if isinstance(s, dict) else k
            if lb not in out:
                out.append(lb)
    return out


def is_perinatal(states) -> bool:
    """是否处于**围产期**（备孕/妊娠/哺乳）——追问与排查项要据此切换。"""
    for s in (states or []):
        k = str(s.get("key") if isinstance(s, dict) else s)
        if k in ("preconception", "pregnancy", "lactation"):
            return True
    return False
