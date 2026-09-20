# -*- coding: utf-8 -*-
"""安全扫描器：把「文本 / 用户档案」翻译成一组结构化的 SafetyHit。

三条设计原则
------------
1. **零模型依赖**：纯字符串匹配 + 规则表，可复现、可单测、零成本。
   这一层是"宁可多提醒一次"的地方——它产出的是**提醒**，不是定罪，
   所以容忍适度误报（例如用户只是问"附子是什么"也会触发安全说明）。
2. **区分"提到"与"在吃"**：命中同一味药，但用户是在问「附子理中丸能不能吃」
   还是「我这两天在吃附子理中丸」，安全强度不同。用 `_TAKING` 词表判断，
   在 hit 上打 `taking` 标记，供话术层调整措辞。
3. **交叉升级**：药材规则自带 tags；一旦与用户档案里的慢病 tag 相交
   （例如规则 tags 含 hypertension 且用户档案有高血压），
   把该 hit 的 level 从 medium 升到 high，并附上"与你的情况相关"的说明。

用法：
    from app.safety import scan_text, scan_profile, merge_hits, build_block
    hits = merge_hits(scan_text(msg), scan_profile(profile))
    if hits:
        block = build_block(hits)        # 注入提示词的强制区块
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.safety import rules as R


@dataclass
class SafetyHit:
    """一次安全命中。"""
    kind: str                 # herb / drug / population
    key: str                  # 规则 key（去重用）
    name: str                 # 展示名
    level: str                # high / medium / info
    sub: str = ""             # toxic=毒性药材 / herb=需辨证药材 / food=食药同源
                              # / drug=西药类别 / pop=特殊人群（分级判据要用）
    why: str = ""             # 客观说明（药性 / 风险机制）
    not_for: str = ""         # 哪类人不宜
    verdict: str = ""         # 「所以对你而言…」模板（**退化为兜底**，
                              # 实际判读由 safety/tiers.py::render 按条件生成）
    matched: str = ""         # 实际命中的词（便于解释"为什么提这一条"）
    tags: list[str] = field(default_factory=list)
    taking: bool = False      # 用户是否正在服用 / 正在使用
    escalated: bool = False   # 是否因与档案交叉而升级为 high
    origin: str = "message"   # message=本轮原话命中 / profile=长期档案带出
    # ---- 五档分级（2026-09-16「安全是分级不是开关」+ 第三轮 P0-1 档位粒度）----
    tier: str = ""            # forbid / confirm / not_needed / conditional / ok
    tier_label: str = ""      # 中文档位名
    tier_tone: str = ""       # 该档位的语气要求（注入提示词）
    cond_labels: list[str] = field(default_factory=list)  # 命中了用户的哪些条件
    unassessed_gaps: list[str] = field(default_factory=list)  # 哪些条件还没评估
    # 「这一档是凭什么定下来的」一句话（tiers.tier_basis）。**必须进 to_dict**——
    # 第三轮首次落地时忘了放进 to_dict，结果前端/落库都拿不到它，
    # `h.tier_basis` 恒为 undefined 并静默退回 cond_labels：功能看着"实现了"，
    # 实际一条都没下发（不报错、只变差，典型的静默失效）。
    tier_basis: str = ""
    # 2026-09-16 第三轮（P0-1 档位粒度）：从规则带来的两个"定档依据"。
    # `contra` 命中 = 方向相反（禁忌）；`indication` 为空交集而信息够 = 无适应症。
    # 不带上这两个字段，分级引擎就只能看 tags，于是"鹿茸 × 阴虚"与
    # "鹿茸 × 健康人"会被定成同一档。
    contra: list[str] = field(default_factory=list)
    indication: list[str] = field(default_factory=list)
    # 上面那几项只是"元信息"；这一项才是**给人看的判读正文**（tiers.render 按
    # 档位×条件生成）。前端必须显示它，不能显示 rules.py 里写死的 verdict——
    # 那段 verdict 是按最坏情况写的，血虚无湿的人问阿胶也会看到"不适合"。
    reading: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.kind, "key": self.key, "name": self.name,
                "level": self.level, "sub": self.sub, "why": self.why,
                "not_for": self.not_for,
                "verdict": self.verdict, "matched": self.matched,
                "tags": self.tags, "taking": self.taking,
                "escalated": self.escalated, "origin": self.origin,
                "tier": self.tier, "tier_label": self.tier_label,
                "tier_tone": self.tier_tone, "tier_basis": self.tier_basis,
                "cond_labels": self.cond_labels,
                "unassessed_gaps": self.unassessed_gaps,
                "contra": self.contra,
                "indication": self.indication,
                "reading": self.reading,
                "tag_labels": R.tag_labels(self.tags)}


# ---------------------------------------------------------------------------
# 「正在吃」的判据
# ---------------------------------------------------------------------------
# 出现这些词说明用户不只是"听说"，而是**已经在服用/长期在吃**，
# 这时话术要从"是否适合"升级为"请暂停并咨询"。
_TAKING = ("我在吃", "我在喝", "我吃了", "我喝了", "正在吃", "正在喝", "在服用",
           "天天喝", "每天都吃", "每天吃", "买了", "最近吃", "刚吃", "喝了",
           "一直在吃", "自己买", "正在服用", "服用中", "开始吃",
           "在喝", "吃了一个", "喝了一",
           # 2026-09-16 补：「在吃」是口语里最常见的说法，
           # 却被"正在吃/在服用"的写法漏掉 —— 实测「有高血压在吃氨氯地平」
           # 判成 taking=False，附子理中丸同句也被连带判 False，
           # 话术从「请暂停并咨询」降级为「是否适合你」。
           # 安全性检查：不引入「朋友推荐我吃阿胶」的误判
           # （该句没有"在"字前缀），回归见 tools/test_safety_rules.py。
           "在吃")

# 「朋友推荐」「听说」「网上说」**故意不算在服用**：
# 这词一开始在 _TAKING 里，结果实测出现反例——用户说「朋友推荐我吃阿胶」，
# 系统把阿胶标成 taking=True，话术升级为「请暂停服用」，
# 更糟的是 intake.record_herbs 会据此把阿胶**永久写进档案**「在服中药/食疗」，
# 之后每一轮都当成"正在吃"来交叉判断，一路错下去。
# 判据：推荐 ≠ 使用。此时该给的仍是「是否适合你」（由 verdict 模板负责），
# 而不是「请暂停」。
_CONSIDERING = ("朋友推荐", "别人推荐", "听说", "都说", "网上说", "刷到", "看到说")

# ---------------------------------------------------------------------------
# 儿童规则的「年龄守卫」（2026-09-15 上线清单③：模板泄漏修复）
# ---------------------------------------------------------------------------
# 真实事故：用户问"我女儿25岁，湿热怎么调"，回答里冒出
# 「您女儿25岁，是成年人……不存在'按成人量减半'的问题」——
# 用户从头到尾没提儿童、没问用量，这句是儿童规则被"我女儿"一词误触发后
# 凭空插入的内容。根因：「我儿子/我女儿」本身不构成"是儿童"的证据。
# 守卫判据：文本中若出现**成年年龄**（18~99 岁 / 成年人）且没有**儿童年龄**
# 标记（1~12 岁 / N 个月 / 几岁），则儿童规则与 child tag 一律不生效。
_ADULT_AGE_RE = re.compile(r"(?:1[89]|[2-9]\d)\s*岁|成年人|成人|成年")
_CHILD_AGE_RE = re.compile(r"(?<!\d)(?:[1-9]|1[0-2])\s*岁|\d+\s*个月|几岁|多大了")


def _child_context_ok(text: str) -> bool:
    """文本是否具备"讨论儿童"的语境（供 child 规则/tag 的门控）。"""
    if not any(w in text for w in ("小孩", "孩子", "儿童", "宝宝", "婴儿",
                                   "婴幼儿", "新生儿", "我儿子", "我女儿")):
        return False                                # 没提儿童话题，直接不触发
    if _CHILD_AGE_RE.search(text):
        return True                                 # 出现儿童年龄 → 放行
    return not _ADULT_AGE_RE.search(text)           # 无年龄信息 → 宽松放行；
                                                    # 明确给出成年年龄 → 拦下

_LEVEL_ORDER = {"high": 2, "medium": 1, "info": 0}


def _nearest(text: str, pos: int, words: tuple[str, ...], window: int) -> int | None:
    """命中词 pos 附近最近的标记词距离（没出现返回 None）。"""
    seg_start = max(0, pos - window)
    seg = text[seg_start: pos + window]
    best: int | None = None
    for w in words:
        i = seg.find(w)
        while i >= 0:
            d = abs((seg_start + i) - pos)
            if best is None or d < best:
                best = d
            i = seg.find(w, i + 1)
    return best


def _is_taking(text: str, pos: int, window: int = 40) -> bool:
    """命中词附近是否有"我正在吃"的语境。

    取**最近的标记词**而不是"窗口里有没有"：一句话里常同时出现
    「朋友推荐我吃阿胶，可我自己天天喝红豆薏米茶」，
    粗略的窗口判断会把「天天喝」（虽然它属于薏米）算到阿胶头上。
    按距离取最近，阿胶归"推荐"、薏米归"在喝"，各自都对。
    """
    d_take = _nearest(text, pos, _TAKING, window)
    d_cons = _nearest(text, pos, _CONSIDERING, window)
    if d_take is None:
        return False
    if d_cons is None:
        return True
    return d_take <= d_cons


# ---------------------------------------------------------------------------
# 一、症状 / 慢病 → 风险 tag
# ---------------------------------------------------------------------------
# 否定词守卫（2026-09-16）：中文里"没有湿气""不怕冷""血压正常"都会**字面命中**
# 正向关键词。不处理的话，用户明确否认的体征会被当成"有此表现"——
# 实测「我想吃点阿胶补血，舌淡苔薄白，大便正常，**没有湿气**」
# 里的"湿气"被当成 dampness 命中，于是阿胶被判"先化湿再补"。
# 判据：关键词**左侧 3 字内**出现否定词，则该次命中作废；
# 只要还有一处**未被否定**的出现，tag 仍然成立。
_NEGATORS = ("不", "没", "无", "未", "非", "否", "别")
# 分句标点：否定只作用于它所在的**分句**。"我不吃附子，血压高"里的"不"
# 不能把后一句话的"血压高"一起否掉。
_CLAUSE_BREAK = "，,。；;！!？?\n：:"
# 回看窗口。3 字太短——"没有高血压糖尿病"里的"糖尿病"距"没"5 个字，
# 会用一条"没有…"把列表里后面的病名全判成阳性（实测误判 diabetes）。
# 5 字 + 分句截断，两个方向都对：
#   "没有高血压糖尿病" → 糖尿病 判为**已否认** ✓
#   "没有湿气、但是口苦" → 口苦 判为阳性 ✓
_NEG_WINDOW = 5


def _negated(text: str, pos: int, window: int = _NEG_WINDOW) -> bool:
    seg = text[max(0, pos - window):pos]
    cut = max((seg.rfind(c) for c in _CLAUSE_BREAK), default=-1)
    seg = seg[cut + 1:]
    return any(n in seg for n in _NEGATORS)


def _hit_affirmative(text: str, kw: str) -> bool:
    """kw 在 text 里是否有**未被否定**的出现。

    以 `~` 开头的 kw 按**正则**处理——用于需要"排除式"的关键词，
    例如血虚的「舌淡」不能把「舌淡红」（正常舌色）也算上：
    `~舌(?:质)?淡(?!红)`。
    """
    if kw.startswith("~"):
        try:
            pat = re.compile(kw[1:])
        except re.error:
            return kw[1:] in text
        for m in pat.finditer(text):
            if not _negated(text, m.start()):
                return True
        return False
    i = text.find(kw)
    while i >= 0:
        if not _negated(text, i):
            return True
        i = text.find(kw, i + 1)
    return False


# 公开别名：**档案抽取侧也必须用它**（app/intake.py::extract_conditions/extract_drugs）。
# 见 `hit_affirmative` 的用法注释——档案写错一次，会错到用户删会话为止。
hit_affirmative = _hit_affirmative


def tag_from_text(text: str) -> set[str]:
    """从一段话里提取风险 tag（症状 + 已告知的慢病）。"""
    t = text or ""
    tags: set[str] = set()
    for tag, kws in R.SYMPTOM_TAGS:
        if any(_hit_affirmative(t, k) for k in kws):
            tags.add(tag)
    for tag, kws in R.CONDITION_TAGS:
        if any(_hit_affirmative(t, k) for k in kws):
            tags.add(tag)
    # 年龄守卫：「我女儿25岁」里的「我女儿」不能当作"儿童人群"的证据，
    # 否则 child tag 会触发叫停话术、还会交叉升级（模板泄漏的另一半）。
    if R.TAG_CHILD in tags and not _child_context_ok(t):
        tags.discard(R.TAG_CHILD)
    return tags


# ---------------------------------------------------------------------------
# 二、文本扫描
# ---------------------------------------------------------------------------
def scan_text(text: str) -> list[SafetyHit]:
    """扫描一段用户文本，返回命中的药材 / 西药 / 特殊人群规则。"""
    t = text or ""
    if not t.strip():
        return []
    self_tags = tag_from_text(t)
    hits: list[SafetyHit] = []
    seen: set[str] = set()

    # ---- 药材 / 食药同源 ----
    for rule in R.ALL_HERBS:
        m = _first_alias(t, rule.aliases)
        if not m:
            continue
        key = f"herb:{rule.key}"
        if key in seen:
            continue
        seen.add(key)
        hits.append(SafetyHit(
            kind="herb", key=key, name=rule.name, level=rule.level,
            sub=rule.kind,
            why=rule.why, not_for=rule.not_for, verdict=rule.verdict,
            matched=m, tags=list(rule.tags),
            taking=_is_taking(t, t.find(m)),
            contra=list(getattr(rule, "contra", ()) or ()),
            indication=list(getattr(rule, "indication", ()) or ()),
        ))

    # ---- 西药类别 ----
    for rule in R.DRUG_CLASSES:
        m = _first_alias(t, rule.aliases)
        if not m:
            continue
        key = f"drug:{rule.key}"
        if key in seen:
            continue
        seen.add(key)
        hits.append(SafetyHit(
            kind="drug", key=key, name=rule.cls, level="high",
            sub="drug",
            why="", not_for="", verdict=rule.must_say,      # 西药类强制话术直接进 verdict
            matched=m, tags=list(rule.tags),
            taking=_is_taking(t, t.find(m)),
        ))

    # ---- 特殊人群 ----
    for rule in R.POPULATIONS:
        m = _first_alias(t, rule.aliases)
        if not m:
            continue
        # 儿童规则年龄守卫：明确给出成年年龄且无儿童年龄标记时不触发
        # （2026-09-15 修复："我女儿25岁"被误判为儿童人群，回答里凭空
        # 出现"不存在按成人量减半的问题"）
        if rule.key == "child" and not _child_context_ok(t):
            continue
        key = f"pop:{rule.key}"
        if key in seen:
            continue
        seen.add(key)
        hits.append(SafetyHit(
            kind="population", key=key, name=rule.name, level="medium",
            sub="pop",
            verdict=rule.must_say, matched=m, tags=list(rule.tags),
            taking=True,
        ))

    return _escalate(hits, self_tags)


def _first_alias(text: str, aliases: tuple[str, ...]) -> str:
    """返回第一个命中的别名（长的优先，避免「薏米」抢先命中「红豆薏米」）。"""
    for a in sorted(aliases, key=len, reverse=True):
        if a and a in text:
            return a
    return ""


# ---------------------------------------------------------------------------
# 三、档案扫描（跨会话的慢病 / 用药 / 体质）
# ---------------------------------------------------------------------------
def scan_profile(profile: dict | None) -> list[SafetyHit]:
    """把长期档案里的慢病、用药、体质翻译成安全命中。

    档案是**跨会话**的：用户上一次说过「我有高血压、吃氨氯地平」，
    这一次他只在问「阿胶能吃吗」——如果只看本轮文本，
    就会漏掉"高血压 + 滋腻"的交叉风险。所以档案必须一并扫描。

    注意：档案里出现药材/西药 = 用户已确认在服（写入前经过 _is_taking 判定，
    「朋友推荐」这类进不了档案），所以这里一律标 taking=True。
    """
    if not profile:
        return []
    blob = "\n".join(f"{k}：{v}" for k, v in profile.items())
    hits = scan_text(blob)
    for h in hits:
        if h.kind in ("herb", "drug"):
            h.taking = True
    return hits


def annotate_origins(msg_hits: list[SafetyHit],
                     prof_hits: list[SafetyHit],
                     current_text: str) -> list[SafetyHit]:
    """给命中标注来源，并把「本轮也提到了」的档案命中升为本轮命中。

    为什么必须有这一步（真实事故）：档案扫描是无差别的——用户新开会话
    只问「累、想补补」，档案里留着测试期写入的附子/氨氯地平/阿胶，
    六张判读卡直接糊脸，体验是"突兀、答非所问"。
    分流规则：
    - 本轮原话命中的 → origin=message，渲染成完整判读卡（用户正问它）；
    - 档案带出、且本轮原话也提到了的（同名或别名出现）→ 升为 message
      （用户既然又提了，就值得完整展开）；
    - 其余档案命中 → origin=profile，build_block 里只渲染成一行式提醒，
      不再产出整卡。
    """
    t = current_text or ""
    for h in (msg_hits or []):
        h.origin = "message"
    for h in (prof_hits or []):
        names = {h.name, h.matched} - {""}
        # 别名也算：本轮说「红豆薏米茶」而档案里写「薏米（薏苡仁）」
        rule = next((r for r in R.ALL_HERBS if r.name == h.name), None)
        if rule:
            names |= set(rule.aliases)
        h.origin = "message" if any(n and n in t for n in names) else "profile"
    return merge_hits(list(msg_hits or []), list(prof_hits or []))


# ---------------------------------------------------------------------------
# 四、合并 / 升级
# ---------------------------------------------------------------------------
def merge_hits(*groups) -> list[SafetyHit]:
    """合并多组命中并去重（同一 key 取 level 更高的一条）。

    origin 优先级：message > profile——同一味药本轮也提到了，
    就按"本轮命中"完整展开，不再降级为一行式档案提醒。
    """
    best: dict[str, SafetyHit] = {}
    for group in groups:
        for h in (group or []):
            cur = best.get(h.key)
            if cur is None or _LEVEL_ORDER[h.level] > _LEVEL_ORDER[cur.level]:
                if cur:
                    if cur.taking:
                        h.taking = True
                    if cur.origin == "message":
                        h.origin = "message"
                best[h.key] = h
            else:
                if h.taking:
                    cur.taking = True
                if h.origin == "message":
                    cur.origin = "message"
    out = list(best.values())
    # 排序：high 在前，其中"正在服用"的更靠前
    out.sort(key=lambda h: (-_LEVEL_ORDER[h.level], not h.taking))
    return out


def _escalate(hits: list[SafetyHit], active_tags: set[str]) -> list[SafetyHit]:
    """交叉升级：药材规则的 tags 与当前生效 tag 相交 → 升为 high。

    这条是"档案驱动"的关键。例：阿胶规则 tags=("dampness",)，
    用户档案里写着「舌苔白腻、大便黏」→ dampness 生效 → 阿胶从 medium 升 high，
    话术也从"可以少量"变成"不适合现在吃"。
    """
    for h in hits:
        inter = set(h.tags) & active_tags
        if inter and h.level != "high":
            h.level = "high"
            h.escalated = True
            h.matched = h.matched
    return hits


# ---------------------------------------------------------------------------
# 五、汇总口径
# ---------------------------------------------------------------------------
def active_tags(*texts, profile: dict | None = None) -> set[str]:
    """汇总当前生效的风险 tag（文本 + 档案）。"""
    tags: set[str] = set()
    for t in texts:
        tags |= tag_from_text(t or "")
    if profile:
        tags |= tag_from_text("\n".join(str(v) for v in profile.values()))
    return tags


def needs_medication_stop(hits, tags: set[str] | None = None) -> bool:
    """是否需要触发"叫停自行服药"（而不是给点建议就完了）。

    判据：存在 level=high 且属"用户正在服用"的命中，
    或生效 tag 落在 STOP_MEDICATION_TAGS（慢病/特殊人群）里。
    """
    hs = [_as_dict(h) for h in (hits or [])]
    if any(h.get("level") == "high" and h.get("taking") for h in hs):
        return True
    if tags and (set(tags) & R.STOP_MEDICATION_TAGS):
        return True
    return False


# ---------------------------------------------------------------------------
# 六、渲染成注入 LLM 的强制区块
# ---------------------------------------------------------------------------
_BLOCK_HEAD = (
    "【⛔ 安全提示（由系统硬编码规则命中，非检索所得）】\n"
    "以下结论**不是建议、是要求**：必须在回答中体现，不允许省略、"
    "不允许用「资料库中没有记载」来回避，也不允许用「建议就医」一句带过。"
    "对「本轮命中」的每一个条目，都要给出明确的「能做 / 不能做 / 需先确认」判读；"
    "「档案提醒」只需要自然地带一句，不要展开成大段。\n"
    "**安全是分级、不是开关**：本次处置分五档——明确禁止 / 不建议（无适应症）/ "
    "需专业确认 / 对证但有条件 / 可执行。档位由「东西 × 这个人的具体情况」共同决定，"
    "**不许把所有命中都压成一句「不建议吃」**（那是该软不软），"
    "**也不许把「需确认」含糊成「咨询医师」了事**（那是该硬不硬）。\n"
    "⚠️「明确禁止」与「不建议」**必须分开说**：前者是用了有害（方向相反），"
    "后者是没有对应的虚证、没必要（并非有害）。把「不建议」说成"
    "「伤身体」是错的，反过来把「禁忌」含混成「不推荐」更危险。\n"
    "**规则只提供约束、不替代辨证**：每条的判读理由必须绑定用户**本轮**给出的"
    "具体条件（如「你苔白腻、便黏」）；若用户没有相反条件，就不要照搬通用话术。\n"
)

# 档案命中的渲染截断：一行式提醒的 verdict 取前 N 字
_PROFILE_VERDICT_MAX = 80


def _as_dict(h) -> dict:
    """把 SafetyHit 或普通 dict 统一成 dict。

    为什么要兼容：命中项要存进 LangGraph 的共享状态（必须可序列化），
    所以节点间传的是 dict；而扫描器直接返回的是 SafetyHit 对象。
    渲染函数两头都得吃。
    """
    if isinstance(h, dict):
        return h
    if hasattr(h, "to_dict"):
        return h.to_dict()
    return dict(getattr(h, "__dict__", {}) or {})


def _profile_line(h: dict) -> str:
    v = (h.get("verdict") or h.get("why") or "").strip()
    if len(v) > _PROFILE_VERDICT_MAX:
        v = v[:_PROFILE_VERDICT_MAX].rstrip("，、；") + "…"
    taking = "（在服）" if h.get("taking") else ""
    return f"· {h.get('name', '?')}{taking}：{v}"


def build_block(hits, tags: set[str] | None = None,
                with_stop: bool = True,
                profile_ok: bool = True,
                gaps=None) -> str:
    """把命中项渲染成一段放进 system prompt 的强制区块；无命中返回空串。

    三件事在这一层落地：

    ① **分级渲染**（架构层问题二/三）：每条判读文本由
       `safety/tiers.py::render` 按「档位 × 用户命中的条件」生成，
       不再用规则里写死的 verdict——同一味药在相反证型下必须得出不同结论，
       且理由绑定各自条件。
    ② **来源分流**：origin=message → 完整条目（本轮正问它）；origin=profile →
       一行式「档案提醒」（用户本轮没提，整卡糊脸会被读成"突兀、答非所问"，
       但完全不说又会漏掉慢病用药纪律）。
    ③ **缺条件不升级为警告**（架构层问题四）：活血脉命中、但"是否在服
       抗凝药"未知时，禁止把整套出血风险搬出来，改为明确要求追问。

    `profile_ok=False` 用于"疑似换人/换人设、尚未确认"的场合：此时档案
    带出的命中**不得作为本轮判断依据**，只保留为"待确认"提示。
    """
    from app.safety import tiers as T

    hits = [_as_dict(h) for h in (hits or [])]
    if not hits:
        return ""
    # 防御性补档：调用方若没显式 apply_tiers，这里补一次（保证渲染不落回旧 verdict）
    if not hits[0].get("tier"):
        T.apply_tiers(hits, tags, gaps)

    active = set(tags or set())
    msg = [h for h in hits if h.get("origin", "message") == "message"]
    prof = [h for h in hits if h.get("origin") == "profile"]
    lines: list[str] = [_BLOCK_HEAD]
    if active:
        labels = R.tag_labels(sorted(active))
        if labels:
            lines.append(f"用户当前情况标签：{'、'.join(labels)}")

    # ---- 分级总表：先给一张"谁在什么档位"的表，避免模型把五档混成一句"不建议吃" ----
    if msg:
        lines.append("")
        lines.append("【分级处置总表（档位由「这个东西 × 他的具体情况」共同决定，"
                     "同一条必须用同一档的语气）】")
        for h in msg:
            # 依据用 `tier_basis`（分级引擎算好的定档理由），不再由渲染层
            # 各写一套——档位只有一个判定点，牌子也只有一块（P0-1）
            basis = h.get("tier_basis") or (
                f"命中：{'、'.join(h.get('cond_labels') or [])}"
                if h.get("cond_labels") else "未命中任何禁忌条件")
            lines.append(f"  · {h.get('name', '?')}"
                         f" → **{h.get('tier_label', '')}**（{basis}）")
        # 档位纪律：每档的必须/禁止，一起交代（这是"唯一判定"的另一半）
        lines.append("")
        lines.append(T.tier_discipline_block(msg))

    for i, h in enumerate(msg, 1):
        head = (f"{i}. 【{h.get('name', '?')}】档位＝{h.get('tier_label', '')}"
                f"｜命中词「{h.get('matched', '')}」")
        if h.get("taking"):
            head += "（用户**正在服用/正在使用**）"
        if h.get("cond_labels"):
            head += f"（与他的情况交叉：{'、'.join(h['cond_labels'])}）"
        lines.append(head)
        if h.get("why"):
            lines.append(f"   药性/机制：{h['why']}")
        if h.get("not_for"):
            lines.append(f"   不宜人群：{h['not_for']}")
        # 判读文本由分级引擎按条件生成（不再是规则里写死的那段）
        lines.append(f"   必须给出的判读（{h.get('tier_label', '')}档）："
                     f"{T.render(h)}")
        if h.get("tier_tone"):
            lines.append(f"   语气要求：{h['tier_tone']}")

    if prof:
        lines.append("")
        if profile_ok:
            lines.append("【档案提醒（来自本次对话此前的记录，用户本轮未提及——"
                         "各用一句话带过即可）】")
        else:
            lines.append("【⚠️ 疑似换了咨询对象/人设，此前记录待确认——"
                         "**本轮不得作为判断依据**，只在确认时提一句】")
        lines.extend(_profile_line(h) for h in prof)

    if with_stop and needs_medication_stop(hits, tags):
        lines.append("")
        lines.append(STOP_MEDICATION_BLOCK)

    # 抗凝相关：有证据 → 三件套；只有活血命中、证据缺失 → 追问（不警告）
    ac_note = _anticoag_note(hits, tags)
    if ac_note:
        lines.append("")
        lines.append(ac_note)

    # 在服西药（降压/降糖/利尿…）：有证据 → 注入纪律三件套
    drug_note = _chronic_drug_note(hits)
    if drug_note:
        lines.append("")
        lines.append(drug_note)

    return "\n".join(lines)


def _anticoag_note(hits: list[dict], tags) -> str:
    """抗凝三件套的**触发闸门**：有证据才警告，没证据就追问。

    为什么必须收闸（架构层问题四，真实反馈）：
    原实现里只要命中项带了 anticoag tag 就绑三件套，而 `huoxue`（活血化瘀类）
    规则本身 tags 里就有 anticoag —— 于是**用户只是提了一句丹参、从没说自己在
    吃抗凝药**，回答里也会冒出「华法林、阿司匹林、氯吡格雷会增加出血风险」。
    这是"默认最危险"：把不确定的事当成最坏情况讲，会**吓退正确建议**。

    现在的判据：只有当"用户确实在用抗凝/抗血小板药"有证据时才给三件套；
    否则给出「分两种情况 + 必须问清」的指令，具体追问由 inquiry 负责。
    """
    active = set(tags or set())
    drug_evidence = any(
        h.get("kind") == "drug" and R.TAG_ANTICOAG in (h.get("tags") or [])
        for h in hits)
    herb_risk = [h for h in hits
                 if h.get("kind") == "herb"
                 and R.TAG_ANTICOAG in (h.get("tags") or [])]
    if drug_evidence or R.TAG_ANTICOAG in active:
        return ANTICOAG_TRIO_BLOCK
    if herb_risk:
        names = "、".join(h.get("name", "?") for h in herb_risk)
        return (ANTICOAG_UNKNOWN_BLOCK.format(names=names))
    return ""


def _chronic_drug_note(hits: list[dict]) -> str:
    """在服西药的**纪律三件套**：把规则里写死的 `must_say` 原样注入提示词。

    为什么必须单独一件（2026-09-16 真事故）：`build_block` 为落实
    「档位单源派生」而刻意不再使用规则里的 `verdict`，判读文本统一交给
    `tiers.render()` 生成 —— 可控，但**副作用**是：降压药这类西药的 verdict
    （不能自行停减 / 不可替代 / 监测与急诊阈值）**永远进不了提示词**，
    规划者自然写不出来；审查员再按规定以「红旗：未提示血压监测与就医阈值」
    否决 → 打回重写 ×2 → 必然降级，三专家产出根本到不了用户手上。

    抗凝早有 `_anticoag_note` 三件套兜住同类问题；这里把同一设计推广到
    **所有在服西药**（降压/降糖/利尿/甲功/他汀）。
    触发闸门与抗凝一致：**有证据（命中且 taking=True）才注入**，不预设事实。
    """
    rows = [h for h in (hits or [])
            if h.get("kind") == "drug" and h.get("taking") and h.get("verdict")]
    if not rows:
        return ""
    out = ["【正在服用的西药：以下纪律**必须原样（或近义）体现**，是要求不是建议】"]
    for h in rows:
        out.append(f"· 【{h.get('name', '?')}】{h['verdict']}")
    return "\n".join(out)


# 叫停自行服药：不是免责声明，是结论
STOP_MEDICATION_BLOCK = (
    "【必须包含的「暂停并咨询」话术】\n"
    "涉及毒性药材、处方药、或慢病/特殊人群自行用药时，你必须明确说出：\n"
    "「不建议自行服用 / 请先暂停，并咨询中医师或药师」——"
    "并且**结合用户的具体慢病解释原因**"
    "（例：你在吃降压药，附子制剂里的甘草可能让血压更难控制）。\n"
    "同时必须明确：用户**正在服用的西药（降压/降糖/抗凝等）"
    "不能因为讨论中药而自行停药或减量**，也不要只说「咨询专业医师」这种空话，"
    "要把「为什么对你尤其危险」讲清楚。\n"
)

# 缺条件时的正确动作：**分两种情况说清 + 追问**，而不是先搬最坏情况
# （架构层问题四：把"不知道用不用抗凝药"当成"在用抗凝药"来警告，
#   会吓退本来正确的建议——这是"默认最危险"的典型错法）
ANTICOAG_UNKNOWN_BLOCK = (
    "【{names}：已命中，但「是否在用抗凝/抗血小板药」未知 → 只许追问，不许先警告】\n"
    "用户提到了活血化瘀类中药/食材，但**没有任何证据**说明他本人正在服抗凝/"
    "抗血小板药（华法林、阿司匹林、氯吡格雷等）。\n"
    "**禁止**把「华法林/阿司匹林/氯吡格雷会增加出血风险」「黑便/瘀斑/呕血立即就医」"
    "这一整套铺开来讲——那会吓退本来正确的建议。\n"
    "正确做法（本轮必须照做）：\n"
    "  · 用一句话把两种情况分开说清：**在服**这类药 → 同用会增加出血风险、"
    "必须由医师评估；**没在服** → 常见量的丹参/三七作食养一般问题不大，但要对证；\n"
    "  · 然后**明确问一句**：「你现在有没有在吃阿司匹林、华法林、氯吡格雷这类药？"
    "（或有没有放过支架、做过双抗治疗）——这一条直接决定我给你的建议。」\n"
    "  · 抗凝三件套（西药不能自行停/告知医生药师/出血阈值）**留到确认在服之后**再讲。"
)

# P0-2：叫停相互作用时必须绑定的「西药保护三件套」
ANTICOAG_TRIO_BLOCK = (
    "【相互作用风险必须绑定「三件套」（凡涉及抗凝/抗血小板人群，缺一不可）】\n"
    "只要回答里出现了「某中药/食材 与 抗凝/抗血小板西药 有相互作用风险」：\n"
    "  1) 明确说出：**阿司匹林/华法林等西药不能自行停药、不能自行减量**——"
    "对冠心病/放过支架的人，自行停用抗血小板药可能诱发心梗、支架内血栓；\n"
    "  2) 明确说出：是否加用任何中药，请**先告知主治医生和药师**，由他们评估；\n"
    "  3) 明确说出**出血预警阈值**：黑便/柏油样便、皮下大片瘀斑、"
    "牙龈或鼻腔出血不止、呕血、尿液发红——出现任一情况立即就医。\n"
    "三件套必须完整出现在回答里；只讲中药风险、不叮嘱西药纪律，视为未完成。"
)


# ---------------------------------------------------------------------------
# 七、元信息过滤（P1-6：非结论型内容）
# ---------------------------------------------------------------------------
# 语料里存在"下回分解""详见下节"这类**写作性元信息**——它们是作者的行文
# 提示，不是知识结论。被检索进上下文后，模型会照引出来
# （实测："资料里说有一个简单有效的方法，但原文卖了个关子，写的是我们下回分解"），
# 既是噪声也损害可信度。这里在**取用侧**过滤，不重建索引
# （不重建的理由：重建要重嵌入 12000 块，代价大且无必要）。
_META_PATTERNS = (
    r"下回分解", r"详见下[节文篇章]", r"见下[节文篇章]", r"下[节文篇章]再[讲说叙]",
    r"后文再[讲说叙述]", r"这里(?:先)?不[再]?[展开赘述多讲]", r"暂(?:且|时)不表",
    r"且听下回", r"下文(?:将)?(?:详细)?(?:介?绍|说明|展开)", r"本节(?:到此)?结束",
    r"笔者将", r"我们(?:将在)?下一?[节章]",
)
_META_RE = re.compile("|".join(_META_PATTERNS))


def strip_meta_info(text: str) -> str:
    """剔除"下回分解"式元信息句；保留其余正文。幂等。

    以**句子**为单位删除（不是整块丢弃）：一个 500 字的块里往往只有
    一句话是元信息，整块丢掉会把真知识一起丢掉。
    """
    if not text:
        return text
    # 断句保留分隔符
    pieces = re.split(r"(?<=[。！？；\n])", text)
    kept = [p for p in pieces if not _META_RE.search(p)]
    return "".join(kept)


def has_meta_info(text: str) -> bool:
    return bool(_META_RE.search(text or ""))
