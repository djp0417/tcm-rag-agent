# -*- coding: utf-8 -*-
"""追问引擎：用户信息不全时，**主动开口问**，而不是硬答。

为什么单开一个模块
------------------
养生咨询是**信息不对称**的典型场景：用户上来只说一句"我最近老累"，
而"累"背后可能是气虚、湿困、阳虚、贫血、甲减……方向完全不同的处理。
此时最差的回答是：**凭一句主诉给一套方案**——看起来专业，实际上是在
用用户没提供的信息替用户做假设。

所以追问不是"礼貌性补问"，而是**产品的主干能力**：
信息不足 → 必须先问 → 问完再给方案。本模块负责三件事：

① **意图判定**（要不要追问的闸门）
   - "我最近老累，怎么调理" → 求建议：**该问**；
   - "阳虚体质有什么表现" → 问知识：**不该问**（问了反而像不回答）。
   闸门必须在代码里：早先的实现对所有问题都挂 gap_block，
   结果"阳虚体质有什么表现"这种纯知识题被一句"信息不足"卡住，答不完整。

② **缺口优先级**（问什么、先问什么）
   复用 `safety/scripts.py::GAP_ITEMS` 的话术库，但由本模块决定顺序与上限：
   **一次最多 3 问**（问 7 个问题的问卷式追问，用户会直接关掉页面）。
   排序原则：决定安全边界的（在服药物/慢病）优先于细节（舌象/二便）。

③ **程序化兜底**（模型没问就由代码问）
   提示词说"请给出 2~3 个聚焦追问"，实测模型经常忘（尤其它自认为答得挺好时）。
   所以生成之后做一次校验：回答里一个追问点都没覆盖 → 由代码把追问
   追加到回答末尾。这与 `checks.judgment_violations` / `_judgment_fallback`
   是同一套思路：**要求模型做的关键动作，都要有一条不依赖模型的兜底**。

设计约束
--------
- 追问文案由代码给（用户界面上的确定性内容，别让模型现编）；
- 已在本会话说过的项不再问（`intake.gaps` 会扫描本会话时间线）；
- 追问与"给通用安全建议"并存：不问到就不能给方剂/证型结论，
  但不依赖缺口的通用建议（作息、饮食有节）照给，避免答话一片空白。
"""
from __future__ import annotations

import re

from app.safety import scripts as SG

# 一次最多问几个问题。超过 3 个，用户要么乱答要么直接走。
MAX_QUESTIONS = 3

# ---------------------------------------------------------------------------
# 一、意图判定：这一轮到底该不该追问
# ---------------------------------------------------------------------------
# 「求建议」线索：出现这些词/句式，用户是在要一个针对自己的结论。
# 用正则而不是纯字面量，是因为中文的表达太散：「能吃阿胶吗」「能不能吃阿胶」
# 「可不可以吃」是一个意思，硬列字面量必然漏（实测漏过「能吃阿胶吗」）。
_ADVICE_RES = (
    re.compile(r"能(?:不能)?吃.{0,10}吗"),          # 能吃阿胶吗 / 能不能吃附子
    # 口语里"能不能喝/能不能用"和"能不能吃"一样常见——实测漏过
    # 「我听说丹参泡水好，我能喝吗」（那轮既没追问、也没走安全判读）
    re.compile(r"(?:我)?(?:能|可以|可不可以|能不能|该不该|要不要|适合)"
               r".{0,14}(?:吗|么|呢)"),
    re.compile(r"可不可以|能不能|该不该|要不要|有没有用|管用吗|有效吗|有用吗"),
    re.compile(r"适合.{0,8}(?:吗|么|不)"),
    re.compile(r"怎么(?:调理|改善|缓解|补|养|吃|办|调|治)"),
    re.compile(r"(?:吃|喝|用)点?(?:什么|啥)"),
    re.compile(r"帮我.{0,8}(?:看|分析|调|配|选|定)"),
    re.compile(r"(?:调理|调养|养生)(?:方案|计划|建议|思路)"),
)
_ADVICE_CUES = ("怎么办", "调理", "方案", "建议", "该吃", "给我开", "配点什么")
# 「问知识」线索：出现这些，用户是想弄懂一个概念，不是要方案。
_KNOWLEDGE_CUES = (
    "是什么", "什么是", "有哪些表现", "有什么表现", "什么意思", "为什么",
    "有什么作用", "有何作用", "有什么功效", "出处", "原文", "区别",
    "怎么理解", "原理",
)
_SELF_REF = re.compile(r"我|自己|本人|咱|家里老人")

# 「在说自己不舒服」的宽口径词表。不直接用 safety 的 SYMPTOM_TAGS 是因为
# 那套词表服务于**安全分级**（要求精确、宁可漏），而这里服务于**要不要开口问**
# （要求宽、宁可多问一句）——「我最近老是累」这种最典型的开场白，
# 在安全词表里命不中任何 tag，却显然该追问。
_COMPLAINT_RE = re.compile(
    r"累|疲|乏力|没劲|没精神|睡不好|失眠|睡不着|多梦|头晕|头疼|头痛|"
    r"疼|痛|胀|闷|心慌|出汗|盗汗|怕冷|怕热|手脚|口苦|口干|上火|"
    r"便秘|拉肚子|腹泻|便溏|大便|小便|不舒服|难受|胖|瘦|肿|"
    r"长痘|长斑|过敏|没胃口|口气|湿气|虚|上火")

# 求建议时必须有、且缺了就不能给结论的项（顺序即追问优先级）
REQUIRE_ADVICE: tuple[str, ...] = (
    "medication",      # 在服药物：直接决定安全边界，先问
    "chronic",         # 慢病
    "age_sex",         # 年龄性别
    "cold_heat",       # 寒热：辨证第一分水岭
    "stool_urine",     # 二便
    "tongue",          # 舌象
    "chief_time",      # 病程
)
# 问知识（含轻量养生问答）不追问；只有明确了"这是关于我自己 / 我要方案"才问。
REQUIRE_KNOWLEDGE: tuple[str, ...] = ()


def _has_advice_cue(t: str) -> bool:
    return (any(c in t for c in _ADVICE_CUES)
            or any(r.search(t) for r in _ADVICE_RES))


def intent_of(text: str) -> str:
    """这一轮的意图：'advice'（要针对自己的建议） / 'knowledge'（问知识）。

    判定顺序（顺序即优先级，别随意调）：
      ① 有求建议句式 **且**（提到自己 / 命中症状或慢病）→ advice；
         "能…吗"这类句式单看是歧义的（"红豆薏米茶能吃吗"只是问食材常识），
         必须叠加"这是在说他自己"的证据才追问；
      ② 有问知识句式 → knowledge；
      ③ 提到自己 + 在说自己不舒服 → advice（"我最近老是累"这种开场白）；
      ④ 其余 → knowledge（不追问，正常回答）。
    """
    t = (text or "").strip()
    if not t:
        return "knowledge"
    from app.safety import tag_from_text
    personal = bool(_SELF_REF.search(t)) or bool(tag_from_text(t))
    if _has_advice_cue(t) and personal:
        return "advice"
    if any(c in t for c in _KNOWLEDGE_CUES):
        return "knowledge"
    if _SELF_REF.search(t) and _COMPLAINT_RE.search(t):
        return "advice"
    return "knowledge"


def require_slots(intent: str) -> tuple[str, ...]:
    return REQUIRE_ADVICE if intent == "advice" else REQUIRE_KNOWLEDGE


# ---------------------------------------------------------------------------
# 二、场景化阻塞项：某些条目一旦命中，它的档位**完全取决于一个未知条件**
# ---------------------------------------------------------------------------
# 这就是"缺口 → 追问"而不是"缺口 → 保守默认"的落点（架构层问题四）：
# 用户提到丹参/三七/红花，但从没说自己吃不吃抗凝药。
# 此时：
#   · 错的A（早期）：只说"资料库没记载"——该硬不硬；
#   · 错的B（后期）：把华法林/阿司匹林/氯吡格雷整套出血风险铺开——默认最危险，
#     用户明明没吃抗凝药，却被吓退了本来正确的建议；
#   · 对的：把"你在不在吃"**问出来**，两种情况分开说一句，结论挂在回答后面。
_ANTICOAG_ANSWERED_RE = re.compile(
    r"阿司匹林|华法林|氯吡格雷|波立维|替格瑞洛|利伐沙班|达比加群|双抗|"
    r"抗凝|抗血小板|支架|搭桥|心梗|脑梗")


def scenario_gaps(hits, tags=None) -> list[str]:
    """命中了"档位取决于未知条件"的条目时，返回必须问清的缺项（阻塞项）。"""
    from app.safety import rules as R
    if R.TAG_ANTICOAG in set(tags or set()):
        return []                       # 已经有证据在服抗凝药 → 不需要再问
    herbs = [h for h in (hits or [])
             if (h.get("kind") if isinstance(h, dict) else getattr(h, "kind", ""))
             == "herb"
             and R.TAG_ANTICOAG in (
                 h.get("tags") if isinstance(h, dict) else getattr(h, "tags", []))]
    return ["anticoag_use"] if herbs else []


def followups(text: str, conv_id: int | None = None,
              profile: dict | None = None, intent: str | None = None,
              gaps: list[str] | None = None,
              hits=None, tags=None) -> list[dict]:
    """算出本轮该问的问题（[{key,label,ask,subject,signals}]，已排序、已截断）。

    intent == 'knowledge' 时返回空列表——纯知识问题**不追问**。
    `hits`/`tags`：本轮的安全命中与生效标签，用于插入**场景化**阻塞项
    （见 `scenario_gaps`）。

    2026-09-16 第四轮（P0）三项改造：
    ① **已知比对**：gaps 已改为主体感知版（intake.gaps）——"我爸爸 68 岁
       在吃药"不再被追问"你的年龄性别/有没有吃药"；
    ② **主体跟随**：咨询对象是家人时，问句措辞切换到这一位
       （"（关于您父亲）他的年龄…"），不拿用户本人顶替；
    ③ **主诉关联优先**：`chief_gaps` 按 CHIEF_GAP_RULES 挂载与主诉强相关
       的追问（备孕→周期排卵/药名；认知下降→药名/时间线/卒中影像），
       排在通用三件套前面；"在吃药但没给药名"时把"有没有吃药"
       升级为"具体药名"，而不是重复问。
    """
    intent = intent or intent_of(text)
    if intent != "advice":
        return []
    from app import intake as IL
    from app.safety import scripts as SG
    rel, sblob = IL.subject_scope(text, conv_id)
    if gaps is None:
        gaps = IL.gaps(profile, text, require=require_slots(intent),
                       conv_id=conv_id)
    # 主体感知复核：gaps 已按主体算过，这里再对每个通用缺口做一次
    # "已知"复核（宽口径）——已知的直接滤掉，不进追问清单。
    prof = profile if profile is not None else IL.storage.get_profile(conv_id)
    gaps = [g for g in (gaps or []) if not IL.gap_known(g, sblob, prof, rel)]
    # 「在吃药但没给药名」→ 升级为问药名（而不是重复问"有没有吃药"）
    if ("medication" in gaps
            and IL.gap_known("medication", sblob, prof, rel)
            and not IL.gap_known("drug_names", sblob, prof, rel)):
        gaps = [g for g in gaps if g != "medication"]
        if "drug_names" not in gaps:
            gaps.insert(0, "drug_names")
    # 场景化阻塞项插到最前：它的档位影响最大（决定"能不能碰活血类"）
    scene = [g for g in scenario_gaps(hits, tags) if g not in gaps]
    if scene:
        # 用户已经在会话里回答过（明确提过抗凝药/支架）→ 不再问
        blob = IL._conv_blob(text or "", conv_id) + "\n" + "\n".join(
            str(v) for v in (prof or {}).values())
        if not _ANTICOAG_ANSWERED_RE.search(blob):
            gaps = scene + list(gaps)
    # 主诉关联追问（备孕/认知下降…）：插在通用缺口之前——
    # 反馈实测通用三件套对主诉没有针对性，"最影响下一步判断"的才是该先问的
    chief = chief_gaps(text, sblob, prof, rel, already=set(gaps))
    gaps = chief + [g for g in gaps if g not in chief]
    # 慢病在册但控制值未知 → 值得问一句（不挤掉更关键的：超上限由截断兜底）
    if (IL.gap_known("chronic", sblob, prof, rel)
            and not IL.gap_known("control_values", sblob, prof, rel)
            and "control_values" not in gaps):
        gaps.append("control_values")
    # ---- 围产期状态过滤（2026-09-16 第五轮，版本退化修复）----
    # 反馈实测：孕妇被问"有没有牙龈出血、黑便"（抗凝模板硬贴），
    # 备孕场景反被问与已妊娠无关的项。**追问必须按主诉 + 状态双重过滤**。
    try:
        from app import constraints as CN
        _skeys = {str(s.get("key")) for s in CN.detect(text, prof, conv_id)}
    except Exception:
        _skeys = set()
    if _skeys & {"pregnancy", "lactation"}:
        # 已妊娠 / 哺乳：备孕评估（周期排卵）、抗凝与出血模板都不适用
        _drop = {"anticoag_use", "bleeding", "cardiac_history", "menopause",
                 "fertility_cycle"}
        gaps = [g for g in gaps if g not in _drop]
    elif "preconception" in _skeys:
        # 备孕：禁忌面由档位直接给出（当归类已判「明确禁止」），
        # 不再追问抗凝 / 出血 —— 那不是这个场景要问的。
        _drop = {"anticoag_use", "bleeding", "cardiac_history", "menopause"}
        gaps = [g for g in gaps if g not in _drop]
    if not gaps:
        return []
    out: list[dict] = []
    for k in gaps:
        it = SG.GAP_ITEMS.get(k)
        if not it:
            continue
        out.append({"key": k, "label": it["label"],
                    "ask": _subjectize(it["ask"], rel),
                    "subject": rel,
                    "signals": list(it.get("signals") or ())})
        if len(out) >= MAX_QUESTIONS:
            break
    return out


# require 槽位名与 `gap_known` 的证据 key 一一对应（age_sex 在 gap_known 内拆开验）


def chief_gaps(text: str, sblob: str, prof: dict | None, rel: str,
               already: set[str] | None = None) -> list[str]:
    """主诉关联追问：按 CHIEF_GAP_RULES 挂载（条件在数据里，求值器只有一个）。

    与 `scenario_gaps`（安全阻塞）不同，这一组服务的是**针对性**：
    反馈实测"68 岁认知下降"该问时间线/卒中史/照护人，而不是通用三件套；
    "备孕三年"该问周期排卵与在服药名。
    """
    from app.safety import scripts as SG
    from app import intake as IL
    out: list[str] = []
    for rule in SG.CHIEF_GAP_RULES:
        if not any(re.search(SG.CHIEF_SIGNALS[s], sblob)
                   for s in (rule.get("any") or ())):
            continue
        for g in rule.get("gaps") or ():
            if g in out or g in (already or set()):
                continue
            if g == "drug_names":
                # 「具体药名」只有在**已知在服药**时才成立——
                # 没有在服证据的人被问"你吃的哪几种药"是预设事实（实测翻车）
                if not IL.gap_known("medication", sblob, prof, rel):
                    continue
            if IL.gap_known(g, sblob, prof, rel):
                continue                        # 已经说过的不再问
            out.append(g)
    return out


def _subjectize(ask: str, rel: str) -> str:
    """把问句的**主语**从"你"切到咨询对象（"我爸爸"→ 您父亲/他）。

    2026-09-16 第四轮 P0 失效②：问句永远是"你的年龄和性别"——主体是
    父亲时，用户读到的第一反应就是"这问题不该问我"。
    """
    if not rel:
        return ask
    a = (ask or "").replace("你的", "他的").replace("你", "他")
    return f"（关于您{rel}）{a}"


def followup_block(items: list[dict]) -> str:
    """把追问清单渲染成提示词区块（与 scripts.gap_block 配合使用）。"""
    if not items:
        return ""
    lines = [
        "【本轮必须主动追问（这是产品主干能力，不是客套）】",
        "用户给的信息还不足以给出一套针对他的方案。要求：",
        "1. **回答的最后，必须原样提出下面这几个问题**（可以用列表，不要改写意思）；",
        "2. 提问前先用一句话说清「为什么要问」（例如：在吃什么药决定你"
        "能不能碰某些药材，这是安全底线）；",
        "3. **不要**在追问之前下证型结论、不要开方、不要给具体药材与剂量；",
        "4. 可以先给**不依赖这些信息的通用安全建议**（作息、饮食有节、"
        "避免久坐、情绪调畅），一句「这些是通用的」自然带过即可；"
        "**不要**写「针对你个人的判断要等你补充信息后再做」这类临床式声明，"
        "**不要**在正文里清点用户没说过什么（体质、舌象、大便、用药…）；",
        "5. 一次问 2~3 个就够，不要列一长串问卷；语气像聊天里顺带问一句，"
        "不像问诊表。",
    ]
    subj = next((it.get("subject") for it in items if it.get("subject")), "")
    if subj:
        # 第四轮 P0 失效②：主体是家人时，问句全部切到这一位——
        # 不再出现"你的年龄和性别"这种问错对象的话
        lines.append(f"6. **本轮咨询对象是「{subj}」**：所有追问都问这一位的"
                     f"情况；用户本人已交代的（年龄、慢病、用药）属于本人，"
                     f"**不得**拿来顶替这一位的信息，也不得再向用户本人提问。")
    lines += ["", "本轮要问的问题（逐条写到回答里）："]
    for it in items:
        lines.append(f"· 【{it['label']}】{it['ask']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 三、程序化兜底：模型没问，就由代码问
# ---------------------------------------------------------------------------
def _covered(answer: str, item: dict) -> bool:
    """回答里是否已经覆盖了这条追问（按关键词判，宽松即可——宁可判"已问"）。"""
    sigs = item.get("signals") or ()
    return any(s in answer for s in sigs)


def ensure_questions(answer: str, items: list[dict]) -> tuple[str, list[str]]:
    """校验回答是否真的追问了；没有则由代码追加。

    Returns: (最终回答, 由代码补上的问题文案列表)。
    为什么要有这一层：提示词约束"必须追问"的可靠性大约只有七八成，
    而养生场景里"没问就开方"是产品级缺陷。所以关键动作一律留一条
    不依赖模型的兜底路径（同 `checks.judgment_violations` 的思路）。
    """
    answer = answer or ""
    if not items:
        return answer, []
    missing = [it for it in items if not _covered(answer, it)]
    # 已经问了至少两条 → 视为模型完成了追问，不做追加（避免啰嗦）
    if len(missing) <= len(items) - 2:
        return answer, []
    lines = ["", "---", "",
             "**为了给你更准的建议，我想先确认几件事**"
             "（这几项直接决定方向和安全边界）：", ""]
    for i, it in enumerate(missing or items, 1):
        lines.append(f"{i}. {it['ask']}")
    lines.append("")
    lines.append("先说一句通用的：在弄清楚上面几点之前，作息规律、"
                 "三餐定时、少熬夜、别久坐，这几条对任何体质都成立，"
                 "可以先照着做。")
    return (answer.rstrip() + "\n" + "\n".join(lines)), [
        it["ask"] for it in (missing or items)]


# ---------------------------------------------------------------------------
# 四、给前端的可点选问题（点一下就填进输入框）
# ---------------------------------------------------------------------------
def to_event(items: list[dict]) -> list[dict]:
    """SSE 下发用的问题结构（前端渲染成按钮）。ask 已带主体化措辞。"""
    return [{"key": it["key"], "label": it["label"], "ask": it["ask"],
             "subject": it.get("subject", "")}
            for it in items]
