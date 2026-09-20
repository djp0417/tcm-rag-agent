# -*- coding: utf-8 -*-
"""接诊层：结构化档案 + 一致性校验 + 跨轮串联 + 信息缺口追问。

它解决的三件真实问题（都来自用户实测）
--------------------------------------
① **档案是一团自然语言**：原来的 profile 只有「体质判定」「体质转化分明细」
   等零散键值，慢病、西药、年龄这些**决定安全边界**的信息反而没进结构化档案，
   于是"上次说过有高血压"在下一轮就丢了。
   → 本模块把它们抽成结构化字段（年龄/性别/慢病/西药/体质/主诉时间线）。

② **档案之间会自相矛盾**：用户档案里写「阴虚质（兼夹痰湿质）」，
   但他自己说「冬天特别怕冷、手脚冰凉」——**阴虚与畏寒方向相反**，
   继续按阴虚调（滋阴润燥）就是往反方向使劲。
   → 本模块做读取时的一致性校验，冲突时提示复核 / 建议复测 27 题。

③ **跨轮信息没有被串起来**：第 1 轮说"睡够了也没精神"，第 2 轮说
   "舌胖齿痕、苔白腻、身重"，第 3 轮才补上"45 岁、女、胖了十来斤、便黏、高血压"。
   单看每一轮都只是症状，**合起来才是"脾阳不足 + 寒湿困脾"这个完整病机**，
   而系统此前每轮都在从头解释"什么是湿"，没有一句把三轮串起来的结论。
   → 本模块产出「跨轮整合材料」并强制回答里出现整合结论。

为什么抽取用规则而不是 LLM
--------------------------
年龄、性别、慢病、西药这几项都是**格式高度固定**的信息（"45 岁""女""
氨氯地平""高血压"），规则抽取零成本、零延迟、可复现。
LLM 只负责最后那句"整合结论"的自然语言表达——那是它真正擅长的事。
"""
from __future__ import annotations

import json
import re

from app import storage
from app.safety import rules as R
from app.safety import scan as SC
from app.safety import scripts as SG
from app.safety.scripts import SCREEN_SIGNALS

# 排查规则表的信号正则（编译一次；表在 app/safety/scripts.py::SCREEN_RULES）
_SCREEN_RES: dict = {k: re.compile(v) for k, v in SCREEN_SIGNALS.items()}

# 结构化字段（写进 profile 的键名固定，便于前端记忆面板展示与用户修改）
F_AGE = "年龄"
F_SEX = "性别"
F_CHRONIC = "慢病"
F_DRUGS = "西药"
F_CONSTITUTION = "体质判定"
# 注意：档案键会出现在 REST 路径参数里（DELETE /api/profile/{key}），
# uvicorn 会先把 %2F 解码回 / 再做路由匹配，键里带 / 永远 404——所以不用斜杠。
F_SUPPLEMENTS = "在服中药与食疗"
F_TIMELINE = "主诉时间线"

PROFILE_FIELDS = (F_AGE, F_SEX, F_CHRONIC, F_DRUGS, F_SUPPLEMENTS,
                  F_CONSTITUTION, F_TIMELINE)

TIMELINE_MAX = 6            # 时间线最多保留几条（超出的丢弃最旧的）
_TIMELINE_MARKS = "①②③④⑤⑥⑦⑧⑨⑩"


# ---------------------------------------------------------------------------
# 一、从文本里抽取结构化信息（纯规则）
# ---------------------------------------------------------------------------
_SEX_RE = [
    (re.compile(r"(?:我是|我|本人)?\s*(?:女|女性|女的|女生|女，|女儿身)"), "女"),
    (re.compile(r"(?:我是|我|本人)?\s*(?:男|男性|男的|男生|男，)"), "男"),
]
# 只认"明显指向本人"的写法，避免"我女儿""我老婆"被误判成本人性别
_SEX_GUARD = ("我女儿", "我儿子", "我老婆", "我老公", "我母亲", "我父亲",
              "我妈妈", "我爸爸", "我妻子", "我丈夫")

_AGE_RES = (
    re.compile(r"(\d{1,3})\s*(?:周?岁)"),
    re.compile(r"(?:今年|年龄|我)\s*(\d{2})\s*(?:了|，|,|。|$)"),
    re.compile(r"(\d{2})\s*岁"),
)


def extract_age(text: str) -> str:
    for rex in _AGE_RES:
        m = rex.search(text or "")
        if m:
            n = int(m.group(1))
            if 1 <= n <= 120:
                return str(n)
    return ""


def extract_sex(text: str) -> str:
    t = text or ""
    if any(g in t for g in _SEX_GUARD) and not re.search(r"我(?:是|本人)?(?:女|男)", t):
        return ""
    for rex, val in _SEX_RE:
        if rex.search(t):
            return val
    return ""


def _affirmed(text: str, kw: str) -> bool:
    """kw 能否作为"确有其事"采信：既要**未被否定**，也不能落在"我已经停了"的分句里。

    两层守卫缺一不可（2026-09-16 真机实测，两个方向都错过）：
      · 只有否定守卫时，「降压药已经停了」仍会把降压药写进"在服西药"；
      · 只管停用词时，「没有高血压糖尿病」照样被写进慢病。
    档案是只增不减的 CSV 合并，**写错一次就错到删会话为止**——
    所以抽取这一步的守卫必须比"回答里提不提"更硬。
    """
    t = text or ""
    if not SC.hit_affirmative(t, kw):
        return False
    for cl in re.split(r"[，。；;！？\n]", t):
        if kw not in cl:
            continue
        if _stop_in_clause(cl, kw[:2]):
            continue                      # 这一分句说的是"停掉它"，不算在服/在病
        return True
    return False


def extract_conditions(text: str) -> list[str]:
    """已告知的慢病（用 safety 的 CONDITION_TAGS 词表，口径统一）。

    ⚠️ **必须走否定守卫**（2026-09-16 真机挖出的档案级污染）：
    实测用户说「没有高血压糖尿病」，`k in t` 的朴素匹配把它原样写进档案
    ——「慢病 ＝ 孕产/备孕、高血压、糖尿病」。用户明确否认的病成了他的长期
    事实，此后每一轮安全判读都按"你有高血压"来算（卡片上直接印
    「命中：高血压」），而档案是 CSV **只增不减**的合并，错到删会话为止。
    回答错只错一轮，档案错错一整个会话——所以这里的守卫比回答侧更硬。
    """
    t = text or ""
    out: list[str] = []
    for tag, kws in R.CONDITION_TAGS:
        if tag in (R.TAG_CHILD, R.TAG_ELDERLY):
            continue                              # 不属于"慢病"
        if any(_affirmed(t, k) for k in kws):
            lb = R.TAG_LABEL.get(tag, tag)
            if lb not in out:
                out.append(lb)
    return out


def extract_drugs(text: str) -> list[str]:
    """正在服用的西药（命中类别 + 具体药名）。

    同样要过两层守卫：「我没吃降压药」是**否认**，「降压药已经停了」是**停用**，
    两者都不该写成"在服"。
    """
    t = text or ""
    out: list[str] = []
    for rule in R.DRUG_CLASSES:
        for a in sorted(rule.aliases, key=len, reverse=True):
            if _affirmed(t, a):
                item = f"{a}（{rule.cls}）"
                if item not in out:
                    out.append(item)
                break
    return out


def extract_facts(text: str) -> dict:
    """一轮用户输入 → 结构化事实（只含抽到的项）。"""
    facts: dict[str, str] = {}
    own = _strip_kin_clauses(text)
    if (v := extract_age(own)):
        facts[F_AGE] = v
    if (v := extract_sex(own)):
        facts[F_SEX] = v
    if (v := extract_conditions(own)):
        facts[F_CHRONIC] = "、".join(v)
    if (v := extract_drugs(own)):
        facts[F_DRUGS] = "、".join(v)
    return facts


# ---------------------------------------------------------------------------
# 亲属分句守卫（2026-09-15 上线清单②：记忆隔离的档案侧根因）
# ---------------------------------------------------------------------------
# 事故："我女儿25岁，口苦苔黄腻" → 「25 岁」被写进**本人**档案的年龄，
#       "我父亲有糖尿病" → 糖尿病进本人慢病。下一轮系统把**不同的人**
#       当成同一个人的档案变更，追问"您之前有糖尿病现在还吃吗"。
# 治法：命中亲属词的**分句**整句剥离，只留本人语境参与档案抽取。
# 按分句而不是整段剥离：混合句「我女儿25岁，我自己45岁高血压」
# 必须保住后半句，整段剥离会把本人的信息也丢掉。
_KIN_WORDS = ("女儿", "儿子", "母亲", "父亲", "妈妈", "爸爸", "我妈", "我爸",
              "老公", "老婆", "妻子", "丈夫", "我姐", "我妹", "我哥", "我弟",
              "爷爷", "奶奶", "外公", "外婆", "岳母", "岳父")


def _strip_kin_clauses(text: str) -> str:
    """剥掉"替家人说"的分句，返回只含本人语境的文本（供档案抽取）。"""
    t = text or ""
    if not any(k in t for k in _KIN_WORDS):
        return t
    kept = [cl.strip() for cl in re.split(r"[，。；;！？\n]", t)
            if cl.strip() and not any(k in cl for k in _KIN_WORDS)]
    return "，".join(kept) if kept else ""


# ---------------------------------------------------------------------------
# 咨询对象主体化（2026-09-16 第四轮 P0：追问不做已知比对、问错对象）
# ---------------------------------------------------------------------------
# 反馈实测："我爸爸 68 岁，有高血压和糖尿病，一直在吃药" → 系统追问
# "你的年龄和性别""有没有在吃西药"。三个失效叠在一起：
#   ① 档案抽取正确地把亲属分句剥离了（那是防档案污染的守卫，不能拆），
#     于是 gaps 看到空档案 → 把已经交代过的事当未知再问一遍；
#   ② 追问话术全是"你的XX"——主体是父亲，问的却是用户本人；
#   ③ 该问的（具体药名、控制值、时间线）没问。
# 治法：档案抽取的守卫**原样保留**；在这之上加一层「主体作用域」——
# 同一轮文本 + 时间线里属于**同一主体**的条目，作为"已知事实"参与缺口判断。
# 已知比对和档案写入是两件事：写档案要防污染（严），判已知要认事实（宽）。
_KIN_ROLES: dict[str, tuple[str, ...]] = {
    "父亲": ("爸爸", "我爸", "父亲", "老爹", "爹"),
    "母亲": ("妈妈", "我妈", "母亲", "娘"),
    "儿子": ("儿子", "我儿"),
    "女儿": ("女儿",),
    "丈夫": ("老公", "丈夫", "先生"),
    "妻子": ("老婆", "妻子", "太太"),
    "婆婆": ("婆婆",), "公公": ("公公",),
    "爷爷": ("爷爷",), "奶奶": ("奶奶",),
    "外公": ("外公",), "外婆": ("外婆",),
    "岳父": ("岳父",), "岳母": ("岳母",),
    "姥姥": ("姥姥",), "姥爷": ("姥爷",),
    "姐姐": ("姐姐", "我姐"), "妹妹": ("妹妹", "我妹"),
    "哥哥": ("哥哥", "我哥"), "弟弟": ("弟弟", "我弟"),
    "朋友": ("朋友",), "同事": ("同事",), "邻居": ("邻居",),
    "同学": ("同学",), "领导": ("上司", "领导"),
}
# 词 → 角色名（长词在前，防止"我妈"先于"我妈妈"…这类前缀误配）
_KIN_ROLE_OF: dict[str, str] = {
    w: role for role, ws in _KIN_ROLES.items() for w in ws}
_KIN_ROLE_WORDS = sorted(_KIN_ROLE_OF, key=len, reverse=True)
# 亲属词自带性别（"爸爸"必然是男的）——主体是亲属时，性别算已知
_KIN_GENDER: dict[str, str] = dict((
    ("父亲", "男"), ("丈夫", "男"), ("儿子", "男"), ("公公", "男"),
    ("爷爷", "男"), ("外公", "男"), ("姥爷", "男"), ("岳父", "男"),
    ("哥哥", "男"), ("弟弟", "男"),
    ("母亲", "女"), ("妻子", "女"), ("女儿", "女"), ("婆婆", "女"),
    ("奶奶", "女"), ("外婆", "女"), ("姥姥", "女"), ("岳母", "女"),
    ("姐姐", "女"), ("妹妹", "女"),
))


# 「XX 推荐 / 介绍」里的关系词是**信源**不是**咨询对象**——
# 「朋友推荐我吃阿胶」的咨询对象是用户本人（这正是第三轮挖出的
# 「推荐 ≠ 在服」同一类陷阱在主体识别上的变体，不设守卫会把
# 本人档案整体排除，已知信息全部"失忆"）
_RECO_CUE_RE = re.compile(r"^(?:推荐|介绍|安利|建议|劝)")


def _false_subject(t: str, start: int, word: str) -> bool:
    """关系词出现的位置是不是"信源"用法（后面紧跟推荐/介绍类动词）。"""
    return bool(_RECO_CUE_RE.match(t[start + len(word):]))


def subject_label(text: str) -> str:
    """本轮的**咨询对象角色**（"" = 用户本人）。

    优先用隔离层的 `_RELATION_RE`（它的词表判过"与人相关"且刻意排除
    「他/她/还有一个」这类模糊指代——那些正是要反问确认的情形）；
    关系词不在角色表里（朋友/同事/领导…）就直接用词本身当称呼。
    「朋友推荐我吃阿胶」这类信源用法不算换主体（见 _false_subject）。
    """
    t = text or ""
    rel = relation_of(t)
    if rel:
        m = _RELATION_RE.search(t)
        if not (m and _false_subject(t, m.start(), m.group(0))):
            role = _KIN_ROLE_OF.get(rel)
            return role or rel.lstrip("我") or rel
    # 兜底：隔离层没抓到但明确的亲属词出现了（如"妈妈说她血压高"开头没"我"）
    for w in _KIN_ROLE_WORDS:
        i = t.find(w)
        while i >= 0:
            if not _false_subject(t, i, w):
                return _KIN_ROLE_OF[w]
            i = t.find(w, i + 1)
    return ""


def _entry_roles(entry: str) -> set[str]:
    """一条时间线原话涉及哪些主体角色。"""
    return {_KIN_ROLE_OF[w] for w in _KIN_ROLE_WORDS if w in entry}


def subject_scope(text: str, conv_id: int | None = None) -> tuple[str, str]:
    """返回 (主体角色, 该主体已说过的全部原话)。

    主体 = 本人 → 时间线里**剥掉**提到亲属的条目（那是别人的事）；
    主体 = 某位家人 → 只保留涉及**同一角色**的条目（防止上一人的事实
    冒充这一人的已知信息——隔离验的追问版：不仅档案要隔离，"已知"也要隔离）。
    """
    rel = subject_label(text or "")
    entries = timeline_entries(conv_id)
    if not rel:
        kept = [e for e in entries if not any(k in e for k in _KIN_WORDS)]
    else:
        kept = [e for e in entries if rel in _entry_roles(e)]
    return rel, (text or "") + "\n" + "；".join(kept)


# ---------------------------------------------------------------------------
# 缺口的"已知"判定（第四轮 P0 核心：问之前先比对，问过/说过的不再问）
# ---------------------------------------------------------------------------
# 每个缺口一条证据正则：命中 = 这个信息已经有了。本人场景再叠加档案字段。
# 注意与档案抽取的分工：抽取守卫要**严**（写错一次错到删会话），
# 已知判定要**宽**（漏判的代价是重复提问，误判的代价是多问一句——后者轻得多）。
_GAP_EVID_RES: dict[str, str] = {
    "tongue": r"舌(?:头|体|苔|尖|质)|齿痕|苔[白黄薄厚腻]|白腻|黄腻",
    "stool_urine": r"大便|小便|便溏|便秘|尿液|夜尿|黏马桶",
    "cold_heat": r"怕冷|畏寒|怕热|手脚(?:冰)?凉|手足心热|喜热|喜凉|口干|"
                 r"不渴|喝(?:热|凉)水",
    # 「备孕三年」「高血压十年」都算时间线已知——原表漏了"数字+年"
    "chief_time": r"最近|这(?:两|几)年|半年|几个月|一直|从.{0,6}开始|"
                  r"\d{1,2}\s*年|多年|好几年",
    "control_values": r"\d{2,3}\s*/\s*\d{2,3}|(?:血压|血糖)[^。；;\n]{0,10}\d",
    "cog_timeline": r"什么时候开始|从.{0,8}(?:开始|起)|加重|进展|迷路|走失|"
                    r"淡漠|\d{1,2}\s*年|几个月|半年",
    "stroke_imaging": r"脑梗|中风|脑出血|卒中|腔梗|CT|磁共振|核磁|影像",
    "living_care": r"独居|一个人(?:住|生活)|照护|陪(?:同|护|伴)|照顾|子女",
    # 严格口径：主诉里**提到月经**不等于"周期情况已知"——
    # "月经总是往后推"正是需要追问周期细节的信号（2026-09-16 第四轮实测）
    "fertility_cycle": r"排卵|监测排卵|激素六项|AMH|周期.{0,3}\d{1,2}|"
                       r"月经.{0,6}\d{1,2}\s*天|周期规律|月经规律",
    "bleeding": r"牙龈出血|鼻腔出血|流鼻血|瘀斑|瘀点|黑便|柏油|呕血|出血",
    "cardiac_history": r"支架|搭桥|冠心病|心梗|双抗",
    "menopause": r"绝经|月经紊乱|停经|更年|经期|月经",
}
_GAP_EVID_RE = {k: re.compile(v) for k, v in _GAP_EVID_RES.items()}
# 「在吃药」类表述：说了在服药但没给药名 —— 这算"medication 已知"，
# 但 drug_names（具体药名）仍未知，追问要升级成"哪几种药"，而不是重复问"有没有吃药"
_TAKING_MED_RE = re.compile(r"在吃药|在服药|一直吃药|长期吃药|吃着药|在用药|"
                            r"一直都在吃药|天天吃药|每天吃药")
_SEX_EXPLICIT_RE = re.compile(r"女性?|男性?|女的|男的")
_FEMALE_HINT_RE = re.compile(r"月经|经期|绝经|更年|备孕|怀孕|妊娠|哺乳|"
                             r"子宫|卵巢|她")
_MALE_HINT_RE = re.compile(r"(?<!她)他|前列腺|精子|遗精")


def gap_known(key: str, blob: str = "", prof: dict | None = None,
              rel: str = "") -> bool:
    """某个缺口是否已经有信息了（**主体感知**）。

    rel 非空 = 在替某位家人问：此时**本人档案不参与**（那是另一个人的
    年龄/慢病——拿它顶替正是"问错对象 + 串档"的根源），亲属词自带的
    性别（爸爸=男）可以直接算已知。
    """
    prof = prof or {}
    b = blob or ""
    if key == "age":
        return bool(prof.get(F_AGE)) or bool(extract_age(b))
    if key == "sex":
        if prof.get(F_SEX):
            return True
        if rel and _KIN_GENDER.get(rel):
            return True                     # "爸爸"→男，词面就是证据
        return bool(_SEX_EXPLICIT_RE.search(b)
                    or _FEMALE_HINT_RE.search(b) or _MALE_HINT_RE.search(b))
    if key == "age_sex":
        return gap_known("age", b, prof, rel) and gap_known("sex", b, prof, rel)
    if key == "medication":
        if prof.get(F_DRUGS):
            return True
        # 按**分句**判「在服药」：反例「我没吃药，但我爸爸在吃药」——
        # 整段看会误判，分句后两句各归各的（同 note_stops 的分句判据）。
        for cl in re.split(r"[，。；;！？\n]", b):
            if _TAKING_MED_RE.search(cl) and not re.search(r"没|不|别|停", cl):
                return True
        return False
    if key == "chronic":
        if prof.get(F_CHRONIC):
            return True
        for _tag, kws in R.CONDITION_TAGS:
            if _tag in (R.TAG_CHILD, R.TAG_ELDERLY):
                continue
            if any(_affirmed(b, k) for k in kws):
                return True
        return False
    if key == "drug_names":
        if any(a in b for rule in R.DRUG_CLASSES for a in rule.aliases):
            return True
        return bool(prof.get(F_DRUGS))       # 档案里记着具体药名
    if key == "anticoag_use":
        if any(a in b for rule in R.DRUG_CLASSES
               if "抗凝" in rule.cls or "抗血小板" in rule.cls
               for a in rule.aliases):
            return True
        return bool(_GAP_EVID_RE["cardiac_history"].search(b)
                    or re.search(r"阿司匹林|华法林|氯吡格雷|波立维|替格瑞洛|"
                                 r"利伐沙班|达比加群|抗凝|抗血小板", b))
    rex = _GAP_EVID_RE.get(key)
    return bool(rex and rex.search(b))


# ---------------------------------------------------------------------------
# 二、写入：合并进档案 + 追加主诉时间线
# ---------------------------------------------------------------------------
def _merge_csv(old: str, new: str) -> str:
    """逗号/顿号分隔字段的合并（去重保序）。"""
    items: list[str] = []
    for blob in (old, new):
        for it in re.split(r"[、,，;；]", blob or ""):
            it = it.strip()
            if it and it not in items:
                items.append(it)
    return "、".join(items)


def timeline_entries(conv_id: int | None = None) -> list[str]:
    """读时间线（**去掉序号前缀**，避免重复追加时序号层层叠加）。

    2026-09-15 起时间线**按会话隔离**（存 storage.timelines 表）：
    主诉叙事属于「这次对话」，换人设/换话题后旧会话的症状被新会话
    当成当前事实引用，就是 P0-1 串档事故（寒湿人设的"苔白腻便黏"
    被塞进阴虚火旺人设的回答里）。conv_id 为空时退回读档案里的旧键
    （兼容历史数据），但新写入一律带会话。
    """
    if conv_id is not None:
        raw = storage.get_timeline(conv_id)
    else:
        raw = storage.get_profile().get(F_TIMELINE, "")
    if not raw:
        return []
    out: list[str] = []
    for part in re.split(r"[；;]", raw):
        part = part.strip().lstrip(_TIMELINE_MARKS).strip()
        if part:
            out.append(part)
    return out


def append_timeline(text: str, conv_id: int | None = None,
                    limit: int = TIMELINE_MAX) -> list[str]:
    """把本轮主诉追加进时间线（去重；超长丢最旧的）。"""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) < 4:
        return timeline_entries(conv_id)
    # 只保留前 60 字做摘要式记录，避免时间线膨胀
    entry = text[:60] + ("…" if len(text) > 60 else "")
    items = [e for e in timeline_entries(conv_id) if e not in entry and entry not in e]
    items.append(entry)
    items = items[-limit:]
    storage.set_timeline(
        conv_id,
        "；".join(f"{_TIMELINE_MARKS[i]}{e}" for i, e in enumerate(items)))
    return items


def record_herbs(herb_hits, conv_id: int | None = None) -> str:
    """把「用户正在服用的中药/食疗」写进**本会话**档案。

    为什么必须记：安全风险大半来自**交叉**（降压药 × 含甘草制剂、
    痰湿体质 × 滋腻的阿胶）。用户第 3 轮交代了在吃什么，第 4 轮再问
    「那我该留哪样」时，如果档案里没记，扫描就只能看本轮那一句话，
    交叉判断立刻失效。

    2026-09-16：记录范围从"全局用户"收窄到"本会话"——跨会话的安全
    背景不再自动继承，用户主动问起时由 memory 的 global 条目补位。
    """
    taking = [h.name for h in (herb_hits or []) if getattr(h, "taking", False)]
    if not taking:
        return ""
    merged = _merge_csv(storage.get_profile(conv_id).get(F_SUPPLEMENTS, ""),
                        "、".join(taking))
    storage.set_profile(F_SUPPLEMENTS, merged, conv_id)
    return merged


# 「我停了」「不吃了」类表述——档案**必须能减**，否则安全判断会用过期信息。
# 反例：用户在 3 轮前说在吃附子理中丸，第 5 轮说"我已经停了"，
# 如果档案只增不减，之后每一轮都会继续按"在服附子"给判读，
# 既吓人又失真，用户很快就会不再相信这些提示。
# 这一组只用于**廉价预筛**（没有这些词就整段跳过），真正的判定在
# _stop_in_clause 里做，所以这里宁可宽松：漏筛只多花几微秒，误筛会漏掉停用。
_STOP_WORDS = ("停", "戒", "不吃", "不喝", "没吃", "没喝", "断了", "没有了")


def note_stops(text: str, conv_id: int | None = None) -> list[str]:
    """识别"我已经停了 XX"，从**本会话**档案的「在服中药/食疗」里摘掉它。

    只在**明确提到停用**时移除，不做任何推测（宁可留着让用户手动改，
    也不要凭猜把在服的东西删掉——那才是真的危险）。

    按**分句**匹配而不是整段匹配：反例「我停了降压药，但还在吃附子理中丸」
    若按整段看，"停了"和"附子"同在 30 字内，会把附子误删；
    按 ，。； 切句后，"停了"与"附子"落在不同分句，不会误判。
    """
    t = text or ""
    if not any(w in t for w in _STOP_WORDS):
        return []
    cur = storage.get_profile(conv_id).get(F_SUPPLEMENTS, "")
    if not cur:
        return []
    clauses = [c for c in re.split(r"[，。；;！？\n]", t) if c.strip()]
    stopped: list[str] = []
    for name in re.split(r"[、,，;；/]", cur):
        name = name.strip()
        if not name:
            continue
        # 取档案名的前 2 个字做匹配：用户说"附子理中丸"，档案名可能只记"附子"，
        # 直接用全名匹配会漏（"附子理中丸"里确实含"附子"，但反向不成立）。
        stem = name[:2]
        if len(stem) < 2:
            continue
        for cl in clauses:
            if stem not in cl:
                continue
            if _stop_in_clause(cl, stem):
                stopped.append(name)
                break
    if not stopped:
        return []
    rest = [n.strip() for n in re.split(r"[、,，;；/]", cur)
            if n.strip() and n.strip() not in stopped]
    storage.set_profile(F_SUPPLEMENTS, "、".join(rest), conv_id)
    return stopped


def _stop_in_clause(clause: str, stem: str) -> bool:
    """同一分句里，药名与停用词是否构成"停了某药"的关系。"""
    i = clause.find(stem)
    if i < 0:
        return False
    before, after = clause[:i], clause[i + len(stem):]
    # 药名在前：「附子…已经停了」「附子理中丸我上周停了」——允许中间夹时间状语
    if any(s in after[:12] for s in _AFTER_STOP):
        return True
    # 停用词在前：「停了附子」「不吃附子了」
    return any(s in before[-8:] for s in _BEFORE_STOP)


_AFTER_STOP = ("停了", "戒了", "停掉", "停了药", "不吃了", "不喝了", "没吃了",
               "没喝了", "不再吃", "没再吃", "已经停", "已经不吃")
_BEFORE_STOP = ("停了", "戒了", "停掉", "不吃", "不喝", "没吃", "没喝", "别再吃")


def record_safety_hits(hits, conv_id: int | None = None) -> None:
    """兜底：把本轮所有命中写进**本会话**档案摘要（便于记忆面板展示与用户纠正）。"""
    if not hits:
        return
    names = [h.name for h in hits]
    merged = _merge_csv(storage.get_profile(conv_id).get("安全提示记录", ""),
                        "、".join(names))
    storage.set_profile("安全提示记录", merged, conv_id)


def update_from_message(text: str, conv_id: int | None = None) -> dict:
    """一轮输入 → 更新**本会话**结构化档案 + 时间线；返回本轮新增/更新的字段。"""
    changed = extract_facts(text)
    prof = storage.get_profile(conv_id)
    for k, v in changed.items():
        if k in (F_CHRONIC, F_DRUGS):
            merged = _merge_csv(prof.get(k, ""), v)
            if merged != prof.get(k, ""):
                storage.set_profile(k, merged, conv_id)
        elif prof.get(k) != v:
            storage.set_profile(k, v, conv_id)
    # 体质判定由问诊流程写入，这里不动
    # 档案要能减：用户说"我停了 XX"，就把在服清单里的它摘掉
    stopped = note_stops(text, conv_id)
    if stopped:
        changed["_已停用"] = "、".join(stopped)
    if len((text or "").strip()) >= 6:
        append_timeline(text, conv_id)
    return changed


# ---------------------------------------------------------------------------
# 三、一致性校验：档案之间自相矛盾时提示复核 / 复测
# ---------------------------------------------------------------------------
# 判据是"体质标签"与"自述症状标签"的方向是否相反。方向相反的调理原则
# 几乎是镜像的（滋阴 vs 温阳），照旧执行会往反方向使劲，所以必须暴露出来。
_CONSTITUTION_TAGS: dict[str, set[str]] = {
    "阴虚": {R.TAG_YIN_DEF},
    "阳虚": {R.TAG_COLD},
    "气虚": set(),
    "痰湿": {R.TAG_DAMPNESS},
    "湿热": {R.TAG_DAMP_HEAT},
    "血瘀": set(),
    "气郁": set(),
    "特禀": set(),
    "平和": set(),
}

# 冲突表：(体质关键词, 与它相反的 tag, 说明)
_CONFLICTS = (
    ("阴虚", R.TAG_COLD,
     "档案判定为**阴虚质**，但你自述明显**畏寒怕冷、手脚冰凉**——"
     "阴虚的调理方向是滋阴润燥（偏凉润），而畏寒提示阳气不足（需温阳），"
     "两者方向相反，不能同时按阴虚来调。"),
    ("阳虚", R.TAG_YIN_DEF,
     "档案判定为**阳虚质**，但你自述**口燥咽干 / 手足心热 / 盗汗**——"
     "这是阴虚的表现，与阳虚的调理方向（温阳）相反。"),
    ("阴虚", R.TAG_DAMPNESS,
     "档案判定为**阴虚质**，但你的表现（舌苔白腻、身重、大便黏）更偏**痰湿**——"
     "滋阴药多滋腻，会加重湿困。"),
    ("痰湿", R.TAG_COLD,
     "档案判定为**痰湿质**，同时又有明显**畏寒**表现——"
     "这更像「寒湿 / 脾阳不足」，与单纯痰湿的清热化湿思路不同，需区分。"),
    ("湿热", R.TAG_COLD,
     "档案判定为**湿热质**，但你自述**畏寒**——"
     "寒与热方向相反，需先辨清寒热再谈调理。"),
)


def profile_conflicts(profile: dict | None = None,
                      extra_text: str = "",
                      conv_id: int | None = None) -> list[dict]:
    """**本会话**档案（+本轮自述）的一致性校验，返回冲突列表。

    冲突不阻塞回答，但**必须显式提示用户复核**，并在界面给出复测入口
    （P2-3 的"记忆面板"就是用来干这个的）。
    """
    prof = profile if profile is not None else storage.get_profile(conv_id)
    const = prof.get(F_CONSTITUTION, "")
    blob = extra_text or ""
    tags = _tags_of(blob)
    # 档案自身也带症状线索（如"兼夹痰湿质"）
    tags |= _tags_of(" ".join(f"{k}{v}" for k, v in prof.items()))

    out: list[dict] = []
    for key, bad_tag, why in _CONFLICTS:
        if key in const and bad_tag in tags:
            out.append({
                "type": "体质与症状方向相反",
                "constitution": const,
                "conflict_tag": R.TAG_LABEL.get(bad_tag, bad_tag),
                "detail": why,
                "action": "建议先复核体质判定，必要时重新做一次 27 题辨识"
                          "（可在「我的档案」里点「重新测一次」）。",
            })
    return out


def _tags_of(text: str) -> set[str]:
    from app.safety import tag_from_text
    return tag_from_text(text or "")


def conflict_block(conflicts: list[dict]) -> str:
    """冲突渲染成提示词区块。"""
    if not conflicts:
        return ""
    lines = ["【⚠️ 档案一致性校验发现问题（必须在本轮回答里提示用户）】"]
    for c in conflicts:
        lines.append(f"· {c['detail']}")
        lines.append(f"  处理：{c['action']}")
    lines.append("注意：语气要平和，不要说「你的档案错了」，"
                 "而是「这里有个对不上的地方，我们核一下」。")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 四、跨轮串联：把"第几轮说了什么"整理成材料，强制回答里出现整合结论
# ---------------------------------------------------------------------------
# 症状方向反转：不是"同一个人改口"，而可能是"换了咨询对象"。
# ---------------------------------------------------------------------------
# 为什么从"两对硬编码"改成"两条轴"（2026-09-16，架构层问题一）：
# 早先只写了 (寒, 阴虚) 与 (寒, 湿热) 两对。实测的换人设路径是
# **寒湿 → 湿热 → 阴虚**：第一跳（寒→湿热）能抓到，第二跳
# （湿热→阴虚）抓不到——湿热与阴虚都属于"热"的方向，但**湿与燥相反**，
# 需要另一条轴才能识别。轴化之后任何方向反转都能覆盖，不必穷举组合。
_FAMILY_AXES: dict[str, tuple[str, ...]] = {
    "寒": (R.TAG_COLD,),
    "热": (R.TAG_DAMP_HEAT, R.TAG_YIN_DEF),
    "湿": (R.TAG_DAMPNESS, R.TAG_DAMP_HEAT),
    "燥": (R.TAG_YIN_DEF,),
}
# 互斥的轴对：同时出现说明"当前主诉"与"既往记录"不是同一个状态
_CONFLICT_AXES: tuple[tuple[str, str], ...] = (("寒", "热"), ("湿", "燥"))
# 轴的人话名（用于"你之前说的是寒，这次说热"这种确认话术）
_AXIS_LABEL: dict[str, str] = {
    "寒": "寒（畏寒怕冷、喜热）",
    "热": "热（口苦苔黄腻 / 潮热盗汗、手足心热）",
    "湿": "湿（苔腻、身重、大便黏）",
    "燥": "燥（口燥咽干、便干）",
}


def _axes(tags: set[str]) -> set[str]:
    """一组 tag 落在哪些症状轴上。"""
    return {name for name, ts in _FAMILY_AXES.items()
            if set(ts) & (tags or set())}


def _axis_conflict(prev_tags: set[str], cur_tags: set[str]) -> dict | None:
    """两组 tag 是否在互斥轴上相反（返回 {'axis','prev','cur'}）。"""
    a, b = _axes(prev_tags), _axes(cur_tags)
    for x, y in _CONFLICT_AXES:
        if x in a and y in b:
            return {"axis": f"{x}↔{y}", "prev": x, "cur": y}
        if y in a and x in b:
            return {"axis": f"{y}↔{x}", "prev": y, "cur": x}
    return None

_RESET_CONFIRM = """【⚠️ 症状方向反转：先确认对象，再辨证（本轮最高优先级之一）】
当前描述与本次会话此前的记录在**互斥方向**上相反（寒↔热、或湿↔燥）。
这不是"同一个人改口"的同义词——也可能是**这次说的是另一个人**。
在用户确认之前，必须执行：

1. 回答开头先问一句对象确认：「我们这次说的还是**您本人**吗？
   之前记下的那些表现（{prev_label}），现在还适用吗——**还是这次说的是另一位**？」
2. 既往记录**不得**作为本轮辨证依据，也不得复述成「你之前提过…」来论证本轮证型；
3. **不要**把两边硬合并成「兼夹证」（如"阴虚兼湿""寒热错杂"）——
   方向相反的信息硬融合，会得出"先化湿再滋阴"这类两头不讨好的方案，
   对任何一边都可能是有害的；
4. 得到确认前，**只按当前主诉**辨证，旧记录仅作背景信息保留。"""

# P1-2：关系词已明确（"我妹妹""我爱人""我儿子"）→ **只声明边界，不反问**。
# 为什么这也写成硬块：不写清楚，模型仍会习惯性补一句"是不是问您自己"，
# 而用户刚刚已经说了是替谁问——那是"没在听"的观感（实测反馈的原话）。
_RESET_DECLARE = """【✅ 咨询对象已由用户说明为「{relation}」：直接按新对象处理】
用户本轮明确说这是**另一位**（{relation}），并且该对象的表现与会话此前
记录的方向相反（寒↔热、或湿↔燥）。**不要**反问"是不是您本人"——
他已经说清了，再问一次是多余的摩擦。正确做法：

1. 开头**一句话声明边界**即可，例如：「明白，这次说的是{relation}，
   前面记的那些是您本人的情况，**我不会套到他/她身上**。」
2. 既往记录（含本会话档案里的慢病、用药、体质判定）**一律不得**参与
   本轮辨证——**严禁**出现"你之前提过…所以…"这种指代；
   **严禁**把两边硬合并成"兼湿""寒热错杂"这类兼夹证；
3. 只按**本轮给出的这位的情况**辨证；他/她的信息不足时，按正常追问
   机制问这一位的（舌象、二便、寒热、在服药物…），不要用本人的信息顶替；
4. 若这位的情况也明显不足（如只说了"潮热盗汗"四个字），
   先追问再给方案，别为了给方案而借用任何旧信息。"""

# 主语切换确认（2026-09-15 上线清单②）：同一会话里"替女儿问"切回"我说自己"
# 时，必须显式确认对象，防止女儿的年龄/症状被当成用户本人的档案。
_SUBJECT_SWITCH_CONFIRM = """【⚠️ 咨询对象疑似切换（本轮先确认再辨证）】
本会话此前的记录是关于用户的**家人**（原话提到了亲属），本轮描述回到了
「我」自己。回答开头必须先向用户确认一句：
「我们现在聊的是您本人的情况，对吧？——之前提到您家人的那些信息
（年龄、症状、体质），我不会套用到您身上。」
在用户确认之前：亲属相关的既往信息一律**不得**作为本轮辨证依据；
若用户确认是本人，则只按**本人**在本会话中给出的信息辨证，
亲属的年龄/症状/体质不得参与本证判断。"""

# ---------------------------------------------------------------------------
# P1-2：隔离确认要**按置信度分级**，不是每次都反问
# ---------------------------------------------------------------------------
# 实测反馈：「每次都问一遍"是不是您本人"」属于过度保守，带来多余摩擦。
# 用户明确说出关系词（"我妹妹""我爱人""我儿子"）时，**直接推断为新对象**即可，
# 不必再反问——反问一件他已经说清的事，只会让人觉得系统没在听。
# 只有关系词**缺失或模糊**（"再问一个""还有一个""他"）时才触发确认。
#
# 关系词判据：必须是"与人相关"的词。刻意不把单独出现的「他/她」算作明确关系词——
# 「他说我湿气重」里的"他"可能只是转述，不足以断定换了咨询对象。
_RELATION_RE = re.compile(
    r"我(?:的)?(?:妹妹|妹|姐姐|姐|弟弟|弟|哥哥|哥|爱人|老婆|老公|先生|太太|"
    r"媳妇|妻子|丈夫|儿子|女儿|孩子|宝宝|爸|爹|父亲|妈|娘|母亲|婆婆|公公|"
    r"岳父|岳母|奶奶|爷爷|姥姥|姥爷|外婆|外公|外甥|侄子|侄女|表妹|表姐|表哥|"
    r"朋友|同事|邻居|同学|上司|领导)"
    r"|(?:我)?(?:老婆|老公|爱人|太太|婆婆|公公|岳父|岳母)(?:他|她)?"
    r"|(?:家人|家里人|亲戚|另一半)")
# 注意：**故意不包含**「还有一个」「再问一个」「他/她」——那些是"关系词模糊"
# 的情形，正是应该反问确认的那一类（P1-2 的两侧判据就在这一行）。


def relation_of(text: str) -> str:
    """本轮文本里的**明确关系词**（空串 = 没提到关系，属模糊情形）。"""
    m = _RELATION_RE.search(text or "")
    return m.group(0) if m else ""


def subject_switch(current: str, conv_id: int | None = None,
                   profile: dict | None = None) -> dict | None:
    """检测本会话里**症状方向反转**——判断"是不是换人了"，而不是默认改口。

    三条既往来源：① 本会话时间线（本轮之前各轮）；② 本会话的体质判定；
    ③ 时间线里更早的记录。任何一个与当前主诉在互斥轴上相反，就返回冲突描述，
    由 `cross_turn_brief` 渲染成"先确认对象/时间点"的强制块。

    为什么不能默认"同一个人改口"（架构层问题一，真实事故）：
    用户先以寒湿人设提问（苔白腻、便黏、畏寒），再以阴虚火旺人设提问
    （潮热盗汗、舌红少苔）。系统把两边硬融成"阴虚兼湿"，给出
    "先化湿再滋阴"——**两头不讨好**；更糟的是把上一人设的用药与慢病
    当成"这个人的既有事实"，整段建议建立在虚假前提上。

    Returns: {"prev_label":..., "cur_label":..., "where":..., "axis":...,
              "explicit":..., "relation":..., "mode":...} 或 None。
    """
    from app.safety import tag_from_text
    cur_tags = tag_from_text(current or "")
    if not cur_tags:
        return None
    prof = profile if profile is not None else storage.get_profile(conv_id)

    past_sources = (
        ("本会话此前的描述", " ".join(timeline_entries(conv_id)[:-1])),
        ("本会话的体质判定", prof.get(F_CONSTITUTION, "")),
    )
    for where, blob in past_sources:
        if not blob:
            continue
        hit = _axis_conflict(tag_from_text(blob), cur_tags)
        if hit:
            return _switch_payload(hit, where, current)
    return None


def _switch_payload(hit: dict, where: str, current: str) -> dict:
    """统一装配反转描述 + **该不该反问**（P1-2 置信度分级）。"""
    rel = relation_of(current or "")
    return {"prev_label": _AXIS_LABEL.get(hit["prev"], hit["prev"]),
            "cur_label": _AXIS_LABEL.get(hit["cur"], hit["cur"]),
            "axis": hit["axis"], "where": where,
            "prev_axis": hit["prev"], "cur_axis": hit["cur"],
            # ── P1-2：关系词明确 → 直接按新对象处理（不反问，减少摩擦）；
            #         关系词缺失/模糊 → 才反问"这是不是您本人"。
            "explicit": bool(rel),
            "relation": rel,
            "mode": "new_subject" if rel else "confirm_subject"}


# 兼容旧名（早期版本使用的函数名，语义已由 subject_switch 扩展）
direction_shift = subject_switch


def cross_turn_brief(conv_id: int | None = None,
                     current: str = "",
                     profile: dict | None = None) -> str:
    """产出**本会话**的跨轮整合材料（确定性组装，整合句由 LLM 写）。

    2026-09-15 起**只读本会话的时间线**；2026-09-16 起档案也按会话隔离，
    所以这里看到的一切都只属于"这次对话"。跨会话的长期信息由
    memory.recall 单独注入，且**只有用户主动问起时才注入**——
    混入旧会话原话正是串档事故的源头。

    Returns: 供注入提示词的材料；不足两轮时返回空串（没什么可整合的）。
    """
    entries = timeline_entries(conv_id)
    # 最后一条是本轮（update_from_message 先写时间线再生成回答），
    # 「跨轮整合」要看的是本轮之前的各轮
    if len(entries) < 2:
        return ""
    switch = subject_switch(current, conv_id, profile)
    if switch:
        # 方向反转时，既往原话**不得**以"这个人的事实"的形态出现——
        # 只作为"待确认材料"列出（架构层问题一：记忆是带归属的证据）。
        lines = [f"【跨轮材料（本次会话历轮原话——⚠️ 方向反转，**归属待确认**，"
                 f"不得作为本轮辨证依据）】"]
        for e in entries[:-1]:
            lines.append(f"  [往轮·待确认] {e}")
        lines.append(f"  [本轮] {entries[-1]}")
        lines.append("")
        if switch.get("explicit"):
            # P1-2：用户**已经说明**换人了（"我妹妹""我爱人"）→ 不许再反问
            # "是不是您本人"，那是在问一件他已经说清的事（实测反馈：
            # 每次都反问属于过度保守、带来多余摩擦）。直接声明隔离边界即可。
            lines.append(_RESET_DECLARE.replace(
                "{relation}", switch.get("relation") or "另一位"))
        else:
            # 关系词缺失/模糊（"再问一个""还有一个"）→ 反问确认是对的
            lines.append(_RESET_CONFIRM.replace("{prev_label}",
                                                switch["prev_label"]))
        lines.append(f"（反转方向：{' ↔ '.join([switch['prev_axis'], switch['cur_axis']])}"
                     f"；来源：{switch['where']}。）")
    else:
        lines = ["【跨轮材料（本次会话中用户历轮原话，按时间顺序）】"]
        for e in entries:
            lines.append(f"  {e}")
    # 主语切换确认：此前在替家人问、本轮回到"我"（上线清单②验证场景：
    # 从"女儿"切回"我"时，女儿的黄腻苔不得带到父亲身上）
    cur_text = entries[-1] or current or ""
    if (any(k in e for e in entries[:-1] for k in _KIN_WORDS)
            and not any(k in cur_text for k in _KIN_WORDS)
            and re.search(r"我|自己|本人", cur_text)):
        lines.append("")
        lines.append(_SUBJECT_SWITCH_CONFIRM)
    return "\n".join(lines)


CROSS_TURN_RULE = """【每轮必须给出「跨轮整合结论」】
用户是一轮一轮慢慢交代的（先说累，再说舌象，最后才补年龄、慢病和正在吃的东西）。
**你必须把历轮信息合起来解读，而不是只回应最后一句**。回答里必须出现
一句明确的整合结论，形如：

  「把你这几轮说的串起来看：**你第 1 轮说的『睡够了也没精神』，加上这轮的
   舌胖齿痕、苔白腻、身重，以及 45 岁女性、体重增加、大便黏——**
   这不是单纯的『虚』，而是**本虚（脾阳不足）+ 标实（寒湿困脾）**，
   阳气不够、湿浊运化不掉，所以才会又累又沉。」

要求：① 必须**指回具体的轮次或具体原话**，不要泛泛说"综合来看"；
② 结论要落到病机（本虚是什么、标实是什么），不要停在症状罗列；
③ 若本轮信息与之前**矛盾**，要主动指出并询问，而不是选一个用。"""


# ---------------------------------------------------------------------------
# 五、信息缺口 → 强制追问
# ---------------------------------------------------------------------------
def _conv_blob(text: str, conv_id: int | None) -> str:
    """扫描用上下文 = 本轮文本 + **本会话**历史主诉。

    为什么必须带时间线：自述（舌象/二便/寒热）往往分散在前面几轮，
    只扫本轮会让"上一轮刚说过舌象"的用户被重复追问（2026-09-15 验收
    实测：补了自述仍走追问分支，因为 gaps 看不到时间线）。
    时间线已按会话隔离（P0-1），这里不会引入别的会话的信息。
    """
    blob = text or ""
    if conv_id is not None:
        blob += "\n" + "；".join(timeline_entries(conv_id))
    return blob


def gaps(profile: dict | None = None, text: str = "",
         require: tuple[str, ...] | None = None,
         conv_id: int | None = None) -> list[str]:
    """判断当前缺哪些"给方案前必须有"的信息（**阻塞项**）。

    require 默认按场景给：问调理方案时要求最全；普通问答只要求年龄性别。
    只有这些项缺失才走 need_more 追问分支、不给方剂级内容。
    场景化的"值得同时问"（出血倾向/支架史/月经）走 advisory_gaps——
    它们重要但不阻塞：一个在服抗凝药的人永远会被追问出血史，
    方案就永远出不来，正确的做法是"方案+同时追问"。

    2026-09-16 第四轮（P0）：**主体感知 + 已知比对**。
    ① 缺口判定基于 `subject_scope`：本人场景看本人档案+本人时间线；
       替家人问（"我爸爸 68 岁"）时看**这一位**已交代的原话——
       反馈实测"68 岁/男/在吃药"三项俱在仍被追问"你的年龄性别/有没有吃药"，
       根因是档案抽取正确地剥离了亲属分句（防污染守卫，保留），
       而 gaps 只看档案、看不到那些被剥离的事实。
    ② 每项判定改走 `gap_known`（宽口径证据正则）——已知判定漏判的代价
       是重复提问（用户体感"没在听我说话"），误判的代价是多问一句，
       所以这一层刻意比档案抽取的写入守卫宽。
    """
    prof = profile if profile is not None else storage.get_profile(conv_id)
    require = require or ("age_sex", "tongue", "stool_urine", "cold_heat",
                          "medication", "chronic")
    rel, sblob = subject_scope(text, conv_id)
    if rel:
        # 替家人问：本人档案是**另一个人**的事实，一律不参与
        blob = sblob
        prof_for = {}
    else:
        blob = sblob + "\n" + "\n".join(str(v) for v in prof.values())
        prof_for = prof
    have = {
        "age_sex": gap_known("age_sex", blob, prof_for, rel),
        "tongue": gap_known("tongue", blob, prof_for, rel),
        "stool_urine": gap_known("stool_urine", blob, prof_for, rel),
        "cold_heat": gap_known("cold_heat", blob, prof_for, rel),
        "medication": gap_known("medication", blob, prof_for, rel),
        "chronic": gap_known("chronic", blob, prof_for, rel),
        "chief_time": gap_known("chief_time", blob, prof_for, rel),
    }
    return [g for g in require if not have.get(g, True)]


def advisory_gaps(tags: set[str] | None = None,
                  profile: dict | None = None,
                  text: str = "",
                  conv_id: int | None = None) -> list[str]:
    """场景化追问（**非阻塞**）：按人群提示"这一类用户还应补问什么"。

    - 抗凝/抗血小板人群 → 出血倾向、支架/双抗史（2026-09-15 测试：
      抗凝场景零追问，出血倾向直接决定能不能碰活血类中药）；
    - 女性 + 潮热/盗汗/阴虚 → 月经情况（围绝经期与甲亢的区分线索）。

    2026-09-16 第四轮：**抗凝两问加证据闸门**。反馈实测 47 岁月经量少的
    女性被问"有没有放过心脏支架"——抗凝模板硬贴到毫无抗凝语境的主诉上。
    现在"在服抗凝药"必须有**直接证据**（档案记着抗凝药 / 原话提到
    华法林/阿司匹林/支架/双抗）才问出血史与支架史；只是聊到活血类食材
    而不知道吃不吃抗凝药的，由阻塞项 `anticoag_use` 先问"在不在服"，
    不越过这一问直接铺开出血风险。
    """
    prof = profile if profile is not None else storage.get_profile(conv_id)
    blob = _conv_blob(text, conv_id) + "\n" + "\n".join(str(v) for v in prof.values())
    # 2026-09-16 第五轮（版本退化修复）：**围产期状态下不套抗凝 / 出血模板**。
    # 反馈实测孕妇被追问"有没有牙龈出血、流鼻血、瘀斑、大便发黑"——那是
    # 抗凝模板硬贴到孕期主诉上，与她的情况无关。妊娠/备孕/哺乳期一律跳过
    # 这一组场景追问（真要谈抗凝，是"她本人在服抗凝药"才成立，而那属于
    # 阻塞项 anticoag_use 的范畴，不是这里的模板）。
    try:
        from app import constraints as CN
        perinatal = CN.is_perinatal(CN.detect(text, prof, conv_id))
    except Exception:
        perinatal = False
    out: list[str] = []
    if not perinatal and tags and R.TAG_ANTICOAG in tags:
        # 证据闸门：确认在服/有史才进入出血·支架两问
        if gap_known("anticoag_use", blob, prof):
            if not gap_known("bleeding", blob, prof):
                out.append("bleeding")
            if not gap_known("cardiac_history", blob, prof):
                out.append("cardiac_history")
    female = prof.get(F_SEX) == "女" or "女" in blob
    if not perinatal and female and (R.TAG_YIN_DEF in (tags or set())
                                     or "潮热" in blob or "盗汗" in blob):
        if not re.search(r"绝经|月经紊乱|停经|更年|经期|月经", blob):
            out.append("menopause")
    return out


def screening_keys(tags: set[str], profile: dict | None = None,
                   text: str = "",
                   conv_id: int | None = None) -> list[str]:
    """根据**规则表**决定要不要给西医排查建议（P0-2：规则化触发，不靠场景联想）。

    2026-09-16 第三轮改造：早先这里是一串写死的 if，判断与"证型"耦合，
    于是**同样的慢性症状换个证型就漏给排查**（实测：有的轮次给完整检查项，
    有的轮次整块缺失）。现在触发条件全部写在数据里
    （`app/safety/scripts.py::SCREEN_RULES`），这里只剩一个通用求值器：
    **满足条件必触发，与辨证结果无关**。新增场景 = 加一条数据。

    ⚠️ 2026-09-16 修复记录：此前漏声明 `conv_id` 形参（但函数体回退分支却在用它），
    调用方 `rag._safety_guard` / `agent._prescan` / `multiagent._build_intake`
    三处**全部 TypeError**，又被上层 `except Exception` 静默吞掉 ——
    表现为"安全事件与追问清单整条链路消失"，而四个纯规则单测全绿都查不出来。
    凡是给这些函数加/改签名，必须同步跑 `tools/e2e_safety_check.py` 打真服务。
    """
    from app.safety.scripts import SCREEN_RULES

    prof = profile if profile is not None else storage.get_profile(conv_id)
    blob = _conv_blob(text, conv_id) + "\n" + "\n".join(str(v) for v in prof.values())
    cur = set(tags or set())

    fired: set[str] = set()
    for rule in SCREEN_RULES:
        need_tags = set(rule.get("tags") or ())
        if need_tags and not need_tags <= cur:
            continue
        # `not_tags`：命中即**不触发**。
        skip_tags = set(rule.get("not_tags") or ())
        if skip_tags & cur:
            continue
        # `not_any`：blob 命中任一信号即**不触发**。比 tag 更精确——能把
        # "备孕"与"已妊娠"分开（两者 tag 相同、场景不同，第五轮实测）。
        no_sig = rule.get("not_any") or ()
        if no_sig and any(_SCREEN_RES[s].search(blob) for s in no_sig):
            continue
        all_sig = rule.get("all") or ()
        if all_sig and not all(_SCREEN_RES[s].search(blob) for s in all_sig):
            continue
        any_sig = rule.get("any") or ()
        if any_sig and not any(_SCREEN_RES[s].search(blob) for s in any_sig):
            continue
        fired.add(rule["key"])
    # 输出顺序 = 规则表顺序（稳定可复现；慢病类排在前面）
    return [r["key"] for r in SCREEN_RULES if r["key"] in fired]


# ---------------------------------------------------------------------------
# 六、给前端「记忆面板」用的视图（P2-3）
# ---------------------------------------------------------------------------
def panel(conv_id: int | None = None) -> dict:
    """用户可见、可修改的**本会话**档案视图（P2-3）。

    2026-09-16：面板只展示本次对话的档案；历史遗留的全局档案
    （legacy）单独给出计数，让用户知道"库里还有这些旧数据，
    但它们不会进入任何对话"，并可在面板里删掉。
    """
    prof = storage.get_profile(conv_id)
    conflicts = profile_conflicts(prof, conv_id=conv_id)
    fields = []
    for k in PROFILE_FIELDS:
        if prof.get(k):
            fields.append({"key": k, "value": prof[k], "editable": True})
    others = [{"key": k, "value": v, "editable": True}
              for k, v in prof.items() if k not in PROFILE_FIELDS]
    return {"fields": fields + others, "conflicts": conflicts,
            "timeline": timeline_entries(conv_id),
            "scope": "conversation",
            "conv_id": conv_id,
            "archived_profile_keys": sorted(storage.archived_profile().keys()),
            "archived_memories": storage.count_archived_memories()}
