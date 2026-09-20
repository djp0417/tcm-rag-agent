# -*- coding: utf-8 -*-
"""判断框架：**先有框架，再用知识填充**（架构层问题六）。

要治的毛病
----------
当前实现是"知识命中什么就讲什么"——输出结构由检索命中情况决定，
于是：同一个问题在不同轮次长得完全不一样；某个维度一旦检索不到就**整段消失**
（实测：穴位维度在多轮里反复缺失）；典籍里的过渡话术被原样输出。

原则（一句话）
--------------
> **框架完整性与知识命中率解耦。**

系统内部先有一套临床判断框架：这个问题属于哪一类、需要哪几类信息、
应该从哪几个维度回答、有哪些必须排除的风险。知识只负责**给框架提供佐证和细节**，
不决定框架长什么样。某维度的知识缺位时，**框架维度仍然存在**，
只是标注「此处缺乏典籍依据，按通用原则处理」，而不是整个维度消失。

引文的价值在判读（同一原则的下半句）
------------------------------------
引用之后必须紧跟「所以对你而言意味着什么」。只引不判比不引更危险——
用户会自己往错误方向理解。这条由 `safety/scripts.py::CITATION_JUDGMENT` 强制。
"""
from __future__ import annotations

from dataclasses import dataclass, field

# 问题类别
CAT_SAFETY = "用药与安全"        # 命中了药材/西药/特殊人群
CAT_ADVICE = "症状调理"          # 求针对自己的调理方案
CAT_SINGLE = "单味宜忌"          # 只问某味药/食材能不能吃
CAT_KNOW = "知识解释"            # 问概念，不要方案

# 各维度的固定名称（可执行部分的四维——无论检索是否命中都必须出现）
DIM_FOOD = "食疗"
DIM_MERIDIAN = "经络穴位"
DIM_ROUTINE = "起居作息"
DIM_EXERCISE = "运动导引"
ACTION_DIMS: tuple[str, ...] = (DIM_FOOD, DIM_MERIDIAN, DIM_ROUTINE, DIM_EXERCISE)

# ---------------------------------------------------------------------------
# 每个维度的**量化要素**（P1-1：维度完整性要契约化）
# ---------------------------------------------------------------------------
# 实测反馈：各维度"每轮齐全"做到了，但**量化程度参差**——食疗有时没有配比、
# 经络有时只说"腹部按揉"而不给穴位定位与时长。定性建议用户没法执行，
# 于是"给了建议"和"没给"在体感上差不多。所以把每个维度必须交代的字段
# 写成契约，和六模块一样由代码要求（缺字段的补全在 contract 里）。
DIM_SPEC: dict[str, tuple[str, ...]] = {
    DIM_FOOD: ("用哪一味/哪几味（药食同源的平性方向优先）",
               "配比（如 陈皮 3~5g + 茯苓 10g）",
               "做法（煮水/煮粥/入膳、煮多久）",
               "频次（每日几次 或 每周几次）",
               "周期（连续吃多久为一个周期，之后停一停再评估）"),
    DIM_MERIDIAN: ("具体穴位名（足三里、关元、三阴交、脾俞…）",
                   "定位（在腿上/脐下几寸，用体表标志描述，不用专业术语）",
                   "每个穴位的时长（按揉 3~5 分钟 / 艾灸 10~15 分钟）",
                   "频度（每日 1 次 或 隔日 1 次）",
                   "禁忌条件（孕期禁合谷/三阴交、皮肤破损处禁灸等）"),
    DIM_ROUTINE: ("具体做什么（几点睡、怎么吃、避什么）",
                  "频次（每天/每周几次）",
                  "要避免的事（熬夜、久坐超过多少小时、生冷）"),
    DIM_EXERCISE: ("运动类型（散步、八段锦、太极这类中低强度）",
                   "单次时长（20~30 分钟）",
                   "强度（微汗即止，能说话不喘为度）",
                   "频率（每周 3~5 次）"),
}

# 排查项 → 人话（"必须排除的风险"）
_SCREEN_LABEL: dict[str, str] = {
    "fatigue_cold_midlife_woman":
        "甲减 / 贫血 / 血糖血脂 / 肝肾功能——这些会**伪装成「亚健康」**，"
        "和「阳虚·寒湿」的表现高度重叠，谈调理前先排掉",
    "hypertension": "血压控制情况（家庭血压监测 + 按医嘱复查血脂血糖肾功能电解质）",
    "diabetes": "血糖控制情况（空腹/餐后血糖、糖化血红蛋白，按医嘱频次）",
    "thyroid": "甲功（TSH、FT4）——甲状腺功能本身会明显影响精力与体重",
    "ambiguous_damp":
        "寒湿 / 湿热 / 脾虚湿困的区分——三者调法相反，要靠舌苔与二便分清，"
        "不能只按一句「湿气重」来调",
    "perimenopause_hot":
        "甲亢 / 血糖异常 / 围绝经期改变——潮热盗汗除更年期外，甲亢等也会这样表现",
}


@dataclass
class Plan:
    """一轮回答的判断框架。"""
    category: str
    dimensions: list[str] = field(default_factory=list)   # 必须覆盖的回答维度
    must_exclude: list[str] = field(default_factory=list)  # 必须排除的风险
    need_info: list[str] = field(default_factory=list)     # 缺则挂起针对性结论
    follow_question: str = ""                              # 缺信息时唯一的追问方向

    def to_dict(self) -> dict:
        return {"category": self.category, "dimensions": self.dimensions,
                "must_exclude": self.must_exclude,
                "need_info": self.need_info}


def build(text: str, intent: str = "knowledge", tags=None,
          hits=None, screening=None, gaps=None) -> Plan:
    """按"问题属于哪一类"给出框架（与检索结果**无关**）。"""
    hits = hits or []
    has_drug = any(_k(h) == "drug" for h in hits)
    has_herb = any(_k(h) in ("herb", "population") for h in hits)
    only_generic = bool(hits) and not has_drug

    if has_drug or has_herb:
        cat = CAT_SINGLE if only_generic else CAT_SAFETY
    elif intent == "advice":
        cat = CAT_ADVICE
    else:
        cat = CAT_KNOW

    if cat == CAT_SAFETY:
        dims = ["结论与档位（能不能用 / 需不需要先确认）",
                "原处方药纪律（不能自行停、不能自行减量）",
                "相对安全的替代方向"]
    elif cat == CAT_SINGLE:
        dims = ["结论与档位", "用法、频次与周期上限", "同用禁忌与停止信号"]
    elif cat == CAT_ADVICE:
        dims = list(ACTION_DIMS)
    else:
        dims = ["概念本义（白话解释）", "对提问者本人的意义"]

    must_exclude = [_SCREEN_LABEL[k] for k in (screening or [])
                    if k in _SCREEN_LABEL]
    return Plan(category=cat, dimensions=dims, must_exclude=must_exclude,
                need_info=list(gaps or []))


def _k(h) -> str:
    return (h.get("kind") if isinstance(h, dict)
            else getattr(h, "kind", "")) or ""


def render(plan: Plan | None) -> str:
    """框架 → 注入提示词的区块。"""
    if not plan:
        return ""
    lines = [
        "【本轮回答框架（**先按框架组织，再用检索到的资料填充佐证**）】",
        f"问题类别：{plan.category}",
        "· **开场直接回答问题本身**。【严禁】用「你这次只提到…没有说过…"
        "所以只能给…」这类清点用户没提供过什么信息的声明开场——那是问诊表，"
        "不是聊天。用户没说过的项目（体质、舌象、大便、怕冷怕热、用药…）"
        "**一个都不要在正文里罗列**；需要他补什么，只在回答结尾的追问里"
        "自然带出（追问清单由系统单独给出）。",
        f"必须覆盖的回答维度：{'、'.join(plan.dimensions)}",
        "  · **框架完整性不受检索结果影响**：某个维度在【参考资料】里找不到依据时，"
        "**保留该维度**并按通用原则处理，**不要整段删掉这个维度**（实测缺陷："
        "穴位维度在多轮回答里整段消失，输出结构被检索命中率决定）。",
        "  · 「资料库无直接依据」这类标注**整篇回答最多出现一次**：多个维度"
        "都缺依据时，在第一次补常识的地方合并注明一句即可，其余处自然带过，"
        "**不要每个维度各贴一遍**（逐条复读会变成模板噪音）。",
    ]
    # 维度量化：只对"本轮确实要给的维度"提要求，避免无关维度也来凑字段
    wanted = [d for d in ACTION_DIMS if d in (plan.dimensions or [])]
    if wanted:
        lines.append("  · **每个维度必须给到可执行的量化要素**（只写方向、"
                     "不写用量与频次的建议等于没给，用户没法执行）：")
        for d in wanted:
            lines.append(f"     - {d}：{'；'.join(DIM_SPEC[d])}")
    if plan.must_exclude:
        lines.append("必须排除的风险（谈调理之前先排掉）：")
        for m in plan.must_exclude:
            lines.append(f"  · {m}")
    if plan.need_info:
        # 缺口清单是**内部掌握**的判断依据，不是给用户的声明材料——
        # 2026-09-16 实测：把全部缺口原样渲染 + "挂起针对性结论"的指令，
        # 模型把它说出了口，变成开场"你没说过体质/舌象/大便…所以只能给
        # 不分型的安全建议"的清点式声明 + 5 问问卷（追问引擎明明限 3 问）。
        # 修法：① 截断到 MAX_QUESTIONS（与追问引擎同一上限）；
        # ② 明示"只给你自己看"，要问的交给追问环节，不进正文。
        from app.inquiry import MAX_QUESTIONS as _MAXQ
        shown = list(plan.need_info)[:_MAXQ]
        lines.append("内部掌握的缺口（**只给你自己看，严禁向用户声明或罗列**）："
                     "这些信息缺失时**照常回答**，只给不分型也成立的通用建议"
                     "（作息、饮食有节、避免久坐、情绪调畅），"
                     "**严禁**用历史记忆里的信息填补；需要用户补什么，"
                     "一律交给回答结尾的追问环节，不在正文另列清单，"
                     "也不写「等你补充信息后再做判断」之类的话：")
        lines.append(f"  · {'、'.join(shown)}")
    return "\n".join(lines)


def action_dims_text() -> str:
    """可执行部分的四维固定清单（供输出契约补全时用）。

    补全文本同样要带量化字段（P1-1）——否则"代码补的那一段"会成为
    全篇最不可执行的部分，用户一眼就看出它是模板。
    """
    return ("食疗（药食同源的**平性**方向，如山药 30g + 茯苓 10g 煮粥，"
            "每周 3~4 次，连吃 2~4 周为一周期）／"
            "经络穴位（足三里：外膝眼下四横指、胫骨外侧一横指，按揉 3~5 分钟；"
            "关元：脐下四横指，艾灸 10~15 分钟；每日或隔日 1 次；"
            "孕期禁按合谷、三阴交，皮肤破损处禁灸）／"
            "起居作息（23 点前睡、三餐定时、避寒就温、每坐 1 小时起身活动）／"
            "运动导引（散步或八段锦，每次 20~30 分钟，微汗即止，每周 3~5 次）")
