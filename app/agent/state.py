# -*- coding: utf-8 -*-
"""问诊状态机：把「望闻问切 → 辨证 → 论治」落成一台可推进、可落库、可测试的状态机。

为什么要状态机，而不是让 LLM 自由发挥
--------------------------------------
体质辨识是一条**有明确终点的流程**：问完 27 题 → 算分 → 出报告 → 转入追问。
如果把整个流程交给 LLM 自由控制，会出现三类典型问题（本项目的 RAG 提示词
迭代中已亲历）：
  ① 跳步：用户还没答完就开始下结论；
  ② 复读：反复问同一题，或用户答完还在追问；
  ③ 不可观测：出问题时无法回答"它现在走到哪一步了"。
所以骨架用**确定性状态机**（阶段集合 + 合法转移 + 进度计数都在代码里），
每个阶段内部再交给 LLM 自由对话。这是「**骨架确定 + 血肉自由**」的混合架构，
也是本项目最有面试价值的设计决策。

阶段定义与转移
--------------
    IDLE ──begin()──▶ COLLECTING ──答完最后一题──▶ SCORING ──judge()──▶ REPORTING
                                                              │
                                                              └──enter_followup()──▶ FOLLOWUP
    REPORTING / FOLLOWUP 阶段用户随时可以重新 begin() 开始新一轮辨识。

持久化
------
状态序列化为 JSON 存进 SQLite（表 consult_state），键为会话 id。
这样 Web 服务重启后问诊进度不丢——用户答到第 18 题去吃饭，回来还能接着答。

用法：
    from app.agent.state import ConsultState
    st = ConsultState.load(conv_id)      # 取（或新建）
    st.begin()
    st.record(score=4)                   # 给当前题打分
    st.save(conv_id)
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

from app.agent import constitution as C

# 阶段常量
IDLE = "idle"
COLLECTING = "collecting"
SCORING = "scoring"
REPORTING = "reporting"
FOLLOWUP = "followup"

STAGE_LABEL = {
    IDLE: "未开始",
    COLLECTING: "正在收集信息",
    SCORING: "正在计算判定",
    REPORTING: "已出判定结果",
    FOLLOWUP: "调养追问",
}


def _flat_questions() -> list[tuple[str, int]]:
    """把量表摊平成 [(体质key, 第几题)] 的答题顺序。

    顺序按量表定义走（平和质 → 气虚 → … → 特禀质）。这样设计而不是
    「每种体质一次问完」，是为了让相邻问题**在语义上跳跃**——
    连续问三个阳虚题会诱导用户顺着答「是」，分散开可以降低这种应答偏差。
    """
    return [(t.key, i) for t in C.SCALE for i in range(len(t.questions))]


@dataclass
class ConsultState:
    """一次问诊的完整状态（可 JSON 序列化）。"""

    stage: str = IDLE
    order: list[list] = field(default_factory=lambda: [list(x) for x in _flat_questions()])
    cursor: int = 0                                  # 当前问到第几题（index into order）
    answers: dict[str, list[int]] = field(default_factory=dict)  # key → [每题分值]
    last_result: str = ""                            # 判定结果文本（缓存，避免重复算）
    primary_key: str = ""                            # 主体质 key
    turns: int = 0                                   # Agent 已处理的轮数（护栏用）
    asked_turn: int = -1                             # 当前题是在第几轮被问出口的
    updated_at: float = field(default_factory=time.time)

    # ---------- 序列化 ----------
    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str | None) -> "ConsultState":
        if not s:
            return cls()
        try:
            d = json.loads(s)
        except json.JSONDecodeError:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, conv_id: int) -> None:
        from app import storage
        self.updated_at = time.time()
        storage.save_consult_state(conv_id, self.stage, self.to_json())

    @classmethod
    def load(cls, conv_id: int) -> "ConsultState":
        from app import storage
        st = storage.load_consult_state(conv_id)
        return cls.from_json(st["payload"] if st else None)

    # ---------- 状态推进 ----------
    def begin(self) -> None:
        """开始新一轮辨识：清空作答，回到第一题。

        注意 last_result / primary_key 一并清掉——它们是上一轮的结论，
        留着会让 LLM 误以为已经判过了。
        """
        self.stage = COLLECTING
        self.cursor = 0
        self.answers = {}
        self.last_result = ""
        self.primary_key = ""
        self.turns = 0
        self.asked_turn = -1

    @property
    def total(self) -> int:
        return len(self.order)

    @property
    def answered(self) -> int:
        """已作答题数。

        用 cursor（题目游标）而不是 `sum(len(v))`：answers 里的每个数组是
        按体质**预分配好长度**的（未答位置留 0），直接数长度会把没答的题也算进去，
        实测表现为"答题进度虚高、明明才答 5 题却显示 6/27"。
        """
        return self.cursor

    @property
    def current(self) -> tuple[str, int] | None:
        """当前待答题 (体质key, 题序号)；已答完返回 None。"""
        if self.cursor >= len(self.order):
            return None
        k, i = self.order[self.cursor]
        return k, i

    def current_question(self) -> dict | None:
        """当前题的完整信息（题干、所属体质、题号），供工具返回给 LLM。"""
        cur = self.current
        if cur is None:
            return None
        k, i = cur
        t = C.BY_KEY[k]
        return {
            "index": self.answered + 1,
            "total": self.total,
            "type_key": k,
            "type_name": t.name,
            "question": t.questions[i],
            "choices": "；".join(f"{v}={label}" for v, label in C.FREQ_CHOICES),
        }

    def record(self, score: int) -> dict:
        """记录当前题的评分，并推进到下一题；答完则转入 SCORING。"""
        cur = self.current
        if cur is None:
            return {"ok": False, "reason": "所有题目已答完，请先调用 judge_constitution"}
        score = max(1, min(5, int(score)))          # 夹在 1~5，防模型越界
        k, i = cur
        arr = self.answers.setdefault(k, [0] * len(C.BY_KEY[k].questions))
        arr[i] = score                              # 用下标写，天然幂等（重答覆盖）
        self.cursor += 1
        if self.cursor >= len(self.order):
            self.stage = SCORING
        return {"ok": True, "answered": self.answered, "total": self.total,
                "stage": self.stage, "stage_label": STAGE_LABEL[self.stage]}

    def judge(self) -> C.ConsultResult:
        """算分并转为 REPORTING。未答完时会用现有作答尽力判定。"""
        result = C.compute(self.answers)
        self.last_result = C.format_report(result)
        self.primary_key = result.primary_key
        self.stage = REPORTING
        return result

    def enter_followup(self) -> None:
        self.stage = FOLLOWUP

    # ---------- "先问后记"护栏 ----------
    # 实测踩坑：模型会在**同一轮**里先 next_question 拿到题、紧接着就 record_answer
    # 把答案"记"下——可用户根本还没看到这道题，等于替用户编造答案。
    # 提示词反复强调无效，于是加确定性约束：**一道题只有在更早的轮次问出口之后，
    # 才允许被记录**（turns 每轮 +1，asked_turn 记录该题是哪一轮问出口的）。
    def mark_asked(self) -> None:
        """标记"当前这道题已问给用户"（由 next_question 工具调用）。"""
        self.asked_turn = self.turns

    def can_record(self) -> tuple[bool, str]:
        """当前题是否可被记录；返回 (是否允许, 拒绝原因)。"""
        if self.asked_turn >= self.turns:
            return False, (
                "这道题刚问出口（已由界面卡片展示给用户），用户还没有机会作答。"
                "请本轮只用一句话自然引入或回应用户，然后停下等待；"
                "等用户下一轮回答之后再调用 record_answer。"
                "**绝不允许替用户编造答案，也不允许在正文里提问。**")
        return True, ""

    # ---------- 给 LLM 看的阶段摘要 ----------
    def summary_for_llm(self) -> str:
        """当前状态摘要，注入系统提示词，让模型知道"现在走到哪了"。"""
        lines = [f"当前阶段：{STAGE_LABEL[self.stage]}（{self.stage}）"]
        if self.stage == COLLECTING:
            lines.append(f"答题进度：{self.answered}/{self.total}")
            can, why = self.can_record()
            if can:
                lines.append("状态：用户的回答已就绪，可以调用 record_answer 记录当前题。")
            else:
                lines.append("状态：当前题**刚问出口、用户尚未作答**，"
                             "本轮只能把问题讲给用户后停下等待，不要调用 record_answer。")
            lines.append("⚠️ 流程已在进行中：不要调用任何『开始/重来』类工具，"
                         "直接调用 next_question 提问、或 record_answer 记录。")
            q = self.current_question()
            if q:
                lines.append(f"当前待答题：{q['question']}（属于{q['type_name']}）")
        elif self.stage == SCORING:
            lines.append("所有题目已答完，请调用 judge_constitution 计算判定结果。")
        if self.last_result:
            lines.append("")
            lines.append("已得出的判定结果：")
            lines.append(self.last_result)
        return "\n".join(lines)
