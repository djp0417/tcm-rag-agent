# -*- coding: utf-8 -*-
"""Agent 工具层：DeepSeek function calling 的工具定义与执行器。

设计原则：**工具只做确定性的事，语言理解交给模型**
----------------------------------------------------
LLM 擅长的是"把用户说的『我冬天手脚冰凉』理解成第 3 题回答 5 分"，
不擅长的是"记住现在问到第几题""按公式算转化分"。所以：
  · 状态推进、计分、判定 → 工具（确定性代码，可复现）
  · 语义理解、口语 → 分值的映射 → 模型（调用工具时传参）
  · 组织自然语言回答 → 模型
这条分界线画得越清晰，Agent 越不容易跑偏。

六个工具（按状态机阶段划分可用范围）
------------------------------------
    COLLECTING 阶段：next_question / record_answer / restart_consultation
    SCORING  阶段： judge_constitution
    任意阶段：       search_knowledge / remember_fact
把"当前阶段允许哪些工具"写进代码（`allowed_tools()`），而不是靠提示词祈祷
模型自觉——这是状态机与 Agent 结合的关键：**用工具可见性约束行为空间**。

用法：
    box = ToolBox(state=st, conv_id=1, memory=mgr)
    schemas = box.schemas()                  # 交给 LLM 绑定的工具定义
    result = box.call("record_answer", {"score": 4})   # 执行
"""
from __future__ import annotations

import json

from app.agent import constitution as C
from app.agent import state as S

# ---------------------------------------------------------------------------
# 工具定义（OpenAI function calling 格式，langchain 的 bind_tools 直接吃这个）
# ---------------------------------------------------------------------------
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "next_question",
            "description": (
                "获取当前待答题目的信息。注意：题目会由界面卡片自动展示给用户"
                "（含点选按钮），你一般**无需调用本工具**，也不要在正文里复述题干。"
                "仅在需要确认当前题目内容时才调用。"
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_answer",
            "description": (
                "记录用户对**当前题目**的回答。你需要把用户的口语描述映射为 1~5 分："
                "1=没有，2=很少，3=有时，4=经常，5=总是。"
                "例：『特别怕冷，冬天手脚冰凉』→ 5 分；『偶尔有点』→ 2~3 分。"
                "注意：① 只能记录当前题目的回答；② 用户没有正面回答、或将话题转移时"
                "不要调用本工具，先自然地引导回题目；③ 一次只记录一题。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "score": {"type": "integer", "minimum": 1, "maximum": 5,
                              "description": "根据用户描述判定的频率分值 1~5"},
                    "evidence": {"type": "string",
                                 "description": "用户原话中支撑该分值的依据（便于复核）"},
                },
                "required": ["score"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "judge_constitution",
            "description": (
                "计算并得出体质辨识结果。**仅在所有题目全部答完后调用**。"
                "返回九种体质的转化分、主体质与兼夹体质。"
                "拿到结果后，你必须再调用 search_knowledge 检索对应体质的调养资料，"
                "然后才能给用户完整的解读与建议。"
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "检索中医养生知识库（古籍 + 现代科普语料）。"
                "用于：解释体质特征、给出饮食/起居/运动/情志调养建议、"
                "引用经典原文。检索链路自带自反思优化（结果不好会自动改写查询重试），"
                "所以直接用用户的自然语言问题作为 query 即可。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "检索问题，用自然语言完整表述"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_fact",
            "description": (
                "把用户透露的、**长期有效**的信息记入长期记忆，供以后所有对话使用。"
                "适合记：体质与健康状况、饮食偏好与忌口、生活习惯、所在地区等。"
                "不要记：寒暄、一次性问题、你自己给出的回答内容。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string",
                             "description": "一句话说清的事实，如『用户忌辛辣，作息偏晚』"},
                    "kind": {"type": "string",
                             "enum": ["health", "preference", "fact"],
                             "description": "类别：健康状况 / 偏好忌口 / 其他事实"},
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "begin_consultation",
            "description": (
                "开始体质辨识（共 27 题）。**仅在流程尚未开始时调用**"
                "（即当前阶段为「未开始」）。调用后请立刻用 next_question 取第一题。"
                "若流程已在进行中，本工具不会被提供。"
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reset_consultation",
            "description": (
                "清空此前进度、重新开始体质辨识。"
                "**只在用户明确要求『重新做一遍 / 重新开始辨识』时才调用**，"
                "其他任何情况都绝对不要调用——它会丢弃用户已经作答的全部内容。"
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# 各阶段允许调用的工具（用"工具可见性"约束行为空间）
#
# 这里藏着一个实测踩出来的关键设计：早期版本只有一个 restart_consultation，
# 结果模型**每一轮都调它**（因为它的描述里含"开始辨识"），把进度反复清零，
# 于是永远停在第一题。靠提示词反复强调"不要重复调用"并不可靠——
# 最终解法是把「开始」和「重来」拆成两个工具，并让 begin_consultation
# **只在 IDLE 阶段可见**：进入收集阶段后，模型在结构上就调不到它了。
# 经验：能靠"工具可见性 / 状态约束"杜绝的错误，不要指望提示词。
_STAGE_TOOLS = {
    # 未开始：只能"启动流程"或做普通问答/记忆；不给 next_question，
    # 避免用户只是随口问个养生问题时被莫名其妙拽进问卷
    S.IDLE:        ["begin_consultation", "search_knowledge", "remember_fact"],
    S.COLLECTING:  ["next_question", "record_answer", "reset_consultation",
                    "search_knowledge", "remember_fact"],
    S.SCORING:     ["judge_constitution", "search_knowledge", "remember_fact"],
    S.REPORTING:   ["search_knowledge", "remember_fact", "reset_consultation"],
    S.FOLLOWUP:    ["search_knowledge", "remember_fact", "reset_consultation"],
}


class ToolBox:
    """工具执行器：持有状态与运行时依赖，按名字执行工具。"""

    def __init__(self, state: "S.ConsultState", conv_id: int, memory=None):
        self.state = state
        self.conv_id = conv_id
        self.memory = memory
        self.sources: list[dict] = []        # 本轮检索到的引用来源（回传前端）
        self.trace: list[dict] = []          # 工具调用轨迹（前端展示"思考过程"）
        self.events: list[dict] = []         # 待推送前端的展示事件（由 agent 取走）
        self.last_result: "C.ConsultResult | None" = None
        self.records_this_turn = 0           # 本轮已记录的题数（护栏用）

    # 一轮对话最多记录一题的护栏：
    # 实测踩坑——模型在用户只说「我想测体质」时，会自己把前三题"答"完并连续
    # 调用 record_answer（编造用户答案）。提示词写了"用户回答后再记录"也拦不住。
    # 因此加确定性护栏：本轮已记录过一题，后续 record_answer 一律拒绝，
    # 并在返回里明确要求"先向用户提问、等回答"。工具层的硬约束比提示词可靠。
    MAX_RECORDS_PER_TURN = 1

    def drain_events(self) -> list[dict]:
        """取走并清空待推送的展示事件（agent 在每次工具调用后调用）。"""
        ev, self.events = self.events, []
        return ev

    # ---------- 工具可见性 ----------
    def schemas(self) -> list[dict]:
        """返回当前阶段允许的工具定义。"""
        allowed = set(_STAGE_TOOLS.get(self.state.stage, []))
        return [t for t in TOOL_SCHEMAS if t["function"]["name"] in allowed]

    # ---------- 执行 ----------
    def call(self, name: str, args: dict) -> str:
        """执行工具并返回给 LLM 的文本结果（JSON 字符串）。"""
        fn = getattr(self, f"_t_{name}", None)
        if fn is None:
            return json.dumps({"ok": False, "error": f"未知工具 {name}"},
                              ensure_ascii=False)
        try:
            out = fn(args or {})
        except Exception as e:                      # 工具异常要让模型知道，而不是崩掉
            out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        self.trace.append({"tool": name, "args": args, "ok": out.get("ok", True)})
        return json.dumps(out, ensure_ascii=False)

    # ---------- 各工具实现 ----------
    def _t_next_question(self, _args) -> dict:
        q = self.state.current_question()
        if q is None:
            return {"ok": False, "reason": "所有题目已答完",
                    "hint": "请调用 judge_constitution 得出判定结果"}
        self.state.mark_asked()          # 标记"这道题已问出口"（记录护栏依赖它）
        return {"ok": True, **q}

    def _t_record_answer(self, args) -> dict:
        if "score" not in args:
            return {"ok": False, "reason": "缺少 score 参数"}
        # 护栏 1：同一轮内不许"先问后记"（否则等于替用户编造答案）
        ok, why = self.state.can_record()
        if not ok:
            return {"ok": False, "reason": why}
        # 护栏 2：一轮最多记录一题（防模型连续自答多题）
        if self.records_this_turn >= self.MAX_RECORDS_PER_TURN:
            return {
                "ok": False,
                "reason": "本轮已经记录过一道题了，不能连续记录多题",
                "hint": "请用一句话把回应讲给用户（题目由界面卡片展示，"
                        "不要在正文里提问或复述题干），然后**停下来等待用户回答**。"
                        "用户没有明确作答时，绝对不要调用 record_answer。",
            }
        r = self.state.record(int(args["score"]))
        if not r.get("ok"):
            return r
        self.records_this_turn += 1
        nxt = self.state.current_question()
        return {**r, "next_question": nxt}

    def _t_judge_constitution(self, _args) -> dict:
        if self.state.answered < self.state.total:
            return {"ok": False,
                    "reason": f"还有题目未答完（{self.state.answered}/{self.state.total}）",
                    "hint": "请继续用 next_question 取题并 record_answer 记录"}
        result = self.state.judge()
        self.last_result = result
        # 判定结果归档到长期档案（下次开新会话仍记得体质）
        if self.memory:
            self.memory.save_constitution(
                result.primary, result.tendencies,
                {s.name: s.transform for s in result.scores})
        scores = [{"name": s.name, "code": s.code, "transform": s.transform,
                   "verdict": s.verdict, "primary": s.is_primary}
                  for s in result.scores]
        # 推给前端的判定卡片事件
        self.events.append({"type": "constitution", "report": self.state.last_result,
                            "primary": result.primary,
                            "tendencies": result.tendencies, "scores": scores})
        return {"ok": True, "report": C.format_report(result),
                "primary": result.primary, "primary_key": result.primary_key,
                "tendencies": result.tendencies, "scores": scores}

    def _t_search_knowledge(self, args) -> dict:
        query = (args.get("query") or "").strip()
        if not query:
            return {"ok": False, "reason": "query 不能为空"}
        from app.paths import STORE_DIR
        from app.rag import get_db
        from app.safety import strip_meta_info
        from app.selfrag import retrieve_with_reflection

        res = retrieve_with_reflection(get_db(), query)
        for d, score in res.hits:
            self.sources.append({
                "source": d.metadata.get("source", "?"),
                "chapter": d.metadata.get("chapter", "?"),
                "score": round(score, 3),
            })
        # 推给前端：自反思过程 + 本次命中来源（让"检索在想什么"可见）
        self.events.append({
            "type": "reflection", "reason": res.reason,
            "attempts": [{"round": a.round, "query": a.query,
                          "top_score": a.top_score, "accepted": a.accepted}
                         for a in res.attempts]})
        self.events.append({"type": "sources", "sources": list(self.sources)})
        return {
            "ok": True,
            "final_query": res.query,
            "low_confidence": res.low_confidence,
            "reflection": res.reason,
            "rounds": [{"round": a.round, "query": a.query,
                        "top_score": a.top_score, "accepted": a.accepted}
                       for a in res.attempts],
            "hits": [{"source": d.metadata.get("source", "?"),
                      "chapter": d.metadata.get("chapter", "?"),
                      "score": round(sc, 3),
                      "text": strip_meta_info(d.page_content)[:400]}
                     for d, sc in res.hits],
            "notice": ("检索置信度低，资料库可能没有直接相关内容。"
                       "请如实告知用户，如需补充常识性建议必须注明"
                       "『这是常识性建议，资料库中无直接依据』"
                       "（整篇回答最多注明一次，不要逐条标注）。"
                       if res.low_confidence else "检索质量正常，可放心依据资料回答。"),
        }

    def _t_remember_fact(self, args) -> dict:
        fact = (args.get("fact") or "").strip()
        if not fact:
            return {"ok": False, "reason": "fact 不能为空"}
        from app import storage
        from app.memory import _subject_of, wants_longterm
        # 作用域闸门（2026-09-16）：模型主动记忆**不等于**用户要求长期记住。
        # 只有用户这轮明说「记住／记一下／以后都」才升格为跨会话（global），
        # 否则只记在本会话——"新开的对话没有记忆"是产品语义，不能交给模型裁量。
        scope = (storage.MEM_GLOBAL if wants_longterm(self._last_user_text())
                 else storage.MEM_SESSION)
        r = storage.add_memory(fact, kind=args.get("kind", "fact"),
                               conv_id=self.conv_id,
                               subject=_subject_of(fact), scope=scope)
        return {"ok": True, "saved": fact, "scope": scope,
                "duplicated": r.get("duplicated", False),
                "note": ("已记为本会话内的信息（新开对话不会自动带上）；"
                         "若用户希望长期记住，请他明说一句「记住…」")
                        if scope == storage.MEM_SESSION else
                        "已记为长期记忆，用户以后主动问起时可用"}

    def _last_user_text(self) -> str:
        """上一轮用户原话（判"是否明说要长期记住"用）。

        取不到时返回空串 → 走保守分支（只记本会话）。
        """
        try:
            from app import storage as _s
            msgs = _s.get_messages(self.conv_id)
            for m in reversed(msgs):
                if m["role"] == "user":
                    return m["content"]
        except Exception:
            pass
        return ""

    def _t_begin_consultation(self, _args) -> dict:
        self.state.begin()
        return {"ok": True, "message": "体质辨识已开始，请立刻调用 next_question 取第一题",
                "stage": self.state.stage, "total": self.state.total}

    def _t_reset_consultation(self, _args) -> dict:
        self.state.begin()
        return {"ok": True, "message": "已清空此前作答并重新开始",
                "stage": self.state.stage, "total": self.state.total}
