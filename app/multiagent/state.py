# -*- coding: utf-8 -*-
"""多 Agent 共享状态（LangGraph StateGraph 的"血液"）。

节点之间只通过共享 State 通信
------------------------------
LangGraph 的节点函数返回一个 dict，框架把它**合并**进全局状态。合并规则由
每个字段声明的 reducer 决定：

    字段: list[dict]                          → 覆盖（后写的赢，默认行为）
    字段: Annotated[list[dict], merge_evidence] → 追加且去重

本文件里两个累积型字段各用一种，理由不同：

  · `trace`     —— 每个节点写一条、**允许重复**（planner 在否决回环里会跑
                   多次，每次都要留痕），所以用 `operator.add` 纯追加；
  · `evidence`  —— 全程检索命中的块**集中成一个池**、按块指纹去重。这样
                   最终引用列表天然不重复，且"引用是否真实存在"可以程序化
                   校验（引用必须能在池里找到）——这是 Step 2 强校验的地基。

两个实现时踩到/避开的坑
-----------------------
① **条件边不能改状态。** `add_conditional_edges` 的路由函数只做决策，返回值
   是"下一个节点的名字"；函数里对 state 的任何写入都会被丢弃。所以"重写次数
   +1"必须放在 planner 节点内部——这在设计稿里写成了 `after_safety` 里
   `state["rewrite_count"] += 1`，直接照抄会导致**回环永远只走一次**。
② **累积字段如果没声明 reducer，会被后一个节点整个覆盖掉**，表现为
   "辨证检索到了证据、到方案节点一看没了"。这类 bug 不报错，只是结果变差，
   最难查。

字段命名约定
------------
`planner_runs` = 方案节点**已执行次数**（首次执行 = 1）。
所以「已重写次数」= planner_runs - 1，界面与日志都用后者说话，
避免"第 0 次重写"这种别扭表述。
"""
from __future__ import annotations

import hashlib
import operator
import time
from typing import Annotated, TypedDict

# ---------------------------------------------------------------------------
# 路由取值
# ---------------------------------------------------------------------------
ROUTE_FAST = "fast"                  # 闲聊 / 单点知识问答 → 现有单 Agent 快路径
ROUTE_PIPELINE = "pipeline"          # 完整体质辨识后的调理方案 → 四角色流水线
ROUTE_URGENT = "urgent"              # 红旗症状 → 建议就医（不进调理流程）
ROUTE_NEED_CONSULT = "need_consult"  # 想要方案但还没做完整辨识 → 引导先去测

# 方案最多被安全审查打回重写的次数（超过则降级：保守建议 + 建议就医）
MAX_REWRITE = 2


# ---------------------------------------------------------------------------
# 累积型字段的 reducer
# ---------------------------------------------------------------------------
def merge_evidence(left: list[dict] | None, right: list[dict] | None) -> list[dict]:
    """证据池去重合并：同一块只在池里存一份（保留先到的那份）。

    去重键用**内容指纹**而不是 (source, chapter)：古籍同一篇章里相邻块
    经常被两个不同查询分别命中，指纹相同才真的重复。
    """
    merged: dict[str, dict] = {}
    for item in (left or []):
        merged.setdefault(item["id"], item)
    for item in (right or []):
        merged.setdefault(item["id"], item)
    return list(merged.values())


def evidence_id(source: str, chapter: str, text: str) -> str:
    """块的稳定指纹（16 位十六进制）。

    只取正文前 300 字参与哈希：切块大小变了也不会把同一个块算成两块，
    同时又足够区分同篇章内的不同块。
    """
    raw = f"{source}|{chapter}|{text[:300]}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 共享状态
# ---------------------------------------------------------------------------
class TCMState(TypedDict, total=False):
    """流水线的共享状态。total=False 表示允许中途缺字段（节点产物按需出现）。"""

    # ---- 输入 ----
    conv_id: int
    user_input: str

    # ---- 各节点产物（None / 缺键 = 尚未执行）----
    consult: dict | None      # ConsultReport：9 维分 / 主体质 / 兼夹 / 主诉 / 生活习惯
    intake: dict | None       # Intake：结构化档案 + 硬规则安全命中 + 档案冲突 + 信息缺口
    diagnosis: dict | None    # Diagnosis：证型候选 / 本虚标实分层 / 依据 / 置信度
    # 三个专科专家的产物（多智能体分工，见 nodes.py 的 * _expert_node）
    diet_expert: dict | None      # 食疗：食材性味宜忌 + 替代方案
    meridian_expert: dict | None  # 经络：穴位 / 定位 / 手法 / 时长 / 禁忌
    movement_expert: dict | None  # 运动起居：导引式子 / 时长强度 / 久坐对策 / 作息
    plan: dict | None         # Plan：主控汇总（分层结论 + 四类建议 + 禁忌）
    safety: dict | None       # SafetyReview：pass / violations / red_flags / fix_hints

    # ---- 控制流 ----
    route: str                # ROUTE_* 之一
    route_reason: str         # 路由理由（给前端"思考过程"看）
    planner_runs: int         # 方案节点已执行次数（首次 = 1）
    degraded: bool            # 是否已降级（审查不通过且重写次数用尽）
    urgent_reason: str        # 走建议就医通道的原因

    # ---- 可观测（累积型）----
    evidence: Annotated[list[dict], merge_evidence]
    trace: Annotated[list[dict], operator.add]

    # ---- 出口 ----
    final_answer: str


# ---------------------------------------------------------------------------
# 构造与查看
# ---------------------------------------------------------------------------
def new_state(conv_id: int, user_input: str) -> TCMState:
    """构造一次流水线的初始状态。"""
    return TCMState(
        conv_id=conv_id,
        user_input=user_input,
        consult=None, intake=None, diagnosis=None,
        diet_expert=None, meridian_expert=None, movement_expert=None,
        plan=None, safety=None,
        route="", route_reason="", planner_runs=0, degraded=False,
        urgent_reason="",
        evidence=[], trace=[],
        final_answer="",
    )


def trace_entry(node: str, ms: int, **detail) -> dict:
    """一条轨迹记录：谁、花了多久、做了什么。

    前端"思考过程"面板与链路评估都吃这个结构，所以字段名固定。
    """
    return {"node": node, "ms": ms, "at": time.time(), **detail}


def rewrite_count(state: TCMState) -> int:
    """已发生的重写次数（planner 跑第 1 次不算重写）。"""
    return max(0, int(state.get("planner_runs", 0)) - 1)


def render(state: TCMState) -> str:
    """把状态渲染成一段可读文本（CLI 调试与日志用）。"""
    lines = ["=" * 72,
             f"路由：{state.get('route')} —— {state.get('route_reason', '')}"]
    if state.get("consult"):
        c = state["consult"]
        lines.append(f"问诊：主体质={c.get('primary')}  兼夹={c.get('tendencies')}  "
                     f"主诉={c.get('chief_complaint', '')[:30]}")
    if state.get("intake"):
        it = state["intake"]
        prof = it.get("profile") or {}
        lines.append(f"接诊：档案={list(prof.keys())}  "
                     f"硬规则命中={len(it.get('hits') or [])}  "
                     f"档案冲突={len(it.get('conflicts') or [])}  "
                     f"信息缺口={it.get('gaps') or []}")
    if state.get("diagnosis"):
        d = state["diagnosis"]
        brief = "、".join(f"{s.get('name', '?')}({s.get('likelihood', '-')})"
                         for s in (d.get("syndromes") or [])) or "（未给出证型）"
        lines.append(f"辨证：{brief}  置信度={d.get('confidence')}  "
                     f"资料不足={d.get('insufficient')}")
        lay = d.get("layering") or {}
        if lay:
            lines.append(f"  分层：本虚={lay.get('root_deficiency', '?')} / "
                         f"标实={lay.get('manifestation', '?')}")
    for key, label in (("diet_expert", "食疗"), ("meridian_expert", "经络"),
                       ("movement_expert", "运动起居")):
        ex = state.get(key)
        if ex:
            n = sum(len(v) for k, v in ex.items() if isinstance(v, list))
            lines.append(f"{label}专家：产出 {n} 条")
    if state.get("plan"):
        p = state["plan"]
        counts = {k: len(p.get(k) or []) for k in
                  ("diet", "lifestyle", "exercise", "acupoint", "contra")}
        lines.append(f"方案：{counts}")
    if state.get("safety"):
        s = state["safety"]
        lines.append(f"审查：pass={s.get('pass')}  违规={len(s.get('violations') or [])}  "
                     f"红旗={len(s.get('red_flags') or [])}  严重度={s.get('severity')}")
    lines.append(f"证据池：{len(state.get('evidence') or [])} 块")
    lines.append("")
    lines.append("轨迹：")
    for t in (state.get("trace") or []):
        extra = {k: v for k, v in t.items() if k not in ("node", "ms", "at")}
        lines.append(f"  {t['node']:<12} {t['ms']:>6} ms   {extra}")
    lines.append("=" * 72)
    return "\n".join(lines)
