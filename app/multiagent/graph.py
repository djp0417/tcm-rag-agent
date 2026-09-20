# -*- coding: utf-8 -*-
"""LangGraph 装配：图结构、条件边、否决回环。

为什么用 LangGraph 而不是手写编排
--------------------------------
本流水线有一条**带条件的分支 + 一条回环**（safety 否决 → 回 planner）。
手写就是一个 while + 若干 if，能跑，但会丢掉三样东西：
  ① 状态快照 / 断点续跑（长链路最容易在中途挂掉）；
  ② 图可视化（`--graph` 可以当场把 mermaid 打出来，面试演示很直观）；
  ③ 条件边的声明式表达——**流程长什么样，代码就长什么样**，而不是藏在控制流里。

图结构（v2：加了三个专科专家的并行扇出）
----------------------------------------
    START → router ─┬─ "fast"         → fast         → END
                    ├─ "urgent"       → urgent       → END
                    ├─ "need_consult" → need_consult → END
                    └─ "pipeline"     → collector → diagnoser
                                                     ├─ "urgent"    → urgent → END
                                                     ├─ "need_more" → need_more → END
                                                     └─ 扇出 ┬→ diet_expert     ─┐
                                                             ├→ meridian_expert ─┼→ planner
                                                             └→ movement_expert ─┘
                                                          planner → safety
                                          safety ─┬─ "editor"（通过）      → editor → END
                                                  ├─ "urgent"（红旗）      → urgent → END
                                                  └─ "planner"（打回重写） ↗（最多 2 次，用尽则 editor 降级）

并行扇出怎么表达（LangGraph 的一个细节）
----------------------------------------
条件边**允许返回一个列表**——返回 `["diet_expert", "meridian_expert",
"movement_expert"]` 即表示"这三个节点都跑"，框架会把它们放进同一个
superstep 并发执行；随后三个节点都指向 planner，planner 会在三者
**全部完成后只执行一次**（扇入/join 语义）。

为什么不能给 diagnoser 同时挂"条件边 + 三条静态边"：静态边是无条件触发的，
走 urgent / need_more 分支时三个专家照样会跑一遍——白白多花三次 LLM。

单一事实源
----------
**业务事实（消息 / 体质判定 / 档案 / 长期记忆）仍以 `store/chat.db` 为唯一真相。**
LangGraph 的 checkpointer 只保存"本次流水线的中间态"，默认不启用；
即使启用也建议独立表，**绝不让两处都能改同一个业务字段**
（否则会出现"库里显示阳虚、流水线里是气郁"这种查不出来的 bug）。
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.multiagent import nodes as N
from app.multiagent.state import TCMState, new_state

_COMPILED = None

# 三个专科专家（并行扇出）——名单定义在 nodes.py，避免两处漂移
_EXPERTS = N.EXPERTS


def build_graph(checkpointer=None):
    """装配并编译图。`checkpointer=None` 表示不持久化中间态。"""
    g = StateGraph(TCMState)

    g.add_node("router", N.router_node)
    g.add_node("collector", N.collector_node)
    g.add_node("diagnoser", N.diagnoser_node)
    g.add_node("need_more", N.need_more_node)
    g.add_node("diet_expert", N.diet_expert_node)
    g.add_node("meridian_expert", N.meridian_expert_node)
    g.add_node("movement_expert", N.movement_expert_node)
    g.add_node("planner", N.planner_node)
    g.add_node("safety", N.safety_node)
    g.add_node("editor", N.editor_node)
    g.add_node("fast", N.fast_node)
    g.add_node("urgent", N.urgent_node)
    g.add_node("need_consult", N.need_consult_node)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", N.route_after_router, {
        "collector": "collector",
        "fast": "fast",
        "urgent": "urgent",
        "need_consult": "need_consult",
    })

    g.add_edge("collector", "diagnoser")
    g.add_conditional_edges("diagnoser", N.route_after_diagnoser, {
        "urgent": "urgent",
        "need_more": "need_more",
        # 返回列表时即并行扇出（三条边同 superstep）
        "diet_expert": "diet_expert",
        "meridian_expert": "meridian_expert",
        "movement_expert": "movement_expert",
    })

    for name in _EXPERTS:                 # 扇入：三个专家都指向 planner
        g.add_edge(name, "planner")

    g.add_edge("planner", "safety")
    g.add_conditional_edges("safety", N.route_after_safety, {
        "editor": "editor",
        "planner": "planner",       # ← 否决回环
        "urgent": "urgent",
    })

    g.add_edge("editor", END)
    g.add_edge("fast", END)
    g.add_edge("urgent", END)
    g.add_edge("need_consult", END)
    g.add_edge("need_more", END)

    return g.compile(checkpointer=checkpointer)


def get_graph():
    """模块级单例：编译一次反复用（编译有开销，且便于复用同一实例）。"""
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = build_graph()
    return _COMPILED


def run_pipeline(conv_id: int, user_input: str, on_step=None,
                 checkpointer=None) -> dict:
    """跑完一次流水线，返回最终状态。

    Args:
        on_step: 回调 (node_name, delta) —— 每完成一个节点调一次，
                 用于 CLI 实时打印进度（Web 层可用来推 SSE 事件）。
    """
    app = build_graph(checkpointer) if checkpointer is not None else get_graph()
    init = new_state(conv_id, user_input)
    final: dict = dict(init)
    try:
        for mode, payload in app.stream(init, stream_mode=["updates", "values"]):
            if mode == "updates":
                for node, delta in (payload or {}).items():
                    if on_step:
                        on_step(node, delta or {})
            else:
                final = payload
    except (ValueError, TypeError):
        # 老版本 langgraph 不支持 stream_mode 传列表 → 退回单模式
        final = app.invoke(init)
    return final


def mermaid() -> str:
    """导出 mermaid 流程图（CLI `--graph` 用；也可贴进 README / 面试材料）。"""
    try:
        return get_graph().get_graph().draw_mermaid()
    except Exception as e:                    # 绘图依赖缺失时不影响主流程
        return f"（无法生成流程图：{type(e).__name__}: {e}）"
