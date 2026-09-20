# -*- coding: utf-8 -*-
"""流水线各节点的实现（图上的每个方框）。

节点函数的契约
--------------
    输入：完整共享状态 TCMState
    输出：**部分状态** dict（LangGraph 会按各字段的 reducer 合并进去）

两条纪律（照抄 LangGraph 的用法踩过坑）：
  · 节点**只返回自己产出的字段**，不要把整个 state 回抛——回抛会让
    "累积型字段"（evidence/trace）被整个覆盖；
  · **不要在节点里直接改 state 字典**（`state["x"] = 1`）就完事，
    必须通过返回值，否则框架不知道你改过。

五个角色的分工与"它凭什么单独存在"
-----------------------------------
    router    规则管安全（红旗短路）+ LLM 管意图（语义分流）——两者不可互换
    collector 唯一做量表采集的角色，产出是后续所有节点的唯一事实源
    diagnoser 承担"从数据到结论"的推理责任，**无证据不下证型**
    planner   把证型翻译成"今天能做的事"，最易越界，所以规矩最硬
    safety    唯一能否决的角色（独立于起草者，避免自证清白）
    editor    只管表达，不产生新事实
"""
from __future__ import annotations

import json
import time

from langchain_core.messages import HumanMessage, SystemMessage

from app import storage
from app.agent import constitution as C
from app.agent.state import ConsultState
from app.multiagent import checks
from app.multiagent import prompts as P
from app.multiagent.state import (ROUTE_FAST, ROUTE_NEED_CONSULT,
                                  ROUTE_PIPELINE, ROUTE_URGENT, TCMState,
                                  evidence_id, trace_entry)
from app.safety import (active_tags, annotate_origins, needs_medication_stop,
                        scan_profile, scan_text, strip_meta_info)

# 方案分项检索的参数。k_final 取 3 而不是 4：分项检索只求"每一类都有依据"，
# 条目再多也只是噪音，且能把 5 次检索的上下文总量压住。
PLANNER_K_FINAL = 3
PLANNER_MAX_ROUNDS = 2


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------
def _llm_json(prompt: str, system: str = "") -> dict:
    """调 LLM 并解析 JSON。

    容错策略沿用 app/eval.py 的经验：模型经常会**套一层 markdown 代码块**
    （```json ... ```），甚至前后加一句客套话。这里不要求它绝对听话，
    而是把第一个 `{` 到最后一个 `}` 抠出来解析——宽进严出。
    """
    from app.llm import get_llm

    msgs = ([SystemMessage(content=system)] if system else []) + \
           [HumanMessage(content=prompt)]
    try:
        raw = (get_llm().invoke(msgs).content or "").strip()
    except Exception:
        return {}
    if raw.startswith("```"):
        raw = raw.strip("`")
    i, j = raw.find("{"), raw.rfind("}")
    if i < 0 or j < 0:
        return {}
    try:
        return json.loads(raw[i:j + 1])
    except json.JSONDecodeError:
        return {}


def _llm_text(prompt: str, system: str = "") -> str:
    from app.llm import get_llm

    msgs = ([SystemMessage(content=system)] if system else []) + \
           [HumanMessage(content=prompt)]
    try:
        return (get_llm().invoke(msgs).content or "").strip()
    except Exception as e:
        return f"（生成失败：{type(e).__name__}: {e}）"


def _search(query: str, k_final: int = 4, max_rounds: int = 2) -> dict:
    """走现有的自反思两段式检索，返回给节点用的结构化结果。

    **刻意复用 `app/selfrag`** 而不是另写一条检索链路：多 Agent 的价值在于
    角色分工，不在于再造一个检索器；链路一旦分叉，两边行为就会漂移
    （这个坑本项目在"两段式检索抽成模块级函数"时已经踩过一次）。
    """
    from app.rag import get_db
    from app.selfrag import retrieve_with_reflection

    try:
        res = retrieve_with_reflection(get_db(), query,
                                       k_final=k_final, max_rounds=max_rounds)
    except Exception as e:
        # 检索失败（限流 / 连接 / 索引异常）不应炸掉整个专家节点。
        # 三个专家是**并行**跑的，一个检索异常若向上抛，会连带整条流水线失败——
        # 宁可让这个专家"没检索到资料"，由它的提示词规则（没有依据就不写）
        # 自然降级，也不要整条链路 500。
        return {"block": f"（检索失败：{type(e).__name__}: {e}）", "evidence": [],
                "top_score": 0.0, "low_confidence": True, "final_query": query,
                "reason": f"检索异常：{type(e).__name__}", "rounds": []}
    blocks, evid = [], []
    for doc, score in res.hits:
        src = doc.metadata.get("source", "?")
        ch = doc.metadata.get("chapter", "?")
        # 取用侧过滤"下回分解"式元信息（不重建索引，理由见 app/safety/scan.py）
        body = strip_meta_info(doc.page_content)
        blocks.append(f"[来源: {src} · {ch}]\n{body}")
        evid.append({"id": evidence_id(src, ch, body),
                     "source": src, "chapter": ch, "score": round(score, 3),
                     "text": body[:500], "query": query})
    top = round(res.hits[0][1], 3) if res.hits else 0.0
    return {
        "block": "\n\n---\n\n".join(blocks) or "（检索无结果）",
        "evidence": evid,
        "top_score": top,
        "low_confidence": res.low_confidence,
        "final_query": res.query,
        "reason": res.reason,
        "rounds": [{"round": a.round, "query": a.query,
                    "top_score": a.top_score, "accepted": a.accepted}
                   for a in res.attempts],
    }


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


# ---------------------------------------------------------------------------
# 接诊层：结构化档案 + 硬规则安全预扫（**在路由之前跑**）
# ---------------------------------------------------------------------------
def _build_intake(conv_id: int, text: str) -> dict:
    """跑一遍安全规则库与结构化档案，产出流水线全程要用的"用户情况"。

    这一段是「独立于 RAG 的安全规则库」在多 Agent 侧的落点：
    它不依赖检索结果、不依赖模型判断，纯硬编码规则 + 正则抽取，
    产出的命中项会作为**不可协商的约束**一路带到三个专家与安全审查节点。
    """
    from app import intake as IL

    try:
        prof0 = storage.get_profile(conv_id)
        hits = annotate_origins(scan_text(text), scan_profile(prof0), text)
        IL.update_from_message(text, conv_id)
        IL.record_herbs([h for h in hits if h.kind == "herb"], conv_id)
        prof = storage.get_profile(conv_id)
        tags = active_tags(text, profile=prof)
        from app import inquiry as IQ
        from app.safety import tiers as T
        # 追问清单：流水线只在"要调理方案"时进入，所以这一层必然是该追问的意图
        gap_list = IL.gaps(prof, text, require=IQ.require_slots(IQ.intent_of(text)),
                           conv_id=conv_id)
        fups = IQ.followups(text, conv_id=conv_id, profile=prof,
                            gaps=gap_list, hits=hits, tags=tags)
        from app import constraints as CN
        # 状态必须先于 apply_tiers 算出来（第五轮：状态直接决定档位）
        states = CN.detect(text, prof, conv_id)
        T.apply_tiers(hits, tags, gap_list, states=states)
        switch = IL.subject_switch(text, conv_id, prof)
        from app import framework as FW
        hits_dict = [h.to_dict() for h in hits]
        screen = IL.screening_keys(tags, prof, text, conv_id=conv_id)
        plan = FW.build(text, intent=IQ.intent_of(text), tags=tags,
                        hits=hits_dict, screening=screen, gaps=gap_list)
        return {
            "profile": prof,
            "hits": hits_dict,
            "tags": sorted(tags),
            "conflicts": IL.profile_conflicts(prof, text, conv_id=conv_id),
            "gaps": gap_list,
            "questions": IQ.to_event(fups),
            "followups": fups,
            "advisory": IL.advisory_gaps(tags, prof, text, conv_id=conv_id),
            "screening": screen,
            "stop": needs_medication_stop(hits, tags),
            "brief": IL.cross_turn_brief(conv_id, text, profile=prof),
            # 存**原始 dict**而不是 bool：下游 `not info.get(...)` 照样成立，
            # 但接口能把"哪一轴反了"讲清楚（界面得说得出依据）
            "subject_switch": switch,
            "states": states,
            "text": text,
            "plan": plan,
        }
    except Exception:
        # 兜底必须留痕（同 rag._safety_guard / agent._prescan）：
        # 2026-09-16 的 screening_keys 漏参 bug 三处调用点全被静默吞掉。
        import sys
        import traceback
        print("[multiagent-intake] 接诊层构建失败（本轮流式线无护栏）：",
              file=sys.stderr, flush=True)
        traceback.print_exc()
        return {"profile": {}, "hits": [], "tags": [], "conflicts": [],
                "gaps": [], "screening": [], "stop": False, "brief": "",
                "questions": [], "followups": []}


def _safety_guard(info: dict) -> str:
    """把接诊产物渲染成给专家 / 审查 / 主编的**强制约束块**。"""
    from app.safety import scripts as SG
    from app.safety.scan import build_block

    if not info:
        return ""
    hits = info.get("hits") or []
    tags = set(info.get("tags") or [])
    parts: list[str] = []
    if hits:
        # hits 已经是 dict（存进 state 后要可序列化），build_block 内部兼容 dict；
        # gaps 让分级引擎知道"哪些条件还没评估"；profile_ok 在疑似换人设时
        # 把档案命中降级为"待确认、不得作为依据"。
        parts.append(build_block(hits, tags, with_stop=info.get("stop", False),
                                 gaps=info.get("gaps"),
                                 profile_ok=not info.get("subject_switch")))
    if info.get("conflicts"):
        from app.intake import conflict_block
        parts.append(conflict_block(info["conflicts"]))
    if info.get("states"):
        # 状态类约束优先（P1-3）：插在跨轮材料之前，保证"先识别状态"的顺序
        from app import constraints as CN
        parts.append(CN.render_block(info["states"]))
    if info.get("brief"):
        parts.append(info["brief"])
    if info.get("screening"):
        parts.append(SG.screening_block(info["screening"]))
    if info.get("followups"):
        from app.inquiry import followup_block
        parts.append(followup_block(info["followups"]))
    # 判断框架 + 输出契约：三个专家与主编共用同一份"该覆盖什么"的清单，
    # 保证流水线产出的结构不被检索命中率决定（架构层问题五/六）
    if info.get("plan"):
        from app import framework as FW
        parts.append(FW.render(info["plan"]))
    from app import contract as CT
    parts.append(CT.render_contract())
    parts.append(SG.CITATION_JUDGMENT)
    return SG.combine(*parts)


# ---------------------------------------------------------------------------
# 问诊结果的读取（collector 与 router 都要用）
# ---------------------------------------------------------------------------
def load_consult(conv_id: int) -> dict | None:
    """从 SQLite 取出**已完成**的体质辨识结果，组装成 ConsultReport。

    「已完成」的判据用 `answered >= total`（题都答了），而不是
    `last_result` 非空——后者只是"算过一次分"，中途也能算。
    分值一律用 `constitution.compute()` **重算**而不是读缓存文本：
    重算是纯函数、零成本，且保证与量表定义永远一致。
    """
    try:
        st = ConsultState.load(conv_id)
    except Exception:
        return None
    if st.answered < st.total or st.total == 0:
        return None
    result = C.compute(st.answers)
    if not result.scores:
        return None
    return {
        "primary": result.primary,
        "primary_key": result.primary_key,
        "tendencies": list(result.tendencies),
        "careless": result.careless,
        "note": result.note,
        "report": C.format_report(result),
        "scores": [{"name": s.name, "code": s.code, "transform": s.transform,
                    "verdict": s.verdict, "primary": s.is_primary}
                   for s in result.scores],
        "chief_complaint": _chief_complaint(conv_id),
        "profile": _profile_text(conv_id),
    }


def _chief_complaint(conv_id: int) -> str:
    """主诉 = 最近几条用户消息的拼接（辨证需要的"当前状态"线索）。"""
    try:
        from app import storage
        msgs = storage.get_messages(conv_id)
    except Exception:
        return ""
    us = [m["content"].strip() for m in msgs if m.get("role") == "user"]
    parts = [t for t in us[-3:] if len(t) >= 4]
    return " / ".join(parts)[:300]


def _profile_text(conv_id: int | None = None) -> str:
    """本会话的档案摘要。档案已按会话隔离，必须带 conv_id，
    否则会读到"无会话"那份（旧全局档案），在多 Agent 路径上串档。"""
    try:
        from app.memory import MemoryManager
        return MemoryManager.profile_text(conv_id) or ""
    except Exception:
        return ""


def _consult_brief(c: dict) -> str:
    """给节点看的体质摘要（紧凑，避免把全部 9 维原始数据灌进提示词）。"""
    lines = [f"主体质：{c['primary']}"]
    if c.get("tendencies"):
        lines.append(f"兼夹 / 倾向：{'、'.join(c['tendencies'])}")
    top = sorted(c.get("scores") or [], key=lambda s: -s["transform"])[:5]
    lines.append("转化分（取前 5）：" + "；".join(
        f"{s['name']} {s['transform']:.1f}（{s['verdict']}）" for s in top))
    if c.get("careless"):
        lines.append("⚠️ 本次作答疑似无差别（分值几乎一致），结论可信度低。")
    if c.get("profile"):
        lines.append(f"用户背景：{c['profile']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ① 路由
# ---------------------------------------------------------------------------
# LLM 分流失败时的兜底关键词（宁可退回快路径，也不要误开流水线）
_PIPELINE_HINTS = ("调理方案", "完整方案", "系统的调理", "制定个计划", "做个计划",
                   "出一份方案", "调理计划", "整体调理", "系统调理",
                   "我该怎么调理", "怎么调理身体")


def router_node(state: TCMState) -> dict:
    t0 = time.perf_counter()
    text = state["user_input"]

    # ---- 第 0 层：接诊层扫描（档案 + 硬规则安全命中）----
    # 放在最前面：无论后面走哪条通道，产出的"用户情况"都要带给下游节点。
    # 注意它**不参与路由决策**——路由决策只看红旗与意图，安全命中是"附加约束"，
    # 混在一起会让"有高血压"这种常见情况被误判成需要转诊。
    info = _build_intake(state["conv_id"], text)

    # ---- 第一层：规则管安全（红旗短路，先于一切）----
    flags = checks.scan_red_flags(text)
    if flags:
        hit = "、".join(flags)
        return {
            "route": ROUTE_URGENT,
            "intake": info,
            "route_reason": f"命中需就医信号：{hit}",
            "urgent_reason": f"您提到的「{hit}」属于建议当面就诊的情况",
            "trace": [trace_entry("router", _ms(t0), route=ROUTE_URGENT,
                                  by="rule", flags=flags,
                                  safety_hits=len(info["hits"]))],
            "evidence": [],
        }

    consult = load_consult(state["conv_id"])
    ready = ("已有完整的体质辨识结果" if consult
             else "尚未完成体质辨识（没有可用判定结果）")

    # ---- 第二层：LLM 管意图（语义分流）----
    data = _llm_json(P.ROUTER_PROMPT.format(consult_ready=ready,
                                            user_input=text))
    route = data.get("route") if data.get("route") in (
        ROUTE_PIPELINE, ROUTE_FAST, ROUTE_URGENT) else ""
    reason = data.get("reason") or ""
    by = "llm"
    if not route:
        route = (ROUTE_PIPELINE if any(h in text for h in _PIPELINE_HINTS)
                 else ROUTE_FAST)
        reason = "分流模型未返回有效结果，按关键词兜底"
        by = "keyword"

    # ---- 第三层：前置条件校验（想做方案，但还没测体质）----
    if route == ROUTE_PIPELINE and not consult:
        route = ROUTE_NEED_CONSULT
        reason = "用户想要完整调理方案，但尚无体质判定结果"

    out: dict = {
        "route": route, "route_reason": reason, "intake": info,
        "trace": [trace_entry("router", _ms(t0), route=route, by=by,
                              reason=reason[:60],
                              safety_hits=len(info["hits"]),
                              conflicts=len(info["conflicts"]))],
        "evidence": [],
    }
    if route == ROUTE_PIPELINE:
        out["consult"] = consult
    return out


def route_after_router(state: TCMState) -> str:
    """条件边：只做决策，**不改状态**（LangGraph 不会应用路由函数里的写入）。"""
    r = state.get("route")
    if r == ROUTE_PIPELINE:
        return "collector"
    if r == ROUTE_URGENT:
        return "urgent"
    if r == ROUTE_NEED_CONSULT:
        return "need_consult"
    return "fast"


# ---------------------------------------------------------------------------
# ② 问诊（复用现有实现，不重写）
# ---------------------------------------------------------------------------
def collector_node(state: TCMState) -> dict:
    """问诊 Agent 的产物在流水线里**只读**——它已经由现有实现跑完了。"""
    t0 = time.perf_counter()
    consult = state.get("consult") or load_consult(state["conv_id"])
    if not consult:                        # 正常不会走到（router 已拦截）
        return {"degraded": True,
                "trace": [trace_entry("collector", _ms(t0), ok=False)]}
    return {
        "consult": consult,
        "trace": [trace_entry("collector", _ms(t0), ok=True,
                              primary=consult["primary"],
                              tendencies=len(consult["tendencies"]),
                              careless=consult["careless"])],
    }


# ---------------------------------------------------------------------------
# ③ 辨证
# ---------------------------------------------------------------------------
def diagnoser_node(state: TCMState) -> dict:
    t0 = time.perf_counter()
    consult = state["consult"]

    # 主诉 = 历史里最近几条用户消息 + **本轮这句话**。
    # 本轮必须算进去：用户往往是在这一句里才交代症状
    #（"我最近老是失眠，帮我出个调理方案"），只看历史会把这条最关键的线索漏掉。
    complaint = consult.get("chief_complaint") or ""
    this_turn = (state.get("user_input") or "").strip()
    if this_turn and len(this_turn) >= 6 and this_turn not in complaint:
        complaint = f"{complaint} / {this_turn}".strip(" /") if complaint else this_turn

    # 两条查询：一条按体质/证型术语（检索"病机"），一条按主诉（检索"症状"）。
    # 分开检而不是拼成一句，是因为自反思的 rerank 分数是按整体语义打的，
    # 把"阳虚质 证型 病机"和"最近手脚冰凉"拼在一起会互相稀释。
    tend = " ".join(consult.get("tendencies") or [])
    q_traits = f"{consult['primary']} {tend} 证型 病机 表现 调理".strip()
    queries = [q_traits] + ([complaint] if complaint and complaint != q_traits else [])

    blocks, evidence, rounds, lows, tops = [], [], [], [], []
    for q in queries:
        r = _search(q, k_final=4, max_rounds=2)
        blocks.append(r["block"])
        evidence += r["evidence"]
        rounds.append({"query": q, "rounds": r["rounds"],
                       "top_score": r["top_score"], "reason": r["reason"]})
        lows.append(r["low_confidence"])
        tops.append(r["top_score"])

    context = "\n\n---\n\n".join(blocks)
    data = _llm_json(P.DIAGNOSER_PROMPT.format(
        consult=_consult_brief(consult),
        complaint=complaint or "（用户未额外描述症状）",
        context=context[:12000]))

    syndromes = data.get("syndromes") or []
    insufficient = bool(data.get("insufficient")) or (
        all(lows) and not syndromes)        # 检索全都不达标且模型也没给出证型
    diagnosis = {
        "syndromes": syndromes,
        "differential": data.get("differential") or [],
        "evidence_refs": data.get("evidence_refs") or [],
        "confidence": data.get("confidence") or "低",
        "insufficient": insufficient,
        "note": data.get("note") or "",
        "retrieval_top": max(tops) if tops else 0.0,
        "retrieval_low": all(lows),
    }
    return {
        "diagnosis": diagnosis,
        "evidence": evidence,
        "trace": [trace_entry("diagnoser", _ms(t0),
                              syndromes=[s.get("name") for s in syndromes],
                              confidence=diagnosis["confidence"],
                              insufficient=insufficient,
                              retrieval_top=diagnosis["retrieval_top"],
                              searches=rounds)],
    }


def route_after_diagnoser(state: TCMState):
    """条件边：只做决策，不改状态。

    返回**列表**表示并行扇出——三个专科专家在同一个 superstep 里并发执行，
    之后汇入 planner（扇入只跑一次）。详见 graph.py 的注释。
    """
    d = state.get("diagnosis") or {}
    if d.get("insufficient"):
        return "urgent"
    # 信息不足 → 先去补齐，不要硬凑方案（P1-7 的强制追问）
    if _critical_gaps(state):
        return "need_more"
    return list(EXPERTS)


# ---------------------------------------------------------------------------
# ③b 信息不足分支
# ---------------------------------------------------------------------------
# 判定"能不能给方案级内容"的门槛。选舌象 / 寒热 / 二便三项：
# 它们是**区分寒热虚实的最小充分条件**——同样是"累 + 胖 + 湿"，
# 寒湿要温运、湿热要清利、脾虚湿困要健脾，方向互相矛盾，
# 缺了这三项就只能瞎猜。年龄性别与在服药物则关乎安全边界。
_CRITICAL_GAPS = ("tongue", "cold_heat", "stool_urine")


def _critical_gaps(state: TCMState) -> list[str]:
    info = state.get("intake") or {}
    gaps = list(info.get("gaps") or [])
    return [g for g in gaps if g in _CRITICAL_GAPS]


def need_more_node(state: TCMState) -> dict:
    """信息不足通道：**纯模板 + 已收集信息回显**，不给任何方剂级内容。"""
    from app.safety import scripts as SG

    t0 = time.perf_counter()
    info = state.get("intake") or {}
    consult = state.get("consult") or {}
    gaps = list(info.get("gaps") or [])
    crit = _critical_gaps(state)

    lines = ["## 还差一点信息，先别急着调\n",
             "要给出一份**针对你个人**的调理方案，有几项关键信息我还没拿到——"
             "这几项决定了到底是**温阳**还是**清利**，方向搞反了反而添乱，"
             "所以先不给你具体的药材和食养方。\n"]
    if info.get("brief"):
        lines.append("### 我已经知道的\n")
        for e in (info["brief"].splitlines()[1:] if "\n" in info["brief"] else []):
            lines.append(f"- {e.strip()}")
        lines.append("")
    lines.append("### 还需要你补充\n")
    for g in crit + [x for x in gaps if x not in crit][:2]:
        it = SG.GAP_ITEMS.get(g)
        if it:
            lines.append(f"**{it['label']}**：{it['ask']}\n")
    lines.append("### 这段先可以做的\n")
    lines.append("在补上这些信息之前，这几件事不会出错、也不用等：")
    lines.append("- 作息规律，尽量 23 点前入睡；")
    lines.append("- 饮食有节，少吃生冷黏腻，晚餐不过饱；")
    lines.append("- 避免久坐，每小时起身活动几分钟；")
    lines.append("- 情志平和，别把弦绷太紧。")
    if consult:
        lines.append(f"\n（你的体质判定是「{consult.get('primary', '?')}」，"
                     "这个结论仍然有效，只是不足以支撑具体方案。）")
    lines.append("\n>*把上面几项补给我，我就给你一份完整的、分层（本虚标实）的"
                 "食疗—经络—运动起居方案。*")

    return {"final_answer": "\n".join(lines),
            "trace": [trace_entry("need_more", _ms(t0), gaps=gaps)]}


# ---------------------------------------------------------------------------
# ④ 三个专科专家（多智能体分工：并行 fan-out）
# ---------------------------------------------------------------------------
# 为什么拆三个而不是一个大 planner
# --------------------------------
# 见 prompts.py 的说明：一个节点包办五类内容 → 提示词互相干扰、每类都浅；
# 拆开后每个专家只服务一个目标，**检索词也互不污染**
# （食疗查食材性味，经络查穴位主治，运动查导引功法），召回质量明显提升。
# 图上是 `diagnoser → {diet, meridian, movement} → planner` 的**并行扇出**，
# 三个专家各自独立检索与推理，最后由主控汇总。
EXPERTS = ("diet_expert", "meridian_expert", "movement_expert")
def _expert_context(state: TCMState, queries: list[str],
                    k_final: int = PLANNER_K_FINAL) -> tuple[str, list, dict]:
    """三个专家共用的检索脚手架：跑几条查询，拼上下文与证据。"""
    blocks, evidence, tops = [], [], {}
    for q in queries:
        r = _search(q, k_final=k_final, max_rounds=PLANNER_MAX_ROUNDS)
        blocks.append(r["block"])
        evidence += r["evidence"]
        tops[q[:20]] = r["top_score"]
    return "\n\n---\n\n".join(blocks)[:14000], evidence, tops


def _expert_inputs(state: TCMState) -> tuple[dict, dict, str, str]:
    consult = state["consult"]
    diagnosis = state.get("diagnosis") or {}
    guard = _safety_guard(state.get("intake") or {})
    lay = diagnosis.get("layering") or {}
    base = (f"{consult['primary']} {' '.join(consult.get('tendencies') or [])} "
            f"{lay.get('root_deficiency', '')} {lay.get('manifestation', '')}").strip()
    return consult, diagnosis, guard, base


def _expert_common_args(consult, diagnosis, guard) -> dict:
    return {"consult": _consult_brief(consult),
            "diagnosis": json.dumps(diagnosis, ensure_ascii=False)[:2500],
            "safety": guard or "（本轮无额外安全约束）"}


def diet_expert_node(state: TCMState) -> dict:
    t0 = time.perf_counter()
    consult, diagnosis, guard, base = _expert_inputs(state)
    complaint = (consult.get("chief_complaint") or "").strip()
    queries = [f"{base} 食材 性味 宜忌 食养 食疗方",
               f"{base} 忌口 不宜 慎食 少食"]
    if complaint:
        queries.append(f"{complaint} 饮食 宜忌 食疗")
    context, evidence, tops = _expert_context(state, queries)
    data = _llm_json(P.DIET_EXPERT_PROMPT.format(
        context=context, **_expert_common_args(consult, diagnosis, guard)))
    return {
        "diet_expert": data or {},
        "evidence": evidence,
        "trace": [trace_entry("diet_expert", _ms(t0),
                              items=len((data or {}).get("items") or []),
                              replacements=len((data or {}).get("replacements") or []),
                              retrieval_top=tops)],
    }


def meridian_expert_node(state: TCMState) -> dict:
    t0 = time.perf_counter()
    consult, diagnosis, guard, base = _expert_inputs(state)
    queries = [f"{base} 穴位 按摩 艾灸 定位 主治",
               f"{base} 脾俞 足三里 阴陵泉 丰隆 中脘"]
    context, evidence, tops = _expert_context(state, queries)
    data = _llm_json(P.MERIDIAN_EXPERT_PROMPT.format(
        context=context, **_expert_common_args(consult, diagnosis, guard)))
    return {
        "meridian_expert": data or {},
        "evidence": evidence,
        "trace": [trace_entry("meridian_expert", _ms(t0),
                              points=len((data or {}).get("points") or []),
                              retrieval_top=tops)],
    }


def movement_expert_node(state: TCMState) -> dict:
    t0 = time.perf_counter()
    consult, diagnosis, guard, base = _expert_inputs(state)
    queries = [f"{base} 导引 八段锦 运动 时长 强度",
               f"{base} 起居 作息 久坐 睡眠 调摄"]
    context, evidence, tops = _expert_context(state, queries)
    data = _llm_json(P.MOVEMENT_EXPERT_PROMPT.format(
        context=context, **_expert_common_args(consult, diagnosis, guard)))
    return {
        "movement_expert": data or {},
        "evidence": evidence,
        "trace": [trace_entry("movement_expert", _ms(t0),
                              exercise=len((data or {}).get("exercise") or []),
                              lifestyle=len((data or {}).get("lifestyle") or []),
                              retrieval_top=tops)],
    }


# ---------------------------------------------------------------------------
# ⑤ 主控汇总
# ---------------------------------------------------------------------------
# 首轮**不调模型**：三个专家已经把内容做完了，主控只做确定性汇总
# （合并 + 挂分层 + 收禁忌）。再让模型复述一遍是浪费，还会在复述中
# 丢掉专家的限定条件。
# 只有被安全审查打回重写时，才调模型来"带着修改意见重新落笔"——
# 那时要做的是**改写判断**，不是搬运，这正是模型该干的活。
def _merge_experts(state: TCMState) -> dict:
    consult = state["consult"]
    diagnosis = state.get("diagnosis") or {}
    de = state.get("diet_expert") or {}
    me = state.get("meridian_expert") or {}
    mo = state.get("movement_expert") or {}
    info = state.get("intake") or {}
    lay = dict(diagnosis.get("layering") or {})

    diet: list[dict] = []
    for it in (de.get("items") or []):
        diet.append({"item": it.get("item", ""), "why": it.get("why", ""),
                     "for": it.get("for", ""), "ref": it.get("ref") or {}})
    for rp in (de.get("replacements") or []):
        diet.append({
            "item": f"替换：把「{rp.get('from', '')}」换成「{rp.get('to', '')}」",
            "why": rp.get("reason", ""), "for": "标实", "ref": {}})

    exercise = list(mo.get("exercise") or [])
    lifestyle = list(mo.get("lifestyle") or [])
    acupoint: list[dict] = []
    for p in (me.get("points") or []):
        acupoint.append({
            "item": f"{p.get('point', '')}——{p.get('location', '')}；"
                    f"{p.get('how', '')}；{p.get('duration', '')}",
            "why": p.get("for", ""), "for": p.get("for", ""),
            "caution": p.get("caution", ""), "ref": p.get("ref") or {}})

    # 禁忌：三个专家的忌口 + 经络禁忌 + **硬规则命中的判读**（最后一项是硬底线）
    contra: list[dict] = []
    for src in (de.get("avoid") or [], mo.get("avoid") or []):
        for a in src:
            contra.append({"item": a.get("item", ""), "why": a.get("why", "")})
    for p in (me.get("points") or []):
        if (p.get("caution") or "").strip():
            contra.append({"item": f"{p.get('point', '')}：{p['caution']}",
                           "why": "穴位禁忌"})
    for h in (info.get("hits") or []):
        if h.get("level") in ("high", "medium"):
            contra.insert(0, {"item": f"【{h.get('name', '')}】{h.get('verdict', '')}",
                              "why": "系统安全规则命中（不可协商）"})

    confirm: list[str] = []
    for c in (info.get("conflicts") or []):
        confirm.append(f"档案核对：{c.get('detail', '')}")
    for h in (info.get("hits") or []):
        if h.get("taking") and h.get("level") == "high":
            confirm.append(f"暂停并咨询：{h.get('name', '')}——{h.get('verdict', '')}")
    for k in (info.get("screening") or []):
        from app.safety import scripts as SG
        it = SG.SCREENING.get(k)
        if it:
            confirm.append(f"{it['say']}（建议检查：" + "；".join(it["tests"]) + "）")

    return {
        "layering": lay,
        "summary": (f"本虚：{lay.get('root_deficiency', '（未分层）')}；"
                    f"标实：{lay.get('manifestation', '（未分层）')}。"
                    f"{lay.get('explain', '')}").strip(),
        "diet": diet, "lifestyle": lifestyle,
        "exercise": exercise, "acupoint": acupoint,
        "contra": contra, "confirm": confirm,
        "experts": {"diet": de.get("principle", ""),
                    "meridian": me.get("principle", ""),
                    "movement": mo.get("principle", "")},
    }


def planner_node(state: TCMState) -> dict:
    t0 = time.perf_counter()
    runs = int(state.get("planner_runs", 0)) + 1
    consult = state["consult"]
    diagnosis = state.get("diagnosis") or {}
    info = state.get("intake") or {}

    if runs == 1:
        plan = _merge_experts(state)
        by = "merge"
    else:
        safety = state.get("safety") or {}
        violations = [v.get("detail", "") for v in (safety.get("violations") or [])]
        experts = json.dumps({
            "diet": state.get("diet_expert") or {},
            "meridian": state.get("meridian_expert") or {},
            "movement": state.get("movement_expert") or {},
        }, ensure_ascii=False)[:8000]
        data = _llm_json(P.PLANNER_FIX_PROMPT.format(
            consult=_consult_brief(consult),
            diagnosis=json.dumps(diagnosis, ensure_ascii=False)[:2500],
            experts=experts,
            violations="；".join(v for v in violations if v) or "（未给出明细）",
            fix_hints="；".join(safety.get("fix_hints") or []) or "（未给出明细）",
            safety=_safety_guard(info) or "（无）",
        ))
        plan = data if data else _merge_experts(state)
        by = "llm-rewrite"

    return {
        "plan": plan,
        "planner_runs": runs,
        "trace": [trace_entry("planner", _ms(t0), run=runs, rewrite=(runs > 1),
                              by=by,
                              items={k: len(plan.get(k) or []) for k in
                                     ("diet", "lifestyle", "exercise",
                                      "acupoint", "contra")},
                              confirm=len(plan.get("confirm") or []))],
    }


# ---------------------------------------------------------------------------
# ⑤ 安全审查
# ---------------------------------------------------------------------------
_SEV_ORDER = {"low": 0, "medium": 1, "high": 2}

# 会阻止方案交付的违规类型（"方案本身不能用"）；红旗与完整性类不在其中。
_BLOCKING_TYPES = ("越界", "冲突", "分型越界")


def _render_hits_for_review(hits: list[dict]) -> str:
    """把硬规则命中渲染成审查员可读的清单。"""
    if not hits:
        return ""
    lines = []
    for h in hits:
        tag = "（用户在服用）" if h.get("taking") else ""
        lines.append(f"- [{h.get('level')}] {h.get('name')}{tag}："
                     f"{h.get('verdict') or h.get('why') or ''}")
    return "\n".join(lines)


def _plan_lines(plan: dict) -> str:
    """把方案摊成"一行一条"，供**句子级**规则扫描使用。

    ⚠️ 这是个真实踩出来的坑（自检用例 D 暴露）：规则层按句子判断
    「药材名是否出现在禁忌语境里」，而 `json.dumps(plan)` 里**没有任何中文句读**，
    整份方案会被 `_split_sentences` 当成**一句话**——于是方案里随便哪条
    `contra`（"忌生冷黏腻"）里的那个「忌」，都会把整篇方案豁免掉，
    真正的"推荐附子"就漏过去了。
    所以跨层冲突检查必须走这个逐条展开的文本，而不是原始 JSON。
    """
    lines: list[str] = []
    if plan.get("summary"):
        lines.append(str(plan["summary"]))
    lay = plan.get("layering") or {}
    if lay.get("explain"):
        lines.append(str(lay["explain"]))
    for key in ("diet", "lifestyle", "exercise", "acupoint", "contra"):
        for it in (plan.get(key) or []):
            if isinstance(it, dict):
                lines.append(f"{it.get('item', '')}。{it.get('why', '')}".strip("。"))
            else:
                lines.append(str(it))
    for it in (plan.get("confirm") or []):
        lines.append(str(it))
    for v in (plan.get("experts") or {}).values():
        if v:
            lines.append(str(v))
    return "\n".join(lines)


def _clip_lines(lines: list[str], budget: int) -> str:
    """按**整行**裁剪，绝不切在句子中间；有省略就显式标注。

    2026-09-16 真事故（审查员总说"条目被截断"的真相）：审查员看到的方案是
    `json.dumps(plan)[:6000]` 这类**硬切字符串**，而方案 JSON 实测 6.5k~8.5k，
    于是最后一条必然被切在半句上；审查员看到半个字段，就判「条目被截断、
    未给出完整判读」→ medium 违规 → pass=false → 打回重写 ×2 → 降级。
    **方案写得越全越容易中招**，与质量无关。
    修法：改用逐条渲染的行文本（每行一个完整条目）按行裁剪，
    省略时补一句"不代表缺失"，消除"看起来没写完"的假象。
    """
    out: list[str] = []
    used = 0
    for ln in lines:
        if out and used + len(ln) > budget:
            out.append(f"（…余下 {len(lines) - len(out)} 条因篇幅省略，"
                       f"**不代表内容缺失**）")
            break
        out.append(ln)
        used += len(ln) + 1
    return "\n".join(out)


def safety_node(state: TCMState) -> dict:
    t0 = time.perf_counter()
    diagnosis = state.get("diagnosis") or {}
    plan = state.get("plan") or {}
    info = state.get("intake") or {}
    hits = info.get("hits") or []
    plan_text = json.dumps(plan, ensure_ascii=False)
    all_text = plan_text + "\n" + json.dumps(diagnosis, ensure_ascii=False)

    # ---- 规则层：零歧义违规直接定罪，软线索只当提示 ----
    scan = checks.rule_scan(all_text)
    rule_violations = scan["violations"]
    hints = scan["hints"]
    # 与硬规则命中的药材冲突：用户正在服用的高危药材被当成"建议"写进了方案。
    # 必须用逐条展开的文本（见 _plan_lines 的说明），否则整份 JSON 会被当成一句话。
    conflict_violations = checks.safety_conflicts(_plan_lines(plan), hits)

    data = _llm_json(P.SAFETY_PROMPT.format(
        diagnosis=json.dumps(diagnosis, ensure_ascii=False)[:3000],
        plan=_clip_lines(_plan_lines(plan), 6000),
        rule_hits="；".join(hints) or "（无）",
        safety_hits=_render_hits_for_review(hits) or "（无）"))

    violations = (list(rule_violations)
                  + list(conflict_violations) + list(data.get("violations") or []))
    red_flags = list(data.get("red_flags") or [])
    fix_hints = list(data.get("fix_hints") or [])

    # 去重：按 (类型, 描述前 30 字)
    seen, uniq = set(), []
    for v in violations:
        if not isinstance(v, dict):
            continue
        key = (v.get("type", ""), str(v.get("detail", ""))[:30])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(v)

    # 安全侧的一致性约束：只要还有违规/红旗，pass 必须为 false——
    # **不采信模型返回的 pass**，这就是"规则层比模型更硬"的落点。
    passed = (not uniq) and (not red_flags)
    severity = data.get("severity") if data.get("severity") in _SEV_ORDER else "low"
    if uniq:
        severity = max([severity] + [v.get("severity", "low") for v in uniq
                                     if v.get("severity") in _SEV_ORDER],
                       key=lambda s: _SEV_ORDER[s])
    if red_flags:
        severity = "high"

    review = {"pass": passed, "violations": uniq, "red_flags": red_flags,
              "fix_hints": fix_hints, "severity": severity,
              "rule_hits": len(rule_violations),
              "safety_conflicts": len(conflict_violations),
              "hard_hits": len(hits)}
    return {
        "safety": review,
        "trace": [trace_entry("safety", _ms(t0), passed=passed,
                              violations=len(uniq),
                              rule_violations=len(rule_violations),
                              safety_conflicts=len(conflict_violations),
                              red_flags=red_flags, severity=severity,
                              hints=len(hints))],
    }


def route_after_safety(state: TCMState) -> str:
    """否决回环的出口。

    注意：这里**不能** `state["rewrite_count"] += 1`（设计稿里那样写是错的）——
    条件边的写入会被 LangGraph 丢弃，照抄会导致回环永远只走一次。
    重写计数改由 planner 节点自己维护（`planner_runs`），本函数只读。
    """
    r = state.get("safety") or {}
    if r.get("red_flags"):
        # 2026-09-15 验收缺陷：红旗直接转 urgent 会把三个专家跑出来的
        # 方案**整个丢掉**——慢病用户（高血压本身就是红旗词）永远拿不到
        # 方案，只剩一句"去就医"。正确语义：方案照常交付，
        # 就医建议由主编**置顶写进**「需要先确认的事」（见 EDITOR_PROMPT）。
        # 只有路由层的红旗（便血/胸痛这类"养生咨询不该接"的输入）才走 urgent。
        return "editor"
    if r.get("pass"):
        return "editor"
    runs = int(state.get("planner_runs", 1))
    if runs - 1 < 2:                      # 已重写次数 < 上限（MAX_REWRITE）
        return "planner"
    return "editor"                       # 重写用尽 → 降级交付


# ---------------------------------------------------------------------------
# ⑥ 主编
# ---------------------------------------------------------------------------
def _judgment_fallback(state: TCMState) -> str:
    """交付前的兜底判读：由安全规则直接生成，不经过模型。"""
    hits = (state.get("intake") or {}).get("hits") or []
    lines = ["### 所以对你而言：能做 / 不能做 / 需先确认", ""]
    if hits:
        for h in hits:
            lines.append(f"- **{h.get('name', '')}**："
                         f"{h.get('verdict') or '需先确认（请咨询中医师或药师）'}")
    else:
        lines.append("- 上面引用的资料讲的是**通用规律**；具体能不能用在你身上，"
                     "还要看你自己的寒热虚实。如果你打算长期服用任何药材或食疗方，"
                     "**请先咨询中医师或药师**再定。")
    return "\n".join(lines)


def _blocking_violations(review: dict) -> list[dict]:
    """会**阻止方案交付**的违规：方案本身越界/自相矛盾。

    红旗（要提醒就医）与完整性类（只引不判 / 证据不足）**不算**——
    前者由安全块与就医提示覆盖，后者由判读兜底与契约自检补齐；
    用它们否决整份方案，等于"用一句话的风格换掉用户的方案"。
    """
    return [v for v in (review.get("violations") or [])
            if isinstance(v, dict)
            and v.get("type") in _BLOCKING_TYPES
            and v.get("severity") in ("high", "medium")]


def editor_node(state: TCMState) -> dict:
    t0 = time.perf_counter()
    consult = state.get("consult") or {}
    diagnosis = state.get("diagnosis") or {}
    plan = state.get("plan") or {}
    safety = state.get("safety") or {}

    # 降级判据（2026-09-16 语义对齐）：只有"方案本身不能用"的违规才值得降级。
    # route_after_safety 对"只有红旗"的情形是**转 editor 正常交付**（注释写明：
    # 红旗不该把三专家跑出来的方案整个丢掉，就医建议由主编置顶写进）。若这里
    # 只看 `pass`，就会把路由层已判定"可交付"的方案套成降级模板——实测 RUN 7：
    # 红旗 + 只引不判 → 方案被丢，用户只看到一段保守文案（穴位/运动全没了）。
    # 红旗 = 要提醒就医；只引不判 / 证据不足 = 完整性，两者都由提示词里的安全块、
    # `_judgment_fallback` 判读兜底与 `contract.enforce` 契约自检覆盖。
    blocking = _blocking_violations(safety)
    if not blocking:
        text = _llm_text(P.EDITOR_PROMPT.format(
            consult=_consult_brief(consult),
            diagnosis=json.dumps(diagnosis, ensure_ascii=False)[:3000],
            plan=_clip_lines(_plan_lines(plan), 8000),
            safety=json.dumps(safety, ensure_ascii=False)[:1500],
            guard=_safety_guard(state.get("intake") or {}) or "（无）"))
        degraded = False
    else:
        text = _llm_text(P.EDITOR_DEGRADED_PROMPT.format(
            consult=_consult_brief(consult),
            diagnosis=json.dumps(diagnosis, ensure_ascii=False)[:2000],
            plan=_clip_lines(_plan_lines(plan), 3000),
            safety=json.dumps(safety, ensure_ascii=False)[:1500],
            guard=_safety_guard(state.get("intake") or {}) or "（无）"))
        degraded = True

    # ---- 最后一道兜底：引用必带判读 ----
    # 提示词里已经要求"引用必带判读"，但模型仍可能只引不判——用户实测的
    # 原话就是「资料里说 X，但资料库里没有针对你个人情况的方案」，把原文
    # 摆出来就结束。与其指望提示词，不如在**交付前程序化检查并补齐**：
    # 判定"有引用却没有判读"时，追加一段直接由安全规则生成的判读清单。
    appended = ""
    if checks.has_citation(text) and not checks.has_judgment(text):
        appended = _judgment_fallback(state)
        text = text.rstrip() + "\n\n" + appended

    # ---- 输出契约自检（六模块 + 档位一致性 + 药名外泄守卫）----
    # 与问答/问诊**同一段代码**：三条链路共用一套交付前检查，
    # 否则会出现"问答路径补了风险管理、多 Agent 路径没补"这种不一致。
    mods: list[str] = []
    try:
        from app.contract import enforce
        info = state.get("intake") or {}
        gctx = {"hits": info.get("hits") or [],
                "tags": info.get("tags") or [],
                "screening": info.get("screening") or [],
                "plan": info.get("plan"),
                "gaps": info.get("gaps") or [],
                "states": info.get("states") or [],
                "question": info.get("text") or "",
                "profile": info.get("profile") or {},
                "stop": info.get("stop", False)}
        text, mods = enforce(text, gctx)
    except Exception:
        import sys
        import traceback
        print("[editor] 输出契约自检失败（本轮按原文交付）：",
              file=sys.stderr, flush=True)
        traceback.print_exc()

    return {
        "final_answer": text,
        "degraded": degraded,
        "trace": [trace_entry("editor", _ms(t0), chars=len(text),
                              degraded=degraded,
                              judgment_appended=bool(appended),
                              contract=mods)],
    }


# ---------------------------------------------------------------------------
# ⑦ 三个"短通道"节点（不进调理流程）
# ---------------------------------------------------------------------------
def urgent_node(state: TCMState) -> dict:
    """建议就医通道：**纯模板，不经模型**。

    这条路径承载的是安全结论，交给 LLM 润色等于把最后一道防线交给概率。
    宁可语气朴素一点。
    """
    t0 = time.perf_counter()
    reason = state.get("urgent_reason") or ""
    diag = state.get("diagnosis") or {}
    if not reason and state.get("route") == ROUTE_URGENT:
        reason = state.get("route_reason") or "您描述的情况需要当面判断"
    if not reason and diag.get("insufficient"):
        reason = "现有资料不足以支撑针对性的辨证结论"

    text = (
        "## 建议先就医面诊\n\n"
        f"本次没有为您生成调理方案，原因：**{reason}**。\n\n"
        "这不是推脱——养生调理适合的是**没有明确疾病征象的日常状态**；"
        "一旦出现了需要当面判断的信号，隔着一层文字给出的任何建议都可能"
        "耽误事。\n\n"
        "### 建议您\n"
        "1. 到正规医疗机构的**中医科或相应专科**就诊，由医师面诊；\n"
        "2. 就诊时把最近的变化说清楚：什么时候开始的、有没有加重、"
        "伴随哪些不舒服、正在吃什么药；\n"
        "3. 在明确诊断之前，先不要自行进补或长期服用任何药材。\n\n"
        "### 在此之前可以做的\n"
        "保证睡眠、饮食清淡规律、避免过度劳累——这些不会出错，"
        "也不会掩盖病情。\n\n"
        ">*体质辨识与养生咨询不能替代医疗诊断，本系统不做疾病诊断、不开具处方。*"
    )
    return {"final_answer": text,
            "trace": [trace_entry("urgent", _ms(t0), reason=reason[:60])]}


def need_consult_node(state: TCMState) -> dict:
    """想要方案但还没做过辨识：引导先完成量表，而不是硬凑一份方案。"""
    t0 = time.perf_counter()
    text = (
        "## 先做一次体质辨识吧\n\n"
        "要给出**针对您个人**的调理方案，我需要先知道您的体质底子——"
        "同样是怕冷，阳虚和气郁的调法完全不同，瞎猜比不猜更糟。\n\n"
        "体质辨识是一份 27 题的问卷，大约 5 分钟。做完之后我就能"
        "结合您的体质给出系统的食疗、起居、穴位建议。\n\n"
        "**您可以这样开始**：在这里告诉我「我想做体质辨识」，我会一道一道问您。\n\n"
        "如果您只是随口问问某个养生问题（比如「阳虚体质有什么表现」），"
        "也可以直接问，我会照常回答。"
    )
    return {"final_answer": text,
            "trace": [trace_entry("need_consult", _ms(t0))]}


def fast_node(state: TCMState) -> dict:
    """快路径：**交给现有单 Agent**，不在这里重复实现一遍 RAG。

    Web 层日后可以在进图之前就用 router 的结果分流（省一次图调度），
    但节点本身保持自洽——CLI、回退路径、评估脚本都能直接跑。
    """
    t0 = time.perf_counter()
    from app.rag import RAGSession

    try:
        ans = RAGSession(conv_id=state["conv_id"]).ask(state["user_input"])
        return {
            "final_answer": ans.answer,
            "trace": [trace_entry("fast", _ms(t0), delegated="RAGSession",
                                  sources=len(ans.sources),
                                  standalone=ans.standalone[:60])],
        }
    except Exception as e:
        return {"final_answer": f"（回答失败：{type(e).__name__}: {e}）",
                "trace": [trace_entry("fast", _ms(t0), ok=False)]}
