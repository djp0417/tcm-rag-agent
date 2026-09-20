# -*- coding: utf-8 -*-
"""MCP 工具实现 —— **纯函数层**（刻意不 import `mcp`，所以能脱离协议单独测试）。

三条设计原则（都是这个项目踩出来的）：

1. **薄包装，不复制业务逻辑**。这里只做「参数收口 → 调用现有确定性链路 → 序列化成
   JSON」：检索走 `rag.RAGSession._retrieve`、安全判读走 `safety.scan` +
   `safety.tiers.apply_tiers`、体质走 `agent.constitution.compute`。
   业务规则永远只有一处定义，MCP 层不许长出第二份。

2. **只读**。所有工具都不写库 —— 刻意不调用 `intake.update_from_message` /
   `record_herbs` / `append_timeline` / `record_safety_hits` 这些有副作用的函数。
   这样即使客户端模型误调用，也不会污染真实会话与档案数据。

3. **出口守卫**。返回前统一过 `constraints.scrub`，把**调用方没提过**的药食名按类别
   概化（细节见 `_guard_payload`）。新入口绝不能成为泄漏守卫的绕道 ——
   第五轮那次"内部规则原文外泄"就是因为内部话术被直接填进了正文。
"""
from __future__ import annotations

import json
from typing import Any

from app.mcp import _boot

_boot.ensure_project_root()

__all__ = ["tcm_search", "tcm_safety_check", "tcm_constitution", "tcm_intake_gaps"]


# ---------------------------------------------------------------------------
# 公共设施
# ---------------------------------------------------------------------------
def _jsonable(obj: Any) -> Any:
    """把任意对象转成可 JSON 序列化的形式（先原样试，不行再降级成 str）。"""
    try:
        json.dumps(obj, ensure_ascii=False)
        return obj
    except (TypeError, ValueError):
        if isinstance(obj, dict):
            return {str(k): _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [_jsonable(v) for v in obj]
        return str(obj)


def _guard_payload(payload: dict, allowed) -> dict:
    """工具出口的泄漏守卫：把调用方**没提过**的药食名按类别概化。

    反直觉但必须遵守的一点：这里**只回传"是否概化过"与计数，不回传被替换掉的名字**
    —— 否则守卫本身又成了一条泄漏渠道（"我帮你把 A、B、C 隐去了"等于把 A、B、C 说了）。

    放行集合 `allowed` 来自 `constraints.allowed_names`：调用方传进来的 items、
    states_text、profile，以及类别词（"降压药""活血化瘀类中药"）与家常食材。
    也就是说**类别概括本身是允许输出的**（那是必要的安全信息），被概化掉的只有具体药名。
    """
    from app import constraints as CN

    blob = json.dumps(payload, ensure_ascii=False)
    clean, replaced = CN.scrub(blob, allowed)
    left = CN.foreign_herbs(clean, allowed)
    out = json.loads(clean) if replaced else payload
    if replaced or left:
        # `remaining_count` 非 0 说明守卫没兜干净 —— 上报给调用方，别静默
        out["output_guard"] = {"generalized": bool(replaced),
                               "generalized_count": len(replaced),
                               "remaining_count": len(left)}
    return out


def _run(tool: str, fn) -> dict:
    """统一兜底：**不吞异常**。

    2026-09-16 的教训：`except Exception` 静默兜底会让整条链路消失而单测仍全绿。
    这里把异常打到 stderr 留痕（stdio 下 stdout 是协议流，不能碰），
    同时以**结构化错误**返回，让调用方看得见失败而不是拿到一段空的"成功"。
    """
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 —— 工具边界，必须兜住
        import traceback
        _boot.log(f"[mcp:{tool}] 执行失败: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {"ok": False, "tool": tool,
                "error": f"{type(e).__name__}: {e}",
                "hint": "服务端已记录堆栈，请检查参数是否符合工具说明。"}


def _allow_blob(*parts) -> str:
    return "\n".join(str(p) for p in parts if p)


# ---------------------------------------------------------------------------
# 工具 1：典籍检索
# ---------------------------------------------------------------------------
def tcm_search(query: str, k: int = 4, max_chars: int = 1200) -> dict:
    """典籍检索（纯检索，**不调大模型**，只走 embedding + rerank）。"""
    def _impl() -> dict:
        from app import credibility as CR
        from app.rag import RAGSession
        from app.safety import strip_meta_info

        q = (query or "").strip()
        if not q:
            return {"ok": False, "tool": "tcm_search", "error": "query 不能为空"}

        sess = RAGSession(conv_id=None, k_final=max(1, min(int(k or 4), 12)))
        hits, attempts, reason = sess._retrieve(q)

        rows = []
        for d, score in hits:
            text = CR.filter_chunk(strip_meta_info(d.page_content))[0]
            rows.append({
                "source": d.metadata.get("source", "?"),
                "chapter": d.metadata.get("chapter", "?"),
                "score": round(float(score), 4),
                "text": text[:max(1, int(max_chars or 1200))],
            })
        return {
            "ok": True, "tool": "tcm_search", "query": q,
            "count": len(rows),
            "results": rows,
            "reflection": {
                "attempts": len(attempts) if isinstance(attempts, list) else 1,
                "reason": str(reason or ""),
            },
            "citation_rule": "引用时请标注「来源: 书名 · 篇章」，且只能引用 text 里实际出现的文字。",
        }
    return _run("tcm_search", _impl)


# ---------------------------------------------------------------------------
# 工具 2：安全判读（本服务的核心，P0）
# ---------------------------------------------------------------------------
def tcm_safety_check(items: list[str], states_text: str,
                     profile: dict[str, Any] | None = None) -> dict:
    """安全判读：返回**确定性档位**（不调大模型，毫秒级）。

    【为什么 `states_text` 故意不给默认值】（2026-09-17 部署校验实测）
        它原本是 `states_text: str = ""`。结果在一次调用里参数名被写成了 `states`，
        而 pydantic 对多余字段的默认行为是**静默丢弃** —— 于是 `states_text` 取了
        空值，妊娠状态没进判定，`当归` 返回了 **"可执行"**，不报任何错。
        这不是普通的笔误：**安全判读最坏的失败模式就是漏报禁忌**，
        而"缺输入 → 给最宽松档"恰好制造了它。

        改成必填后，漏传 = 参数校验失败（调用方立刻看到错误）；
        确实没有状态信息就**显式传空串** —— 那是"我确认用户没提"，
        与"我忘了传"是两件事，必须在调用契约上分开。
    """
    def _impl() -> dict:
        from app import constraints as CN
        from app.safety import (active_tags, annotate_origins, scan_profile,
                                scan_text)
        from app.safety import tiers as T
        from app.safety.scripts import TIER_SCOPE_NOTE

        words = [str(x).strip() for x in (items or []) if str(x).strip()]
        st = (states_text or "").strip()
        if not words and not st:
            return {"ok": False, "tool": "tcm_safety_check",
                    "error": "items 与 states_text 不能同时为空"}

        # 判定文本 = 用户提到的东西 + 状态原话。两者都要进扫描，
        # 因为规则命中与状态识别都靠这段文本。
        q = "、".join(words) + ("。" + st if st else "")
        prof = dict(profile or {})

        hits = annotate_origins(scan_text(q), scan_profile(prof), q)
        tags = active_tags(q, profile=prof)
        # 状态必须先算出来、并**直接参与档位判定**（第五轮退化：识别到妊娠、
        # 档位却没跟上，结果"明确禁止"被压成"需专业确认"）
        states = CN.detect(q, prof, None)
        T.apply_tiers(hits, tags, [], states=states)

        rows = []
        for h in hits:
            rows.append({
                "item": h.name,
                "kind": h.kind,                 # herb / drug / population
                "level": h.level,               # high / medium / info
                "has_rule": True,               # 见下方「无规则条目」分支
                "tier": h.tier,
                "tier_label": h.tier_label or T.TIER_LABEL.get(h.tier, ""),
                "severity": T.TIER_SEVERITY.get(h.tier, 0),
                "reading": h.reading,           # 给人看的判读正文（tiers.render 生成）
                "basis": h.tier_basis,          # 「凭什么定这一档」
                "matched": h.matched,           # 实际命中的词
                "cond_labels": list(h.cond_labels or ()),      # 命中了用户的哪些条件
                "unassessed": list(h.unassessed_gaps or ()),   # 还有哪些条件没评估
                "taking": bool(h.taking),       # 是否"正在服用"
            })

        # 未命中任何规则的条目**也必须出现在 items 里**。
        # 反例（2026-09-17 首个真实客户端实测）：问「当归鸡汤」，"鸡肉" 因为
        # 一条规则都没命中，就从 items 里**彻底消失**了 —— 于是「规则库里没有
        # 这条」与「判过、且没有禁忌」在结果里长得一模一样，调用方只能去
        # evaluated 反推。这正是我们一贯反对的静默降级：没有结论 ≠ 没有风险。
        seen = {str(h.matched or "").strip() for h in hits}
        seen |= {str(h.name or "").strip() for h in hits}
        for w in words:
            if w in seen:
                continue
            seen.add(w)
            rows.append({
                "item": w,
                "kind": "unknown",
                "level": "info",
                "has_rule": False,
                "tier": None,               # 刻意留空：它没有档位，不是"可执行"
                "tier_label": "无规则条目",
                "severity": None,
                "reading": (f"规则库里**没有**关于「{w}」的条目 —— 也就是"
                            "**没有可判的禁忌依据**。注意这**不等于**「已确认安全」："
                            "只是没有规则可判，不要读成「可以放心用」。"),
                "basis": "未命中任何规则条目（既不是禁忌，也不代表已放行）",
                "matched": w,
                "cond_labels": [],
                "unassessed": [],
                "taking": False,
            })

        # 放行集合：调用方提到的词 + 状态原话 + 档案 + 命中项 + 类别词/家常食材
        allowed = CN.allowed_names(_allow_blob(st, q), prof, hits)

        payload = {
            "ok": True,
            "tool": "tcm_safety_check",
            "evaluated": words,
            "states": {
                "labels": list(CN.labels(states) or ()),
                "notes": [str(s.get("user_note") or "").strip()
                          for s in (states or ()) if str(s.get("user_note") or "").strip()],
                "absolute": list(CN.absolute_state_labels(states) or ()),
                "perinatal": bool(CN.is_perinatal(states)),
            },
            "items": rows,
            "tier_counts": T.tier_counts(hits),
            # 「没有规则」与「判过且没问题」必须由调用方一眼分开，所以除了
            # has_rule=false 的行本身，还要给一条**怎么解读它**的纪律——
            # 否则客户端模型可能把"规则库没提"读成"可以放心用"。
            "no_rule_policy": (
                "items 里 has_rule=false 的条目表示**规则库没有对应规则**"
                "（tier 为 null，不是档位），即**没有可判的禁忌依据**；"
                "这**不等于**「已确认安全」。回答时如实说「资料库没有关于它的条目」，"
                "不要把 tier=null 与 tier=ok 混为一谈，也不要替它下"
                "「可以放心用」的结论。"
            ),
            # 档位判的是「自行食用」，不是医师处方 —— 用户举古籍质疑时的口径。
            # 单一来源在 app/safety/scripts.py 的 TIER_SCOPE_NOTE。
            "scope_discipline": TIER_SCOPE_NOTE,
            # 回显**实际收到**的入参。SDK 对多余字段是静默丢弃的，参数名写错时
            # 调用方只能从这里看出"我传的东西没进来"，否则会拿到一个偏松的档位
            # 而毫无察觉（2026-09-17 实测：`states` 写成 `states_text` 外，
            # 当归被判成"可执行"）。
            "input_received": {"items": list(words), "states_text": st[:300]},
            "confidence": "low" if not st else "normal",
            "item_naming": (
                "item 是**规则名**（可能是类别，如「活血化瘀类中药」）；"
                "matched 是**实际命中的词**——用它把结论对回你问的那样东西"
                "（例：问「当归」→ item=活血化瘀类中药、matched=当归）。"
            ),
            "judgment_policy": (
                "tier 是规则库的确定性结论，不得改写或弱化；"
                "unassessed 非空时如实在回答里说明还有哪一点没确认，"
                "不要替用户假设，也不要因此把档位说软。"
            ),
        }
        if not st:
            # 显式传空串 = "我确认用户没提状态"。此时档位**不含状态维度**，
            # 必须把这件事说出来，不能让它藏在 confidence 字段里被忽略。
            payload["caveat"] = (
                "本次未提供状态与用药信息（states_text 为空）：上面的档位**只依据"
                "所查项目本身**，未考虑孕期、哺乳、在服西药、慢病等情形。"
                "同一味药在那些状态下档位可能更严 —— **不要把这批档位当作"
                "「可以放心使用」的依据**；补上用户状态原话后重判一次。"
            )
        return _guard_payload(payload, allowed)
    return _run("tcm_safety_check", _impl)


# ---------------------------------------------------------------------------
# 工具 3：体质辨识
# ---------------------------------------------------------------------------
def tcm_constitution(answers: dict[str, list[int]]) -> dict:
    """九分法体质辨识（纯算术，不调大模型）。"""
    def _impl() -> dict:
        from app.agent import constitution as C

        if not isinstance(answers, dict) or not answers:
            return {"ok": False, "tool": "tcm_constitution",
                    "error": "answers 必须是 {题目分组: [各题得分]} 形式的对象",
                    "expected_example": '{"气虚质": [3, 2, 4], "阳虚质": [2, 2, 3]}'}

        r = C.compute(answers)
        return {
            "ok": True, "tool": "tcm_constitution",
            "primary": r.primary,
            "primary_key": r.primary_key,
            "tendencies": list(r.tendencies or ()),
            "note": r.note,
            "careless": bool(r.careless),
            "care_points": _jsonable(C.care_points(r.primary_key)),
            "total_questions": C.total_questions(),
            "scores": [{"key": s.key, "name": s.name, "code": s.code,
                        "raw": s.raw, "transform": round(float(s.transform), 2),
                        "verdict": s.verdict, "is_primary": bool(s.is_primary)}
                       for s in r.scores],
            "report": C.format_report(r),
            "judgment_policy": (
                "careless 为真表示作答无差别，结论不可信 —— 请如实告知，不要硬给结论。"
            ),
        }
    return _run("tcm_constitution", _impl)


# ---------------------------------------------------------------------------
# 工具 4：信息缺口（该问用户什么）
# ---------------------------------------------------------------------------
def tcm_intake_gaps(question: str, profile: dict[str, Any] | None = None) -> dict:
    """算信息缺口 + 建议追问（纯规则，不调大模型）。"""
    def _impl() -> dict:
        from app import constraints as CN
        from app import inquiry as IQ
        from app import intake as IL
        from app.safety import active_tags, annotate_origins, scan_profile, scan_text

        q = (question or "").strip()
        if not q:
            return {"ok": False, "tool": "tcm_intake_gaps", "error": "question 不能为空"}
        prof = dict(profile or {})

        hits = annotate_origins(scan_text(q), scan_profile(prof), q)
        tags = active_tags(q, profile=prof)
        intent = IQ.intent_of(q)
        gaps = IL.gaps(prof, q, require=IQ.require_slots(intent), conv_id=None)
        fups = IQ.followups(q, conv_id=None, profile=prof, intent=intent,
                            gaps=gaps, hits=hits, tags=tags)

        allowed = CN.allowed_names(q, prof, hits)
        return _guard_payload({
            "ok": True, "tool": "tcm_intake_gaps",
            "intent": intent,
            "gaps": _jsonable(gaps),
            "questions": _jsonable(IQ.to_event(fups)),
            "max_questions": 3,
            "tone_rule": ("像聊天一样问，最多 3 条；不要把缺口列表原样念给用户，"
                          "也不要用「因为你没说 X 所以我只能给通用建议」这类清场式开场。"),
        }, allowed)
    return _run("tcm_intake_gaps", _impl)
