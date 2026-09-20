# -*- coding: utf-8 -*-
"""流水线自检：用"剧本"替换 LLM 与检索，**确定性地**验证控制流。

为什么要有它
------------
多 Agent 最怕的不是"某次答得不好"，而是"**控制流悄悄坏了**"：
回环只走一次、降级分支永远不进、审查被模型一句话放行、并行扇出少跑一个专家……
这些在真实跑批里表现为"结果偶尔怪怪的"，**极难定位**。所以把 LLM 换成剧本，
只留真实的条件边与状态流，1 秒内跑完、结果 100% 可复现。

六个用例的取舍
--------------
A 越界方案 + 模型审查漏放  → 验证「规则层压过模型」（最关键的一条）
B 干净方案               → 验证"不该进回环时别进"
C 红旗输入               → 验证路由层 0 LLM 短路
D 硬规则冲突             → 验证「用户档案判定不宜，方案却推荐」会被否决
E 信息不足               → 验证走追问分支、不产出方剂级内容
F 多智能体分工           → 验证三个专家节点**都**跑到了（并行扇出没哑火）

用法：
    python -m app.multiagent --selftest
"""
from __future__ import annotations

BAD_PLAN = {
    "layering": {"root_deficiency": "阳气不足", "manifestation": "寒湿内停",
                 "explain": "阳气虚则水湿不化"},
    "summary": "温阳安神",
    "diet": [{"item": "每日附子10克煎服", "why": "温阳", "for": "本虚"}],
    "lifestyle": [{"item": "23点前入睡", "why": "养阳", "for": "本虚"}],
    "exercise": [{"item": "散步", "why": "流通气血", "for": "本虚"}],
    "acupoint": [{"item": "按揉涌泉", "why": "引火归元", "for": "本虚"}],
    "contra": [{"item": "忌寒凉", "why": "伤阳"}],
    "promo": "此方三天见效，可根治失眠",
}

GOOD_PLAN = {
    "layering": {"root_deficiency": "脾阳不足", "manifestation": "寒湿困脾",
                 "explain": "脾阳不足则湿浊不化"},
    "summary": "温运脾阳、化湿和中",
    "diet": [{"item": "生姜陈皮水（温中化湿）", "why": "对路", "for": "标实"}],
    "lifestyle": [{"item": "23 点前入睡", "why": "养阳", "for": "本虚"}],
    "exercise": [{"item": "八段锦「调理脾胃须单举」", "why": "健脾", "for": "本虚"}],
    "acupoint": [{"item": "足三里（按揉）", "why": "健脾", "for": "本虚"}],
    "contra": [{"item": "忌生冷黏腻", "why": "助湿"}],
}

# 提示词的**独有**标记。注意不能用"辨证/证型"这类泛词做分派——
# 多个提示词都会出现，按这类词分派会张冠李戴（本项目自检时真踩过）。
_MARK_ROUTER = "**分流器**"
_MARK_SAFETY = "**安全审查员**"
_MARK_DIAGNOSER = "你是中医**辨证**分析师"
_MARK_DIET = "**食疗专家**"
_MARK_MERIDIAN = "**经络穴位专家**"
_MARK_MOVEMENT = "**运动起居专家**"
_MARK_PLANNER_FIX = "调理方案的**主控**"


def _diagnosis() -> dict:
    return {"syndromes": [{"name": "寒湿困脾证", "likelihood": "高",
                           "basis": "剧本依据"}],
            "differential": [], "evidence_refs": [], "confidence": "中",
            "layering": {"root_deficiency": "脾阳不足",
                         "manifestation": "寒湿困脾",
                         "explain": "脾阳不足则湿不化"},
            "insufficient": False, "note": ""}


def _fake_consult(conv_id: int) -> dict:
    """剧本化的问诊结果。

    **必须一起 mock**：否则 router 会因为"该会话没有体质判定"而把 pipeline
    降级成 need_consult，整条链路根本跑不到专家节点——自检就变成了
    "在测前置条件校验"，而不是在测控制流。这也说明自检不该依赖数据库状态。
    """
    return {
        "primary": "阳虚质", "primary_key": "yangxu",
        "tendencies": ["痰湿质"], "careless": False, "note": "",
        "report": "【剧本】主体质：阳虚质",
        "scores": [{"name": "阳虚质", "code": "C", "transform": 91.7,
                    "verdict": "是", "primary": True},
                   {"name": "痰湿质", "code": "E", "transform": 66.7,
                    "verdict": "是", "primary": False}],
        "chief_complaint": "最近老是累、身体沉", "profile": "",
    }


def _fake_search(query, k_final=4, max_rounds=2) -> dict:
    return {"block": f"[来源: 测试库 · 篇章]\n{query} 的原文片段",
            "evidence": [{"id": f"ev{abs(hash(query)) % 99999}", "source": "测试库",
                          "chapter": "篇章", "score": 0.8, "text": query,
                          "query": query}],
            "top_score": 0.8, "low_confidence": False, "final_query": query,
            "reason": "剧本", "rounds": [{"round": 1, "query": query,
                                          "top_score": 0.8, "accepted": True}]}


def _intake_stub(hits=None, gaps=None, stop=False) -> dict:
    """剧本化的接诊产物（避免自检写真实档案）。"""
    return {"profile": {"年龄": "45", "性别": "女"},
            "hits": hits or [], "tags": ["dampness"],
            "conflicts": [], "gaps": gaps or [],
            "screening": ["hypertension"] if hits else [],
            "stop": stop,
            "brief": "【跨轮材料】\n  ①我最近老是觉得累、身体沉"}


class _Script:
    """剧本：记录每个节点的提示词，并按需返回预设产物。

    `expert_overrides` 用来替换某个专家的产出——**用例 A / D 必须从这里注入
    坏内容**：首轮方案现在由三个专家产出、主控只做确定性汇总，
    往 planner 里塞坏方案是塞不进去的（这一点本身就是自检发现的）。
    """

    def __init__(self, plan: dict, safety_pass: bool,
                 expert_overrides: dict | None = None,
                 safety_resp: dict | None = None):
        self.plan = plan
        self.safety_pass = safety_pass
        self.expert_overrides = expert_overrides or {}
        # 审查结论注入：用例 H 需要造出"只红旗/只完整性"的情形，
        # 验证这类违规**不该**把整份方案换掉（降级判据 2026-09-16 修正）
        self.safety_resp = safety_resp or {}
        self.planner_calls = 0
        self.expert_calls: list[str] = []
        self.editor_kind: list[str] = []      # "正常版" / "降级版"

    def _expert(self, name: str, default: dict) -> dict:
        self.expert_calls.append(name)
        return self.expert_overrides.get(name, default)

    def json(self, prompt: str, system: str = "") -> dict:
        if _MARK_ROUTER in prompt:
            return {"route": "pipeline", "reason": "剧本：用户要完整方案"}
        if _MARK_SAFETY in prompt:
            # ★ 故意按 safety_pass 返回，用于验证"模型漏放"时规则层能否兜住
            resp = {"pass": self.safety_pass, "violations": [], "red_flags": [],
                    "fix_hints": ["请移除任何药材用量与疗效承诺"],
                    "severity": "low"}
            resp.update(self.safety_resp)
            return resp
        if _MARK_DIAGNOSER in prompt:
            return _diagnosis()
        if _MARK_DIET in prompt:
            return self._expert("diet_expert", {
                "principle": "温运脾阳",
                "items": [{"item": "生姜陈皮水", "why": "温中化湿",
                           "for": "标实", "ref": {}}],
                "replacements": [{"from": "生薏米", "to": "炒薏米",
                                  "reason": "生薏米偏凉"}],
                "avoid": [{"item": "阿胶", "why": "滋腻助湿"}]})
        if _MARK_MERIDIAN in prompt:
            return self._expert("meridian_expert", {
                "principle": "健脾化湿",
                "points": [{"point": "足三里", "location": "犊鼻下三寸",
                            "how": "按揉", "duration": "每次 3 分钟",
                            "for": "本虚", "caution": "", "ref": {}},
                           {"point": "丰隆", "location": "外踝尖上八寸",
                            "how": "按揉", "duration": "每次 3 分钟",
                            "for": "标实", "caution": "", "ref": {}}]})
        if _MARK_MOVEMENT in prompt:
            return self._expert("movement_expert", {
                "principle": "温阳化湿",
                "exercise": [{"item": "八段锦", "how": "每天 1 遍",
                              "for": "本虚", "why": "健脾", "ref": {}}],
                "lifestyle": [{"item": "23 点前入睡", "how": "固定作息",
                               "for": "本虚", "why": "养阳", "ref": {}}],
                "avoid": [{"item": "大汗淋漓", "why": "耗气伤阳"}]})
        if _MARK_PLANNER_FIX in prompt:
            self.planner_calls += 1
            return self.plan
        return {}

    def text(self, prompt: str, system: str = "") -> str:
        if "降级交付" in prompt:
            self.editor_kind.append("降级版")
            return "【降级】本次无法给出针对性方案，建议就医。"
        self.editor_kind.append("正常版")
        return "【正常方案】"


def _patch(N, sc: _Script, intake: dict | None = None) -> None:
    """把真实的 LLM / 检索 / 接诊全部换成剧本。"""
    N._search = _fake_search
    N.load_consult = _fake_consult
    N._llm_json, N._llm_text = sc.json, sc.text
    N._build_intake = lambda conv_id, text: (intake or _intake_stub())


def run() -> int:
    """跑完全部用例；返回 0 表示全部通过。"""
    import app.multiagent.nodes as N
    from app.multiagent.graph import run_pipeline

    results: list[tuple[str, bool, str]] = []

    # ---------------- 用例 A：越界方案，模型审查漏放 ----------------
    print("=" * 74)
    print("A. 食疗专家产出「附子10克 / 三天见效」，而模型审查返回 pass=true")
    print("=" * 74)
    bad_diet = {"principle": "此方三天见效，可根治失眠",
                "items": [{"item": "每日附子10克煎服", "why": "温阳",
                           "for": "本虚", "ref": {}}],
                "replacements": [], "avoid": []}
    sc = _Script(BAD_PLAN, safety_pass=True,
                 expert_overrides={"diet_expert": bad_diet})
    _patch(N, sc)
    st = run_pipeline(0, "帮我出一份完整的调理方案")
    safe = st.get("safety") or {}
    print(f"  专家节点执行      : {sc.expert_calls}")
    print(f"  主控重写次数      : {st.get('planner_runs')}   （期望 3 = 首版 + 2 次重写）")
    print(f"  审查结论          : pass={safe.get('pass')}  违规 {len(safe.get('violations') or [])} 条")
    for v in (safe.get("violations") or []):
        print(f"      [{v.get('by', 'llm')}] {v.get('detail', '')[:52]}")
    print(f"  是否降级交付      : {st.get('degraded')}    主编走：{sc.editor_kind[-1] if sc.editor_kind else '?'}")
    ok_a = (st.get("planner_runs") == 3 and safe.get("pass") is False
            and st.get("degraded") is True and sc.editor_kind == ["降级版"]
            and len(sc.expert_calls) == 3)
    results.append(("A 否决回环 + 规则层压过模型", ok_a,
                    "回环跑满 2 次重写后降级" if ok_a else "控制流不符预期"))

    # ---------------- 用例 B：干净方案 ----------------
    print()
    print("=" * 74)
    print("B. 方案干净 → 应一次通过，不进回环")
    print("=" * 74)
    sc = _Script(GOOD_PLAN, safety_pass=True)
    _patch(N, sc)
    st = run_pipeline(0, "帮我出一份完整的调理方案")
    safe = st.get("safety") or {}
    plan = st.get("plan") or {}
    print(f"  主控重写次数      : {st.get('planner_runs')}   （期望 1，且首轮不调 LLM）")
    print(f"  汇总方案条目      : 食疗{len(plan.get('diet') or [])} / "
          f"穴位{len(plan.get('acupoint') or [])} / "
          f"运动{len(plan.get('exercise') or [])} / 起居{len(plan.get('lifestyle') or [])} / "
          f"禁忌{len(plan.get('contra') or [])}")
    print(f"  审查结论          : pass={safe.get('pass')}   降级={st.get('degraded')}")
    ok_b = (st.get("planner_runs") == 1 and sc.planner_calls == 0
            and safe.get("pass") is True and st.get("degraded") is False
            and sc.editor_kind == ["正常版"]
            and len(plan.get("acupoint") or []) == 2
            and len(plan.get("contra") or []) >= 1)
    results.append(("B 正常路径：首轮零 LLM 汇总、不进回环", ok_b,
                    "一次通过" if ok_b else "误入回环或汇总缺项"))

    # ---------------- 用例 C：红旗短路 ----------------
    print()
    print("=" * 74)
    print("C. 输入「我最近一直消瘦，还胸口疼」→ 应短路到建议就医")
    print("=" * 74)
    sc = _Script(GOOD_PLAN, safety_pass=True)
    _patch(N, sc)
    st = run_pipeline(0, "我最近一直消瘦，还胸口疼")
    print(f"  路由              : {st.get('route')}  （期望 urgent）")
    print(f"  轨迹节点          : {[t['node'] for t in st['trace']]}")
    ok_c = (st.get("route") == "urgent"
            and [t["node"] for t in st["trace"]] == ["router", "urgent"])
    results.append(("C 红旗短路（规则层，不经 LLM）", ok_c,
                    "0 ms 直达建议就医" if ok_c else "未短路"))

    # ---------------- 用例 D：硬规则冲突（用户不宜，方案却推荐）----------------
    print()
    print("=" * 74)
    print("D. 安全规则判定「用户在服附子理中丸、高血压」→ 方案里推荐附子 → 应否决")
    print("=" * 74)
    hard_hits = [{"kind": "herb", "key": "herb:fuzi", "name": "附子",
                  "level": "high", "matched": "附子理中丸", "taking": True,
                  "escalated": False, "why": "有毒，需炮制控量",
                  "not_for": "高血压者不宜自行服用",
                  "verdict": "不建议你自行服用", "tags": ["hypertension"],
                  "tag_labels": ["高血压"]}]
    conflict_plan = {**GOOD_PLAN,
                     "diet": [{"item": "附子理中丸可温阳，建议继续服用",
                               "why": "温中散寒", "for": "本虚"}]}
    bad_diet = {"principle": "温中散寒",
                "items": [{"item": "附子理中丸可温阳，建议继续服用",
                           "why": "温中散寒", "for": "本虚", "ref": {}}],
                "replacements": [], "avoid": []}
    sc = _Script(conflict_plan, safety_pass=True,      # ★ 模型审查仍返回 pass
                 expert_overrides={"diet_expert": bad_diet})
    _patch(N, sc, intake=_intake_stub(hits=hard_hits, stop=True))
    st = run_pipeline(0, "帮我出一份完整的调理方案")
    safe = st.get("safety") or {}
    print(f"  审查结论          : pass={safe.get('pass')}（模型口径 pass=true）")
    for v in (safe.get("violations") or []):
        print(f"      [{v.get('by')}] {v.get('detail', '')[:60]}")
    ok_d = (safe.get("pass") is False and st.get("degraded") is True
            and any(v.get("by") == "rule" and v.get("type") == "冲突"
                    for v in (safe.get("violations") or [])))
    results.append(("D 硬规则 × 方案 跨层冲突否决", ok_d,
                    "规则层驳回模型放行" if ok_d else "冲突未被拦下"))

    # ---------------- 用例 E：信息不足 → 追问分支 ----------------
    print()
    print("=" * 74)
    print("E. 用户只说要方案、没给舌象/寒热/二便 → 应走追问分支，不出方剂")
    print("=" * 74)
    sc = _Script(GOOD_PLAN, safety_pass=True)
    _patch(N, sc, intake=_intake_stub(gaps=["tongue", "cold_heat", "stool_urine"]))
    st = run_pipeline(0, "帮我出一份完整的调理方案")
    ans = st.get("final_answer") or ""
    nodes = [t["node"] for t in st["trace"]]
    print(f"  轨迹节点          : {nodes}")
    print(f"  是否产出方案 JSON : {st.get('plan') is not None}  （期望 False）")
    print(f"  回答含追问        : {'舌' in ans}")
    ok_e = ("need_more" in nodes and st.get("plan") is None
            and sc.expert_calls == [] and "舌" in ans)
    results.append(("E 信息不足 → 强制追问，不出方剂", ok_e,
                    "先补齐信息再给方案" if ok_e else "越过了追问"))

    # ---------------- 用例 F：多智能体并行扇出 ----------------
    print()
    print("=" * 74)
    print("F. 三个专科专家是否都跑到（并行的 fan-out 没哑火）")
    print("=" * 74)
    sc = _Script(GOOD_PLAN, safety_pass=True)
    _patch(N, sc)
    st = run_pipeline(0, "帮我出一份完整的调理方案")
    order = [t["node"] for t in st["trace"]]
    print(f"  专家调用顺序      : {sc.expert_calls}")
    print(f"  完整轨迹          : {order}")
    plan = st.get("plan") or {}
    ok_f = (set(sc.expert_calls) == {"diet_expert", "meridian_expert",
                                     "movement_expert"}
            and len(plan.get("acupoint") or []) == 2
            and any("替换" in (d.get("item") or "") for d in (plan.get("diet") or [])))
    results.append(("F 三专家扇出 + 主控汇总", ok_f,
                    "三个专家都产出且被汇总" if ok_f else "专家未全部执行"))

    # ---------------- 用例 G：在服西药被方案提及 ≠ 推荐它 ----------------
    print()
    print("=" * 74)
    print("G. 规则层：方案提到「在服的西药」不算冲突（药材仍要抓）")
    print("=" * 74)
    from app.multiagent import checks as _ck
    drug_hit = [{"kind": "drug", "name": "降压药", "matched": "氨氯地平",
                 "level": "high", "taking": True, "tags": ["hypertension"]}]
    herb_hit = [{"kind": "herb", "name": "甘草", "matched": "甘草",
                 "level": "high", "taking": False, "tags": ["hypertension"]}]
    g1 = _ck.safety_conflicts("你有高血压在服氨氯地平，控盐是基础；"
                              "同时守住降压药纪律——调理是配合，不是替代", drug_hit)
    g1b = _ck.safety_conflicts("守住降压药纪律，不能自行停药或减量", drug_hit)
    g2 = _ck.safety_conflicts("建议每天用甘草泡水喝，可以降压", herb_hit)
    print(f"  ①「在服氨氯地平」(事实陈述/依从提醒) → 冲突 {len(g1)} 条（期望 0）")
    print(f"  ②「守住降压药纪律，不能自行停减」   → 冲突 {len(g1b)} 条（期望 0）")
    print(f"  ③「建议每天用甘草泡水喝」(药材推荐) → 冲突 {len(g2)} 条（期望 ≥1）")
    ok_g = (g1 == [] and g1b == [] and len(g2) >= 1)
    results.append(("G 在服西药被提及 ≠ 建议用药", ok_g,
                    "豁免依从语境、药材照抓" if ok_g else "误判规则未修好"))

    # ---------------- 用例 H：只有红旗/完整性违规时不得降级 ----------------
    print()
    print("=" * 74)
    print("H. 审查只给红旗 + 只引不判（无越界/冲突）→ 方案照常交付，不降级")
    print("=" * 74)
    sc = _Script(GOOD_PLAN, safety_pass=False, safety_resp={
        "red_flags": ["高血压长期服药人群需提示监测血压与就医阈值"],
        "violations": [{"type": "只引不判", "detail": "某条未给判读",
                        "severity": "medium"}]})
    _patch(N, sc)
    st = run_pipeline(0, "帮我出一份完整的调理方案")
    safe = st.get("safety") or {}
    print(f"  审查 pass / 红旗  : {safe.get('pass')} / {len(safe.get('red_flags') or [])} 条")
    print(f"  是否降级交付      : {st.get('degraded')}    主编走：{sc.editor_kind[-1] if sc.editor_kind else '?'}")
    ok_h = (safe.get("pass") is False and st.get("degraded") is False
            and sc.editor_kind == ["正常版"])
    results.append(("H 红旗/完整性违规不吞方案", ok_h,
                    "只有越界/冲突才降级" if ok_h else "仍把方案换成了降级文案"))

    # ---------------- 用例 I：喂给审查员的方案不切在句子中间 ----------------
    print()
    print("=" * 74)
    print("I. 送审方案按整行裁剪（不切半句、省略显式标注）")
    print("=" * 74)
    demo = ["第一条完整内容" * 12, "第二条完整内容" * 12, "第三条完整内容" * 12]
    out = N._clip_lines(demo, 200)
    parts = out.split("\n")
    print(f"  行数 {len(demo)} → 输出 {len(parts)} 行；末行：{parts[-1][:34]}")
    ok_i = (all((p in demo) or p.startswith("（…余下") for p in parts)
            and len(parts) <= len(demo))
    results.append(("I 送审文本整行裁剪", ok_i,
                    "无半句截断、省略有标注" if ok_i else "仍会切在句子中间"))

    # ---------------- 汇总 ----------------
    print()
    print("=" * 74)
    for name, ok, why in results:
        print(f"  {'✅' if ok else '❌'} {name:<38} {why}")
    all_ok = all(ok for _n, ok, _w in results)
    print("=" * 74)
    print(f"自检结果：{'全部通过' if all_ok else '存在失败用例'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(run())
