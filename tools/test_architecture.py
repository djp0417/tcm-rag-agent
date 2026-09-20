# -*- coding: utf-8 -*-
"""架构层验收：按「验收方式（不再逐条考知识，只验架构）」的五条跑。

对应反馈文档的七、验收方式：

    1. 隔离验：同一会话先 A 人设问，再切 B 人设问，
              B 不得出现 A 的专属信息
    2. 弹性验：同一味药在两种相反证型下问，结论必须不同，理由各自绑定条件
    3. 追问验：只给主诉、不给舌象/用药史时，必须挂起结论并追问，
              不得借用历史信息
    4. 契约验：输出模块齐全度应稳定，不允许某轮缺失追问或风险管理
    5. 分级验：五种场景分别落到 明确禁止/需确认/不建议(无适应症)/对证有条件/可执行 五档

**全部是确定性检查（零 API、秒级）**：验的是"框架/分级/契约/归属"这些
由代码保证的部分。真实模型输出的验收走 `tools/e2e_safety_check.py`（打真服务）。

跑法：
    python -m tools.test_architecture
"""
from __future__ import annotations

import sys

from app import contract as CT
from app import framework as FW
from app import inquiry as IQ
from app import intake as IL
from app import storage
from app.safety import (active_tags, annotate_origins, build_block,
                        scan_profile, scan_text)
from app.safety import tiers as T

_OK = 0
_BAD = 0


def check(label: str, ok: bool, detail: str = "") -> bool:
    global _OK, _BAD
    if ok:
        _OK += 1
        print(f"  ✅ {label}")
    else:
        _BAD += 1
        print(f"  ❌ {label}" + (f" —— {detail}" if detail else ""))
    return ok


def _hits(q: str, profile: dict | None = None, gaps=None, conv_id=None):
    """跑一遍完整的确定性预扫（与 rag._safety_guard 同序）。"""
    hits = annotate_origins(scan_text(q), scan_profile(profile), q)
    tags = active_tags(q, profile=profile)
    intent = IQ.intent_of(q)
    if gaps is None:
        gaps = IL.gaps(profile, q, require=IQ.require_slots(intent),
                       conv_id=conv_id)
    fups = IQ.followups(q, conv_id=conv_id, profile=profile, intent=intent,
                        gaps=gaps, hits=hits, tags=tags)
    T.apply_tiers(hits, tags, gaps)
    screen = IL.screening_keys(tags, profile or {}, q, conv_id=conv_id)
    plan = FW.build(q, intent=intent, tags=tags,
                    hits=[h.to_dict() for h in hits], screening=screen,
                    gaps=gaps)
    return {"hits": hits, "tags": tags, "gaps": gaps, "fups": fups,
            "screen": screen, "plan": plan, "intent": intent}


# ---------------------------------------------------------------------------
# 分级验
# ---------------------------------------------------------------------------
def test_tiers() -> None:
    print("\n【分级验】五个场景必须落到五个不同档位")
    # 第三轮（P0-1）加的是第五档「不建议（无适应症）」：它与「明确禁止」是
    # **两件事**——前者是"没有对应的虚、没必要"，后者是"方向相反、用了有害"。
    # 鹿茸正是这条的分界样本：同一味药，阴虚火旺者是**禁忌**，
    # 健康人是**没必要**（反馈原话：两者都被答成"不能吃"）。
    cases = [
        ("附子能自己炖肉吃吗", "附子", T.TIER_FORBID, "毒性药材自行使用"),
        ("我最近手足心热、夜里盗汗，想吃点鹿茸补补", "鹿茸",
         T.TIER_FORBID, "阴虚 × 鹿茸 = 方向相反（禁忌）"),
        # 信息**齐全**的健康人：这是"无适应症"的成立条件——
        # 否定性结论（"你没有这个需要"）要求证据足够，否则应挂起追问
        ("我25岁男的，不怕冷不怕热，舌淡红苔薄白，大小便都正常，"
         "也没有高血压糖尿病，想吃点鹿茸补补行吗",
         "鹿茸", T.TIER_NOT_INDICATED, "无虚可补 → 不建议（并非有害）"),
        ("我45岁女，想用鹿茸补补，没有高血压也没有怕热",
         "鹿茸", T.TIER_CONFIRM, "信息不足 → 挂起 + 追问（不是保守默认）"),
        # ↓ 2026-09-16 真机补验挖出的缺陷：高血压 / 备孕原本落在"命中条件"里，
        #   被定成「对证但有条件」——**暗示"只要对证就能用"**，
        #   可鹿茸自己的 not_for 写着"高血压…禁用；孕妇禁用"。
        #   判据：not_for 写"禁用"的进 contra（禁忌），写"不宜/慎用"的留在 tags。
        ("我45岁，有高血压在吃降压药，想吃点鹿茸补补",
         "鹿茸", T.TIER_FORBID, "高血压 × 鹿茸 = 明文禁用（非'对证可用'）"),
        ("我32岁，正在备孕，想吃点鹿茸补补行吗",
         "鹿茸", T.TIER_FORBID, "备孕 × 鹿茸 = 明文禁用（孕产禁忌优先）"),
        ("我想吃点阿胶补血，我苔白腻、身重、大便黏马桶",
         "阿胶", T.TIER_CONDITIONAL, "命中湿困，方向对但先化湿"),
        ("陈皮泡水喝行吗", "陈皮", T.TIER_OK, "食药同源、无禁忌条件"),
    ]
    got: dict[str, str] = {}
    for q, name, want, why in cases:
        # 后两个鹿茸场景要"信息齐全"才有意义：显式传 gaps=[]
        r = _hits(q, gaps=[] if "没有高血压糖尿病" in q else None)
        hit = next((h for h in r["hits"] if h.name.startswith(name)), None)
        if not check(f"{q[:18]}… → {name}", hit is not None, "未命中"):
            continue
        got[f"{name}@{why[:6]}"] = hit.tier
        check(f"    档位={hit.tier_label}（期望 {T.TIER_LABEL[want]}）｜{why}",
              hit.tier == want, f"实际 {hit.tier}")
    check("五档全部出现（区分度）", set(got.values()) == set(T.TIER_ORDER),
          f"实际 {sorted(set(got.values()))}")
    # 禁忌 与 无适应症 的**结论口径**必须不同（不能都用"不能吃"打发）
    r1 = _hits("我最近手足心热、夜里盗汗，想吃点鹿茸补补", gaps=[])
    r2 = _hits("我25岁男的，不怕冷不怕热，舌淡红苔薄白，大小便都正常，"
               "也没有高血压糖尿病，想吃点鹿茸补补行吗", gaps=[])
    h1 = next(h for h in r1["hits"] if h.name == "鹿茸")
    h2 = next(h for h in r2["hits"] if h.name == "鹿茸")
    check("禁忌侧理由指向'方向相反'", "禁忌" in h1.reading or "添柴" in h1.reading,
          h1.reading[:60])
    check("无适应症侧说清'不是有害、是没必要'",
          ("不是" in h2.reading and "有害" in h2.reading), h2.reading[:60])
    check("两侧判读文本不同（同一味药不同人不能同话术）",
          h1.reading != h2.reading)
    # 档位依据必须**随事件下发**。第三轮首次落地时忘了把 tier_basis 放进
    # SafetyHit.to_dict，前端/落库拿到的是 undefined（静默退回 cond_labels）——
    # 功能看着"实现了"、实际一条没下发。这条断言就是拦它的。
    check("档位依据进了事件载荷（to_dict 里有 tier_basis）",
          bool(h1.to_dict().get("tier_basis")) and bool(h2.to_dict().get("tier_basis")),
          f"h1={h1.to_dict().get('tier_basis')!r}")
    check("依据措辞按语义分流（证候说明'方向相反'，人群说明'禁用情形'）",
          "方向相反" in h1.tier_basis,
          h1.tier_basis)
    r3 = _hits("我45岁，有高血压在吃降压药，想吃点鹿茸补补")
    h3 = next(h for h in r3["hits"] if h.name == "鹿茸")
    check("人群类禁忌不说'方向相反'（对孕妇说这句等于没说）",
          "禁用情形" in h3.tier_basis and "方向相反" not in h3.tier_basis,
          h3.tier_basis)


# ---------------------------------------------------------------------------
# 弹性验
# ---------------------------------------------------------------------------
def test_elasticity() -> None:
    print("\n【弹性验】同一味药，相反证型下结论与理由都必须变")
    wet = _hits("我想吃点阿胶补血，我苔白腻、身重、大便黏马桶")
    dry = _hits("我想吃点阿胶补血，舌淡苔薄白，大便正常，没有湿气")
    unknown = _hits("我想吃点阿胶补血，能吃吗")

    a_w = next(h for h in wet["hits"] if h.name == "阿胶")
    a_d = next(h for h in dry["hits"] if h.name == "阿胶")
    a_u = next(h for h in unknown["hits"] if h.name == "阿胶")

    check("湿困条件下 = 对证但有条件", a_w.tier == T.TIER_CONDITIONAL,
          a_w.tier_label)
    check("无湿象时 = 可执行", a_d.tier == T.TIER_OK, a_d.tier_label)
    check("条件未评估时 = 挂起（需专业确认）", a_u.tier == T.TIER_CONFIRM,
          a_u.tier_label)

    tw, td = T.render(a_w), T.render(a_d)
    check("两者结论不同", tw != td)
    check("湿困侧理由绑定用户条件", "痰湿" in tw or "湿困" in tw)
    check("无湿象侧理由不再提湿困禁忌", "你痰湿的表现" not in td)
    check("未评估侧明确说'结论先挂起'", "挂起" in T.render(a_u))

    # 前后端契约：判读正文必须**跟着命中项一起下发**（apply_tiers 里预渲染的
    # `reading`）。没有这一条，界面就只能显示规则里写死的 verdict——
    # 于是"后端已条件化、前端还在念通用话术"，而且**不报错**，最难查。
    check("命中项自带 reading（判读正文随事件下发）",
          bool(getattr(a_w, "reading", "")) and bool(getattr(a_d, "reading", "")),
          f"wet={bool(getattr(a_w, 'reading', ''))} dry={bool(getattr(a_d, 'reading', ''))}")
    check("reading 与 render() 一致（同一套判据）",
          a_w.reading == tw and a_d.reading == td)
    check("reading 确实按档位不同（湿困≠无湿象）", a_w.reading != a_d.reading)
    check("to_dict 携带 reading/tier/cond_labels（SSE 与刷新后回放同源）",
          all(k in a_w.to_dict() for k in
              ("reading", "tier", "tier_label", "cond_labels", "unassessed_gaps")),
          str(sorted(a_w.to_dict().keys()))[:120])


# ---------------------------------------------------------------------------
# 追问验
# ---------------------------------------------------------------------------
def test_inquiry() -> None:
    print("\n【追问验】只给主诉 → 挂起结论 + 追问；知识题不问")
    cid = storage.create_conversation("架构验收-追问")["id"]
    try:
        q = "我最近老是累，睡够了也没精神，怎么调理"
        r = _hits(q, conv_id=cid)
        check("意图判定 = advice", r["intent"] == "advice", r["intent"])
        check("产出追问 ≥2 条", len(r["fups"]) >= 2,
              str([f["label"] for f in r["fups"]]))
        check("追问不超过上限 3 条", len(r["fups"]) <= IQ.MAX_QUESTIONS)
        # 缺口里必须有"没有历史可借"的那几项（舌象/二便/在服药物）
        keys = {f["key"] for f in r["fups"]}
        check("优先问安全边界（在服药物）", "medication" in keys or
              "chronic" in keys, str(keys))

        # 追问兜底：模型一个都没问 → 代码必须补
        ans, added = IQ.ensure_questions("你的情况需要综合看，先别急着补。",
                                         r["fups"])
        check("模型没问 → 代码补上", bool(added) and "？" in ans or "?" in ans,
              str(added))

        # 知识题不许追问
        k = _hits("阳虚体质有什么表现？")
        check("知识题不追问", not k["fups"], str([f["label"] for f in k["fups"]]))

        # 借用历史：新会话时间线必须为空（不得拿别的会话当事实）
        cid2 = storage.create_conversation("架构验收-新会话")["id"]
        try:
            check("新会话时间线为空", not IL.timeline_entries(cid2))
            check("新会话档案为空", not storage.get_profile(cid2))
        finally:
            storage.delete_conversation(cid2)
    finally:
        storage.delete_conversation(cid)
        print("  （已清理测试会话）")


# ---------------------------------------------------------------------------
# 隔离验
# ---------------------------------------------------------------------------
def test_isolation() -> None:
    print("\n【隔离验】同一会话 A 人设 → 切 B 人设：B 不得沿用 A 的专属信息")
    cid = storage.create_conversation("架构验收-换人设")["id"]
    try:
        # A 人设带上"慢病 + 西药"，这样 A 的信息会**进本会话档案**——
        # 切到 B 人设时，档案扫出来的命中就是"A 的专属信息"，
        # 正是要验证"不得沿用"的那部分。
        qa = ("我45岁女，有高血压在吃氨氯地平，最近老累，怕冷，苔白腻，"
              "大便黏马桶，还胖了十来斤，怎么调理")
        rb = "我五心烦热，夜里盗汗，舌红少苔，口燥咽干，怎么调理"

        IL.update_from_message(qa, cid)
        # A 人设的专属信息（寒湿 + 体重增加）应已写入本会话档案/时间线
        check("A 人设已写入本会话时间线", len(IL.timeline_entries(cid)) >= 1)
        check("A 人设已写入本会话档案（慢病/西药）",
              bool(storage.get_profile(cid).get(IL.F_DRUGS)),
              str(storage.get_profile(cid)))

        IL.update_from_message(rb, cid)
        prof = storage.get_profile(cid)
        sw = IL.subject_switch(rb, cid, prof)
        check("检出方向反转（换人设）", sw is not None,
              "未检出——换人设会被当成'同一个人改口'")
        if sw:
            check("抓到的反转轴是 寒↔热", sw["axis"] == "寒↔热", sw["axis"])

        brief = IL.cross_turn_brief(cid, rb, profile=prof)
        check("既往原话被标为「待确认」", "待确认" in brief)
        check("明确禁止把既往记录当本轮依据",
              "不得作为本轮辨证依据" in brief)

        # 档案带出的命中在"待确认"状态下不得作为依据
        rb_hits = _hits(rb, prof, conv_id=cid)
        has_prof = any(h.origin == "profile" for h in rb_hits["hits"])
        check("B 人设确实会扫出 A 的档案命中（前提成立）", has_prof,
              str([(h.name, h.origin) for h in rb_hits["hits"]]))
        blk = build_block(rb_hits["hits"], rb_hits["tags"],
                          with_stop=False, profile_ok=False,
                          gaps=rb_hits["gaps"])
        check("档案命中不再以'本轮事实'呈现（profile_ok=False 生效）",
              "不得作为判断依据" in blk or "待确认" in blk)
    finally:
        storage.delete_conversation(cid)
        print("  （已清理测试会话）")


# ---------------------------------------------------------------------------
# 契约验
# ---------------------------------------------------------------------------
def test_contract() -> None:
    print("\n【契约验】六模块齐全度稳定，缺哪个补哪个")

    full = ("你的底子是脾阳不足，当前壅滞的是寒湿（本虚标实）。"
            "阿胶现在不适合，先化湿健脾。"
            "食疗可以吃山药、茯苓；足三里、关元可以按揉；"
            "起居上早睡、避寒；运动选散步、八段锦。"
            "降压药不能自行停、也不能自行减量，每天记录血压。"
            "建议先去查一下甲功和血常规。你的舌苔是什么样的？")
    check("六模块齐全时不再补", CT.missing(full) == [], str(CT.missing(full)))

    r = _hits("我45岁女高血压在吃氨氯地平，阿胶能吃吗")
    ctx = {"hits": [h.to_dict() for h in r["hits"]], "tags": sorted(r["tags"]),
           "screening": r["screen"], "plan": r["plan"], "gaps": r["gaps"],
           "stop": True}
    exp = CT.expected(ctx)
    check("含西药 → 要求风险管理", "risk" in exp, str(exp))
    check("有排查项 → 要求就医引导", "referral" in exp, str(exp))

    thin = "阿胶这个方向要看你有没有湿困，你的舌苔是什么样的？"
    out, mods = CT.enforce(thin, ctx)
    check("薄回答被补全", bool(mods), str(mods))
    check("补全内容包含用药纪律", "不能自行停" in out)
    # 断言"本轮该补的模块都补齐了"，而不是"六模块全齐"——
    # 用药安全类问题本就不该被补「分层判断」（那是症状调理框架的模块）
    left = [m for m in exp if m in CT.missing(out)]
    check("本轮应补模块已全部补齐", not left, str(left))

    # 知识题不许被补（否则"什么是阴虚"会被塞一段"分层看：你底子偏阴液不足"）
    k = _hits("什么是阴虚？有什么表现")
    kctx = {"hits": [h.to_dict() for h in k["hits"]], "tags": sorted(k["tags"]),
            "screening": k["screen"], "plan": k["plan"], "gaps": k["gaps"]}
    check("知识题 expectations 为空", CT.expected(kctx) == [],
          str(CT.expected(kctx)))
    out2, mods2 = CT.enforce("阴虚就是阴液不足，表现为口燥咽干、手足心热。", kctx)
    check("知识题不被补全", not mods2, str(mods2))

    # 框架：维度不因检索缺失而消失
    fr = FW.render(k["plan"])
    check("框架要求'维度不许整段消失'", "不要整段删掉这个维度" in fr)


def main() -> int:
    print("=" * 62)
    print("架构层验收（五条｜全确定性、零 API）")
    print("=" * 62)
    test_tiers()
    test_elasticity()
    test_inquiry()
    test_isolation()
    test_contract()
    print("\n" + "=" * 62)
    print(f"结果：{_OK} 项通过，{_BAD} 项失败")
    print("=" * 62)
    return 1 if _BAD else 0


if __name__ == "__main__":
    sys.exit(main())
