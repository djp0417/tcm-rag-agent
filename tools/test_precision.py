# -*- coding: utf-8 -*-
"""精度与稳定性验收（第三轮）：按反馈文档"六、验收方式"的五条跑。

    1. 档位一致性验：同一回答中标题 / 结论 / 操作指南的档位表述必须一致；
                     构造「禁忌」与「无适应症」两个场景，验证给出**不同**档位
    2. 模块稳定性验：连续 5 轮，西医排查 / 经络 / 食疗量化 三模块齐全率必须 100%
    3. 引用验：检索含迷信 / 传说色彩的段落，验证被正确处理（剔除或标注文化背景）
    4. 弹性验（回归）：同一食材在寒 / 热两证下，结论必须相反
    5. 隔离验（回归）：同一会话三人切换，后问不得出现前人的专属信息

另加两条同源的机制验收（都在 P1 里）：
    6. 状态约束优先：备孕 / 妊娠类约束必须先于药名触发的规则生效
    7. 药名外泄守卫：回答里不许出现用户未提及、本轮也未命中的药食名

**全部确定性（零 API、秒级）**：验的是"机制"，不是"知识答得对不对"。
真实模型输出的验收走 `tools/e2e_safety_check.py`（打真服务）。

跑法：
    python -m tools.test_precision
"""
from __future__ import annotations

import sys

from app import constraints as CN
from app import contract as CT
from app import credibility as CR
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
    # 状态必须先于 apply_tiers 算出来（第五轮：状态直接决定档位）
    states = CN.detect(q, profile or {}, conv_id)
    T.apply_tiers(hits, tags, gaps, states=states)
    screen = IL.screening_keys(tags, profile or {}, q, conv_id=conv_id)
    plan = FW.build(q, intent=intent, tags=tags,
                    hits=[h.to_dict() for h in hits], screening=screen,
                    gaps=gaps)
    return {"hits": hits, "tags": tags, "gaps": gaps, "fups": fups,
            "screen": screen, "plan": plan, "intent": intent, "states": states,
            "query": q}


def _ctx(r: dict) -> dict:
    return {"hits": [h.to_dict() for h in r["hits"]],
            "tags": sorted(r["tags"]), "screening": r["screen"],
            "plan": r["plan"], "gaps": r["gaps"], "states": r["states"],
            "question": r["query"], "stop": False}


# ---------------------------------------------------------------------------
# 1. 档位一致性验
# ---------------------------------------------------------------------------
def test_tier_consistency() -> None:
    print("\n【档位一致性验】标题 / 结论 / 操作指南必须同档，且「禁忌」≠「无适应症」")

    # ---- 场景 A：禁忌侧（阴虚 × 鹿茸）----
    ra = _hits("我最近手足心热、夜里盗汗，想吃点鹿茸补补", gaps=[])
    ha = next(h for h in ra["hits"] if h.name == "鹿茸")
    check("A 场景档位 = 明确禁止", ha.tier == T.TIER_FORBID, ha.tier_label)
    check("A 判定依据指向'方向相反'",
          "方向相反" in ha.tier_basis, ha.tier_basis)

    # ---- 场景 B：无适应症侧（健康人 × 鹿茸）----
    rb = _hits("我25岁男的，不怕冷不怕热，舌淡红苔薄白，大小便都正常，"
               "也没有高血压糖尿病，想吃点鹿茸补补行吗", gaps=[])
    hb = next(h for h in rb["hits"] if h.name == "鹿茸")
    check("B 场景档位 = 不建议（无适应症）",
          hb.tier == T.TIER_NOT_INDICATED, hb.tier_label)
    check("两场景档位不同（同一味药因条件不同而分档）", ha.tier != hb.tier)
    check("B 依据说「未见需要它的依据」", "未见需要它" in hb.tier_basis,
          hb.tier_basis)
    check("B 判读明确说「不是有害」", "不是" in hb.reading, hb.reading[:60])
    check("A 判读不含「不必补 / 没必要」这类无适应症口径",
          "不是它有害" not in ha.reading)

    # ---- 档位纪律块：每档必须/禁止都下发给模型 ----
    blk = T.tier_discipline_block(ra["hits"])
    check("档位纪律块含「禁止出现」清单", "禁止出现" in blk)
    check("纪律块封死'实在想少量'这个口子",
          any("实在想" in x for x in T.TIER_SPEC[T.TIER_FORBID]["never"]))
    bb = build_block(ra["hits"], ra["tags"], with_stop=False, gaps=[])
    check("安全区块里带上了档位纪律（三条链路共用）",
          "档位纪律" in bb and "唯一判定点" in bb)

    # ---- 程序化一致性检查：回答里若出现与档位相反的口子 → 必须被更正 ----
    bad = ("鹿茸这个方向你可以考虑，如果你实在想补，少量试一点问题不大。"
           "**鹿茸**（明确禁止）：不要自行服用。")
    ctx = _ctx(ra)
    vs = CT.tier_violations(bad, ctx)
    check("检出与档位相反的口子", bool(vs), str(vs)[:100])
    out, mods = CT.enforce(bad, ctx)
    check("补了一句以档位为准的更正", "口径更正" in out and "tier_fix" in mods,
          str(mods))
    check("更正里点名了那一味与它的档位",
          "鹿茸" in out and "明确禁止" in out)
    # 合规回答不该被误判
    good = ("鹿茸对你属禁忌，不要自行服用；已经在吃的先停下并告诉经治医生。"
            "为什么：你有阴虚内热的表现，温燥峻补会加重口干、心烦。")
    check("合规回答不触发更正", not CT.tier_violations(good, ctx))


# ---------------------------------------------------------------------------
# 2. 模块稳定性验
# ---------------------------------------------------------------------------
def test_module_stability() -> None:
    print("\n【模块稳定性验】连续 5 轮：西医排查 / 经络 / 食疗量化 齐全率 100%")

    rounds = [
        "我45岁女，最近半年老是累，怕冷，苔白腻，大便黏马桶，还胖了十来斤，怎么调理",
        "我58岁，这两年一直乏力，夜里要起夜三四次，怎么调理",
        "我50岁男，长期腹泻两年了，一天好几次大便，人也没精神，怎么调理",
        "我夜里盗汗、潮热，睡不好，已经半年多了，怎么调理",
        "我62岁，腿沉、身重、口黏，苔黄腻，血压也有点高，怎么调理",
    ]
    ok_referral = ok_actions = ok_quant = 0
    for i, q in enumerate(rounds, 1):
        r = _hits(q)
        ctx = _ctx(r)
        thin = "你这个情况要从生活方式入手调理。"      # 故意给一份薄回答
        out, mods = CT.enforce(thin, ctx)
        has_ref = bool(CT.audit(out)["referral"])
        has_act = bool(CT.audit(out)["actions"])
        # 量化：补出来的可执行部分必须带用量/频次/时长这类要素
        quant = any(tok in out for tok in ("g", "克", "分钟", "每周", "每日",
                                           "次为一周期", "连吃", "为一周期"))
        ok_referral += has_ref
        ok_actions += has_act
        ok_quant += quant
        print(f"     第 {i} 轮：排查={has_ref} 可执行={has_act} 量化={quant}"
              f" | 命中排查项 {r['screen']}")
    n = len(rounds)
    check(f"西医排查 5 轮全齐（{ok_referral}/{n}）", ok_referral == n)
    check(f"经络/食疗可执行段 5 轮全齐（{ok_actions}/{n}）", ok_actions == n)
    check(f"可执行段含量化要素 5 轮全齐（{ok_quant}/{n}）", ok_quant == n)
    check("慢性主诉必定触发基础排查（规则表，与证型解耦）",
          all("chronic_basic" in _hits(q)["screen"] for q in rounds[:4]))
    # 框架侧也要求量化（不是只靠补全兜底）
    r = _hits(rounds[0])
    fr = FW.render(r["plan"])
    check("框架区块要求每个维度给量化要素", "可执行的量化要素" in fr)
    check("量化要求里含配比/频次/周期",
          all(k in fr for k in ("配比", "频次", "周期")))
    check("经络维度要求穴位定位与禁忌",
          all(k in fr for k in ("定位", "禁忌条件")))


# ---------------------------------------------------------------------------
# 3. 引用验
# ---------------------------------------------------------------------------
def test_credibility() -> None:
    print("\n【引用验】典籍里的迷信 / 传说记载必须被剔除或标注")

    mixed = (
        "鹿茸味甘咸，性温，主补肾壮阳、益精血、强筋骨，用于肾阳虚衰。"
        "李时珍云：鹿茸之中有小白虫，入人鼻必为虫颡，故用之须慎。"
        "服用宜从小量起，配以人参、熟地之类，忌与寒凉同用。"
    )
    cleaned, info = CR.filter_chunk(mixed)
    check("整块级别 = reject（含传说性记载）", info["level"] == CR.REJECT,
          info["level"])
    check("剔除了「小白虫入鼻」那句", "小白虫" not in cleaned and "虫颡" not in cleaned,
          cleaned[:80])
    check("保留了药性 / 宜忌这类可采信内容",
          "补肾壮阳" in cleaned and "忌与寒凉同用" in cleaned)
    check("给模型留了'本段有省略'的说明", "已略去" in cleaned)

    lore = "相传神农尝百草，一日而遇七十毒。本方自民间流传已久，古人云其效验。"
    cleaned2, info2 = CR.filter_chunk(lore)
    check("传说性表述标为 background（保留但标注）",
          info2["level"] == CR.BACKGROUND, info2["level"])
    check("background 段落注明不得作为依据", "仅供文化背景" in cleaned2)

    normal = "山药味甘性平，健脾益气，用于脾虚食少、久泻不止，可煮粥常服。"
    cleaned3, info3 = CR.filter_chunk(normal)
    check("正常医理段落原样通过（零误伤）",
          info3["level"] == CR.OK and cleaned3 == normal, cleaned3[:60])

    # 幂等：同一块处理两次不会重复补注释
    once, _ = CR.filter_chunk(mixed)
    twice, _ = CR.filter_chunk(once)
    check("重复处理是幂等的", once == twice)

    # 整块都是传说的 → 整块丢弃
    pure = "此方乃神仙所授，服之可成仙长生，鬼邪不敢近身，符咒祈禳即愈。"
    cleaned4, info4 = CR.filter_chunk(pure)
    check("整块迷信 → 整块丢弃（不喂给模型）",
          info4["rejected"] and cleaned4 == "")

    check("引用约束里含「可采信性筛选」与「引用必带判读」",
          "可采信性筛选" in CR.CREDIBILITY_BLOCK
          and "只引不判" in CR.CREDIBILITY_BLOCK)


# ---------------------------------------------------------------------------
# 4. 弹性验（回归）
# ---------------------------------------------------------------------------
def test_elasticity() -> None:
    print("\n【弹性验（回归）】同一食材，寒 / 热两证下结论必须相反")
    cold = _hits("我怕冷、手脚冰凉，苔白腻，想喝点绿豆汤")
    hot = _hits("我口苦、苔黄腻、小便黄，想喝点绿豆汤")
    c = next(h for h in cold["hits"] if h.name == "绿豆")
    h = next(h for h in hot["hits"] if h.name == "绿豆")
    check("寒证侧与热证侧档位不同", c.tier != h.tier,
          f"{c.tier_label} vs {h.tier_label}")
    check("两侧判读文本不同", c.reading != h.reading)
    check("寒证侧提到畏寒 / 寒象", "寒" in c.reading, c.reading[:50])
    check("热证侧为可执行（未命中禁忌条件）", h.tier == T.TIER_OK, h.tier_label)


# ---------------------------------------------------------------------------
# 5. 隔离验（回归）
# ---------------------------------------------------------------------------
def test_isolation() -> None:
    print("\n【隔离验（回归）】三人切换：后问不得出现前人的专属信息")
    cid = storage.create_conversation("精度验收-三人切换")["id"]
    try:
        IL.update_from_message(
            "我45岁女，有高血压在吃氨氯地平，怕冷，苔白腻，大便黏马桶", cid)
        # 第二个对象（明确关系词：爱人）→ 直接按新对象处理，不再反问
        q2 = "我爱人最近手足心热、夜里盗汗，该注意什么"
        IL.update_from_message(q2, cid)
        prof = storage.get_profile(cid)
        sw2 = IL.subject_switch(q2, cid, prof)
        check("检出方向反转", sw2 is not None)
        if sw2:
            check("关系词明确 → mode=new_subject（不反问）",
                  sw2["mode"] == "new_subject" and sw2["explicit"] is True,
                  str(sw2))
            check("带上了具体关系词（用于声明边界）",
                  "爱人" in (sw2.get("relation") or ""), str(sw2.get("relation")))
        brief2 = IL.cross_turn_brief(cid, q2, profile=prof)
        check("既往记录被标为待确认", "待确认" in brief2)
        # 判据取"声明块的表头"与"确认块的原话"，不要拿被引号括起来的禁令词去判
        # （声明块里写着「**不要**反问"是不是您本人"」，直接搜那句会误判成"在反问"）
        check("关系词明确 → 只声明边界，不反问",
              "直接按新对象处理" in brief2 and "我们这次说的还是" not in brief2,
              brief2[-260:])

        # 第三个对象（关系词缺失/模糊）→ 这时才该反问
        q3 = "再问一个，五心烦热、舌红少苔，怎么调理"
        IL.update_from_message(q3, cid)
        sw3 = IL.subject_switch(q3, cid, storage.get_profile(cid))
        if sw3:
            check("关系词模糊 → mode=confirm_subject（反问）",
                  sw3["mode"] == "confirm_subject", str(sw3))
            brief3 = IL.cross_turn_brief(cid, q3,
                                         profile=storage.get_profile(cid))
            check("模糊场景才反问（确认块原话出现）",
                  "先确认对象" in brief3 and "我们这次说的还是" in brief3,
                  brief3[-200:])
        else:
            check("模糊场景仍应检出方向反转", False, "未检出")
    finally:
        storage.delete_conversation(cid)
        print("  （已清理测试会话）")


# ---------------------------------------------------------------------------
# 6. 状态约束优先级
# ---------------------------------------------------------------------------
def test_state_priority() -> None:
    print("\n【状态约束优先验】备孕 / 妊娠必须先于药名规则生效")
    q = "我32岁，正在备孕，有高血压在吃氨氯地平，能喝点红花水吗"
    r = _hits(q)
    labels = CN.labels(r["states"])
    check("识别出状态约束（备孕）", any("备孕" in x for x in labels), str(labels))
    blk = CN.render_block(r["states"])
    check("块里写明强制判断顺序（状态→辨证→药食）",
          "先按上面这个状态确定安全边界" in blk)
    check("块里写明状态优先于药名规则", "高于任何由药名触发的规则" in blk)
    check("块里明确禁止把状态降级为背景信息",
          "降级为「背景信息」" in blk or "背景信息" in blk)
    check("块里注明不得引入用户未提过的药食名", "不要引入用户没有提过" in blk)
    ctx = _ctx(r)
    check("契约层把状态约束纳入风险管理", "risk" in CT.expected(ctx))
    out, mods = CT.enforce("先按你的情况写几句。", ctx)
    check("补出的风险管理里状态排在最前",
          "先按你的状态定边界" in out and out.index("先按你的状态定边界")
          < out.index("用药纪律") if "用药纪律" in out else True)
    check("补出的内容含备孕的具体边界", "备孕" in out)

    # 非状态场景不该误报
    plain = _hits("我最近老是累，怎么调理")
    check("没有状态词时不误报", not plain["states"], str(plain["states"]))


# ---------------------------------------------------------------------------
# 6.5 版块退化守卫（第五轮）：状态决定档位 / 场景适配 / 规则库不输出
# ---------------------------------------------------------------------------
def test_pregnancy_guard() -> None:
    """孕期用药咨询的三条退化守卫（第五轮反馈的验收 1/3/4/5）。

    这一版修的是"上一版已达标的能力被覆盖"，所以必须**锁成常驻回归**：
      ① 绝对禁忌状态 × 禁忌物 → 明确禁止（不允许落到"需专业确认"）；
      ② 家常食疗必须有独立档位，与禁忌物判然有别（不许挤在一档）；
      ③ 追问/排查随状态切换（妊娠不问抗凝、不给备孕套餐、要有产科引导）。
    """
    print("\n【孕期场景守卫】状态直接决定档位 / 场景适配（第五轮退化修复）")

    # ① 备孕 × 活血类（红花）→ 明确禁止
    r = _hits("我32岁，正在备孕，能喝点红花水吗")
    hx = next((h for h in r["hits"] if h.key.endswith("huoxue")), None)
    check("活血类命中（红花）", hx is not None)
    if hx:
        d = hx.to_dict()
        check("备孕 × 活血类 → 明确禁止（不是「需专业确认」）",
              d.get("tier_label") == "明确禁止",
              str(d.get("tier_label")) + " | " + str(d.get("tier_basis")))
        check("禁止理由绑定孕产状态", "孕产" in (d.get("tier_basis") or ""),
              str(d.get("tier_basis")))
        rd = d.get("reading") or ""
        check("判读文案不点名用户没提过的药（阿司匹林/丹参/三七…）",
              not any(n in rd for n in ("阿司匹林", "华法林", "氯吡格雷",
                                        "丹参", "三七", "川芎")),
              rd[:64])

    # ② 妊娠 × 家常食疗（川贝炖梨）→ 可执行/有条件，与①明显不同档
    r2 = _hits("我太太怀孕5个月了，川贝炖梨能吃吗？")
    cb = next((h for h in r2["hits"] if h.key.endswith("chuanbei")), None)
    check("川贝炖梨在库、可定档", cb is not None,
          str([h.key for h in r2["hits"]]))
    if cb:
        check("川贝炖梨 = 可执行 / 对证但有条件（与禁忌物判然有别）",
              cb.to_dict().get("tier_label") in ("可执行", "对证但有条件"),
              str(cb.to_dict().get("tier_label")))

    # ③ 追问 / 排查随状态切换
    r3 = _hits("我太太怀孕5个月了，一直咳嗽，川贝炖梨能吃吗？")
    gk = [f["key"] for f in r3["fups"]]
    check("妊娠期不追问抗凝 / 出血 / 支架项",
          not ({"anticoag_use", "bleeding", "cardiac_history"} & set(gk)), str(gk))
    check("妊娠期排查含产科就诊引导、不含备孕项",
          "pregnancy_care" in r3["screen"]
          and "fertility_workup" not in r3["screen"], str(r3["screen"]))
    # 备孕反过来：要有生育力评估，不要产科套餐
    r4 = _hits("备孕三年了还没怀上，月经总是往后推，该怎么调？")
    check("备孕场景：给生育力评估、不给产科套餐",
          "fertility_workup" in r4["screen"]
          and "pregnancy_care" not in r4["screen"], str(r4["screen"]))


# ---------------------------------------------------------------------------
# 7. 药名外泄守卫
# ---------------------------------------------------------------------------
def test_foreign_guard() -> None:
    print("\n【药名外泄守卫】不得把用户未提及的药材写进建议")
    q = "我最近有点累，喝了点黄芪水，行吗"
    r = _hits(q)
    ctx = _ctx(r)
    leaky = ("黄芪可以喝。你也可以试试红参、西洋参，或者加点川芎、红花一起煮，"
             "效果更好。")
    foreign = CN.foreign_herbs(leaky, CN.allowed_names(q, {}, ctx["hits"]))
    check("检出未提及的药食名", bool(foreign), str(foreign))
    check("检出的名字里含红参/川芎/红花",
          any(n in ("红参", "川芎", "红花") for n in foreign), str(foreign))
    out, mods = CT.enforce(leaky, ctx)
    # 2026-09-16 第五轮：守卫升级——不再"补一句澄清"（名字还留在正文里，
    # 用户照样会记下来），而是**把名字概化掉**。
    check("外泄药名被概化为类别表述（deleak）",
          "deleak" in mods
          and not any(n in out for n in ("红参", "西洋参", "川芎", "红花")),
          str(mods) + " | " + out.replace("\n", " ")[-64:])
    check("概化后仍保留类别表述、且用户提到的名字保留",
          ("其他" in out or "同类" in out) and "黄芪" in out,
          out.replace("\n", " ")[-64:])
    # 用户提到的 + 日常食养常用品不算外泄
    ok_ans = "黄芪可以喝，也可以配陈皮、茯苓一起煮水，每日 10~15g。"
    check("提到的 + 日常食材不误报",
          not CN.foreign_herbs(ok_ans, CN.allowed_names(q, {}, ctx["hits"])))


def main() -> int:
    print("=" * 66)
    print("精度与稳定性验收（第三轮｜全确定性、零 API）")
    print("=" * 66)
    test_tier_consistency()
    test_module_stability()
    test_credibility()
    test_elasticity()
    test_isolation()
    test_state_priority()
    test_pregnancy_guard()
    test_foreign_guard()
    print("\n" + "=" * 66)
    print(f"结果：{_OK} 项通过，{_BAD} 项失败")
    print("=" * 66)
    return 1 if _BAD else 0


if __name__ == "__main__":
    sys.exit(main())
