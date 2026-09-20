# -*- coding: utf-8 -*-
"""MCP **纯函数层**回归（零 API）。

与 `tools/test_mcp_server.py` 的分工（**故意分开**）：
    本文件       = 逻辑对不对 —— 不需要 `mcp` 包、不起进程，秒级
    test_mcp_server = 协议通不通 —— 起真实子进程走 JSON-RPC

分开的理由很实际：合成一个文件后，一个断言失败你分不清是规则库错了还是握手错了。

本文件额外覆盖三件协议测试不合适做的事：
    ① **纯函数层不许依赖 mcp 包**（子进程实测 sys.modules）——
       这样即使宿主环境没装 SDK，也能单独验规则逻辑；
    ② **只读保证**：调用全部 4 个工具前后，SQLite 状态必须一字不变
       （阶段 1 的硬约束：客户端模型误调用也不能污染真实数据）；
    ③ 确定性算例：体质辨识、慢病冲突、缺口上限。

跑法：python -m tools.test_mcp_tools      （判绿只认退出码）
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OK = 0
BAD = 0

_MUST_NOT_LEAK = ("阿司匹林", "华法林", "氯吡格雷", "水蛭", "麝香", "巴豆",
                  "甘遂", "朱砂", "雄黄", "丹参", "三七", "红参", "川芎",
                  "他汀", "红曲", "血脂康", "附子", "川乌", "益母草", "桃仁")


def check(label: str, ok: bool, detail: str = "") -> bool:
    global OK, BAD
    if ok:
        OK += 1
        print(f"  ✅ {label}")
    else:
        BAD += 1
        print(f"  ❌ {label}" + (f" —— {detail}" if detail else ""))
    return ok


def _by_matched(d: dict, word: str) -> dict | None:
    for it in (d.get("items") or []):
        if it.get("matched") == word or word in str(it.get("item", "")):
            return it
    return None


# ---------------------------------------------------------------------------
# ① 纯函数层不依赖 mcp 包
# ---------------------------------------------------------------------------
def test_no_sdk_dependency() -> None:
    print("\n【分层纪律 · 纯函数层不得依赖 mcp 包】")
    code = (
        "import sys; import app.mcp.tools as T;"
        "print('|'.join(sorted(m for m in sys.modules"
        " if m == 'mcp' or m.startswith('mcp.'))))"
    )
    p = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                       capture_output=True, text=True, encoding="utf-8")
    loaded = (p.stdout or "").strip()
    check("import app.mcp.tools 不会拉起 mcp 包", loaded == "",
          f"意外加载={loaded}（stderr={p.stderr[-200:] if p.stderr else ''}）")


# ---------------------------------------------------------------------------
# ② 安全判读：档位区分度
# ---------------------------------------------------------------------------
def test_tier_contrast() -> None:
    from app.mcp.tools import tcm_safety_check

    print("\n【安全判读 · 档位区分度（同一孕妇，两样东西必须不同档）】")
    a = tcm_safety_check(["当归"], "我怀孕5个月了，最近咳嗽，想喝当归鸡汤")
    b = tcm_safety_check(["川贝", "梨", "冰糖"], "我怀孕5个月了，最近咳嗽")

    check("A 调用成功", a.get("ok") is True, str(a)[:150])
    check("B 调用成功", b.get("ok") is True, str(b)[:150])

    ha = _by_matched(a, "当归")
    hb = _by_matched(b, "川贝")
    check("命中当归", ha is not None, f"items={[i.get('item') for i in a.get('items') or []]}")
    check("命中川贝", hb is not None, f"items={[i.get('item') for i in b.get('items') or []]}")

    if ha:
        check("当归 = 明确禁止", ha.get("tier") == "forbid", str(ha.get("tier")))
        check("当归档位中文名 = 明确禁止",
              ha.get("tier_label") == "明确禁止", str(ha.get("tier_label")))
        basis = str(ha.get("basis") or "") + str(ha.get("reading") or "")
        check("理由绑定妊娠（不得落到「需确认」）",
              any(k in basis for k in ("孕产", "孕期", "妊娠")), basis[:120])
        check("当归项不是「需专业确认」档", ha.get("tier") != "confirm")
    if hb:
        check("川贝 = 可执行", hb.get("tier") == "ok", str(hb.get("tier")))
    if ha and hb:
        check("两者档位明显不同", ha.get("tier") != hb.get("tier"),
              f"{ha.get('tier')} vs {hb.get('tier')}")


def test_state_payload_and_leak() -> None:
    from app.mcp.tools import tcm_safety_check

    print("\n【状态块 + 出口守卫】")
    d = tcm_safety_check(["当归"], "我怀孕5个月了，最近咳嗽")
    st = d.get("states") or {}
    check("状态识别出妊娠", "妊娠" in json.dumps(st, ensure_ascii=False),
          json.dumps(st, ensure_ascii=False)[:160])
    check("absolute 列出绝对禁忌状态", bool(st.get("absolute")), str(st.get("absolute")))
    check("围产期标记为真", st.get("perinatal") is True, str(st.get("perinatal")))
    check("states.notes 用「人话版」（不含内部药名）",
          not [k for k in _MUST_NOT_LEAK if k in str(st.get("notes"))],
          str(st.get("notes"))[:160])

    blob = json.dumps(d, ensure_ascii=False)
    leaks = [k for k in _MUST_NOT_LEAK if k in blob]
    check("零泄漏（逐名断言）", not leaks, f"泄漏={leaks}")
    check("零流程语言",
          not [k for k in ("再补充几点", "前面没有展开", "系统检查发现",
                           "挂起针对性结论") if k in blob])
    og = d.get("output_guard") or {}
    check("守卫报告了概化且无漏网",
          int(og.get("remaining_count") or 0) == 0, str(og))
    check("返回值带「不得改写档位」的纪律说明",
          bool(d.get("judgment_policy")), str(d.get("judgment_policy"))[:80])


# ---------------------------------------------------------------------------
# ③ 慢病 + 在服西药的交叉判读
# ---------------------------------------------------------------------------
def test_chronic_drug() -> None:
    from app.mcp.tools import tcm_safety_check

    print("\n【慢病 / 在服西药交叉】")
    d = tcm_safety_check(["甘草"], "我有高血压，一直在吃氨氯地平",
                         {"慢病": "高血压", "在服西药": "氨氯地平"})
    check("调用成功", d.get("ok") is True, str(d)[:150])
    g = _by_matched(d, "甘草")
    check("甘草有档位（不是无结论）", bool(g and g.get("tier")), str(g)[:120])
    drugs = [i for i in (d.get("items") or []) if i.get("kind") == "drug"]
    check("在服西药被识别并给出档位", bool(drugs),
          f"items={[i.get('item') for i in (d.get('items') or [])]}")
    if g:
        check("甘草不是「可执行」（高血压需注意）",
              g.get("tier") != "ok", str(g.get("tier")))
    check("matched 能把结论对回用户问的词",
          bool(g and g.get("matched") == "甘草"), str(g and g.get("matched")))


# ---------------------------------------------------------------------------
# ④ 体质辨识（确定性算例）
# ---------------------------------------------------------------------------
def test_constitution() -> None:
    from app.agent import constitution as C
    from app.mcp.tools import tcm_constitution

    print("\n【体质辨识 · 确定性算例】")
    ans = {}
    for t in C.SCALE:
        ans[t.key] = ([5] * len(t.questions) if t.key == "yangxu"
                      else [1] * len(t.questions))
    r = tcm_constitution(ans)
    check("调用成功", r.get("ok") is True, str(r)[:150])
    check("主体质 = 阳虚质（构造算例）", r.get("primary") == "阳虚质",
          str(r.get("primary")))
    check("无差别作答未被误判", r.get("careless") is False, str(r.get("careless")))
    check("给出调养要点", bool(r.get("care_points")), str(r.get("care_points"))[:120])
    check("给出各维度得分", len(r.get("scores") or []) > 0,
          f"n={len(r.get('scores') or [])}")
    check("量表题数 = 27", r.get("total_questions") == 27, str(r.get("total_questions")))

    print("\n【体质辨识 · 无差别作答护栏】")
    flat = {t.key: [3] * len(t.questions) for t in C.SCALE}
    r2 = tcm_constitution(flat)
    check("全选中间档 → careless 为真", r2.get("careless") is True,
          str(r2.get("careless")))
    check("careless 时仍提示结论不可信",
          "不可信" in str(r2.get("judgment_policy")), str(r2.get("judgment_policy"))[:80])

    print("\n【体质辨识 · 参数校验】")
    r3 = tcm_constitution({})
    check("空 answers → 结构化错误", r3.get("ok") is False and "error" in r3,
          str(r3)[:120])


# ---------------------------------------------------------------------------
# ⑤ 缺口追问（上限 3 条）
# ---------------------------------------------------------------------------
def test_intake_gaps() -> None:
    from app.mcp.tools import tcm_intake_gaps

    print("\n【信息缺口 / 追问上限】")
    r = tcm_intake_gaps("我最近秋招压力大，晚上睡不好，该怎么缓解")
    check("调用成功", r.get("ok") is True, str(r)[:150])
    check("意图判为 advice", r.get("intent") == "advice", str(r.get("intent")))
    qs = r.get("questions") or []
    check("追问不超过 3 条（与追问引擎同一上限）", 0 < len(qs) <= 3,
          f"n={len(qs)}")
    check("追问区带「像聊天不像问诊表」的语气约束",
          "聊天" in str(r.get("tone_rule")), str(r.get("tone_rule"))[:80])

    print("\n【纯知识题不该追问】")
    r2 = tcm_intake_gaps("《黄帝内经》里说的'正气存内，邪不可干'是什么意思")
    check("知识题意图不为 advice", r2.get("intent") != "advice",
          str(r2.get("intent")))
    check("知识题不给追问", len(r2.get("questions") or []) <= 3,
          str(len(r2.get("questions") or [])))


# ---------------------------------------------------------------------------
# ⑥ 只读保证
# ---------------------------------------------------------------------------
def _snapshot() -> str:
    from app import storage
    prof = storage.get_profile(None)
    cons = storage.list_conversations()
    return json.dumps({
        "conversations": len(cons or []),
        "profile": prof,
        "memories": len(storage.list_memories(limit=500, conv_id=None) or []),
    }, ensure_ascii=False, sort_keys=True, default=str)


def test_read_only() -> None:
    from app.agent import constitution as C
    from app.mcp.tools import (tcm_constitution, tcm_intake_gaps,
                               tcm_safety_check)

    print("\n【只读保证 · 调用工具不得改动数据】")
    # 预热：先把 sqlite 连接/迁移跑完，避免把"首次连接"的写入算到工具头上
    tcm_safety_check(["当归"], "我怀孕5个月了")

    db = ROOT / "store" / "chat.db"
    before = _snapshot()
    stat_before = (db.stat().st_size, db.stat().st_mtime_ns)

    tcm_safety_check(["当归"], "我怀孕5个月了，最近咳嗽")
    tcm_safety_check(["甘草"], "我有高血压，在吃氨氯地平", {"慢病": "高血压"})
    tcm_intake_gaps("我最近压力大睡不好，怎么缓解")
    tcm_constitution({t.key: [1] * len(t.questions) for t in C.SCALE})

    after = _snapshot()
    stat_after = (db.stat().st_size, db.stat().st_mtime_ns)
    check("调用前后 SQLite 状态一字不变", before == after,
          f"before={before[:120]} after={after[:120]}")
    check("chat.db 文件大小与修改时间未变", stat_before == stat_after,
          f"{stat_before} -> {stat_after}")


def test_missing_state_input() -> None:
    """★ 缺状态输入不得静默给宽松档（2026-09-17 部署校验实测的坑）。

    起因：校验脚本把参数名写成了 `states`（真名是 `states_text`），
    SDK 对多余字段**静默丢弃**，于是妊娠状态没进判定，
    `当归` 返回了 `tier=ok / 可执行` —— 不报任何错。
    安全判读最坏的失败模式就是漏报禁忌，这条链路必须堵死。
    """
    import inspect

    from app.mcp.tools import tcm_safety_check

    print("\n【缺状态输入 · 不得静默给宽松档】")

    # ① 契约层：states_text 必须无默认值（漏传 → 参数校验拦下）
    sig = inspect.signature(tcm_safety_check)
    p = sig.parameters.get("states_text")
    check("states_text 是必填参数（无默认值）",
          p is not None and p.default is inspect.Parameter.empty,
          f"default={getattr(p, 'default', '?')!r}")

    # ② 显式传空：允许，但必须自曝"不含状态维度"
    d = tcm_safety_check(["当归"], "")
    check("显式传空串 → 调用成功（空串＝我确认用户没提）", d.get("ok") is True)
    check("confidence 标为 low", d.get("confidence") == "low", str(d.get("confidence")))
    check("带 caveat 说明未考虑状态维度",
          bool(str(d.get("caveat") or "").strip()) and "状态" in str(d.get("caveat")),
          str(d.get("caveat"))[:160])
    check("caveat 明确否定「可以放心使用」",
          "放心使用" in str(d.get("caveat") or ""), str(d.get("caveat"))[:160])

    # ③ 回显：调用方据此确认自己传的入参真的进来了
    ir = d.get("input_received") or {}
    check("input_received 回显 items", ir.get("items") == ["当归"], str(ir)[:120])
    check("input_received 回显 states_text 为空串", ir.get("states_text") == "",
          str(ir)[:120])

    d2 = tcm_safety_check(["当归"], "我怀孕5个月了")
    check("给了状态时 confidence 为 normal", d2.get("confidence") == "normal",
          str(d2.get("confidence")))
    check("给了状态时不带 caveat", not d2.get("caveat"), str(d2.get("caveat"))[:120])
    check("给了状态时回显原文",
          (d2.get("input_received") or {}).get("states_text") == "我怀孕5个月了",
          str(d2.get("input_received"))[:120])

    # ④ 缺状态时**不得**把药材判成"可执行"这种宽松结论
    rows = d.get("items") or []
    loose = [x.get("item") for x in rows if x.get("tier") == "ok"
             and x.get("kind") != "food"]
    # 允许列出，但必须已经由 confidence/caveat 覆盖 —— 断言这两者是配套出现的
    check("缺状态时若出现宽松档，必须同时有 low + caveat（配套不缺失）",
          (not loose) or (d.get("confidence") == "low" and d.get("caveat")),
          f"宽松条目={loose}")


# ---------------------------------------------------------------------------
# ⑨ 无规则条目不得消失 + 档位适用边界口径
# ---------------------------------------------------------------------------
def test_no_rule_item_declared() -> None:
    """2026-09-17 首个真实客户端实测暴露的两件事。

    ① **没命中规则的条目会从 items 里消失**：问「当归鸡汤」时 "鸡肉" 一条规则
       都没命中 → 它不在 items 里，于是「库里没有这条」与「判过、没有禁忌」
       在结果里长得一模一样。属静默降级，必须由结果本身说清。
    ② **档位判的是「自行食用」，不是医师处方**：古籍里同类药材出现在妊娠病方中，
       这个"矛盾"用户一定会举出来 —— 口径必须是系统给的常量，不能靠临场发挥，
       更不能成为对方施压时把档位说软的入口。
    """
    from app.mcp.tools import tcm_safety_check
    from app.safety.scripts import TIER_SCOPE_NOTE

    print("\n【无规则条目 · 不得从结果里消失】")
    d = tcm_safety_check(["当归", "鸡肉"], "孕妇，想喝当归鸡汤")
    check("调用成功", d.get("ok") is True, str(d)[:150])

    rows = d.get("items") or []
    by_word = {i.get("matched"): i for i in rows}

    missing = [w for w in (d.get("evaluated") or []) if w not in by_word]
    check("evaluated 里每个词都在 items 里有交代（没有东西会消失）",
          not missing, f"消失={missing}；items={[i.get('item') for i in rows]}")

    chicken = by_word.get("鸡肉")
    check("「鸡肉」出现在 items 里", chicken is not None,
          f"items={[i.get('item') for i in rows]}")
    if chicken:
        check("「鸡肉」has_rule = false", chicken.get("has_rule") is False,
              str(chicken.get("has_rule")))
        check("「鸡肉」tier 为 null（不是 ok）", chicken.get("tier") is None,
              str(chicken.get("tier")))
        check("「鸡肉」tier_label = 无规则条目",
              chicken.get("tier_label") == "无规则条目",
              str(chicken.get("tier_label")))
        check("「鸡肉」reading 明说「不等于已确认安全」",
              "已确认安全" in str(chicken.get("reading") or ""),
              str(chicken.get("reading"))[:160])

    dg = [i for i in rows if i.get("matched") == "当归"]
    check("「当归」has_rule = true 且档位仍是 forbid（本次改动没软化它）",
          bool(dg) and dg[0].get("has_rule") is True
          and dg[0].get("tier") == "forbid",
          str(dg[:1])[:200])

    tc = d.get("tier_counts") or {}
    check("tier_counts 只统计真档位（合计等于有规则的行数）",
          None not in tc and sum(tc.values()) == len([i for i in rows
                                                      if i.get("has_rule")]),
          f"tier_counts={tc} 实有规则行={len([i for i in rows if i.get('has_rule')])}")

    print("\n【无规则条目的解读纪律】")
    pol = str(d.get("no_rule_policy") or "")
    check("带 no_rule_policy", bool(pol))
    check("no_rule_policy 明说「不等于已确认安全」", "已确认安全" in pol, pol[:160])
    check("no_rule_policy 禁止读成「可以放心用」", "放心用" in pol, pol[:160])

    print("\n【档位适用边界 · 自行食用 vs 医师处方】")
    sc = str(d.get("scope_discipline") or "")
    check("带 scope_discipline", bool(sc))
    check("scope_discipline 与安全层常量为同一来源（单一真相源）",
          sc == TIER_SCOPE_NOTE)
    check("明说判的是「自行食用」", "自行食用" in sc, sc[:160])
    check("承认典籍处方里的用法（不装作不存在）",
          "处方" in sc and "典籍" in sc, sc[:160])
    check("反向约束：不得因为举古籍/施压把档位说软", "说软" in sc, sc[:200])

    print("\n【新增文案本身不得成为泄漏渠道】")
    blob = json.dumps(d, ensure_ascii=False)
    leaks = [k for k in _MUST_NOT_LEAK if k in blob]
    check("零泄漏仍成立（口径里没有具体药名）", not leaks, f"泄漏={leaks}")
    check("出口守卫未残留漏网名",
          int((d.get("output_guard") or {}).get("remaining_count") or 0) == 0,
          str(d.get("output_guard")))


def main() -> int:
    test_no_sdk_dependency()
    test_tier_contrast()
    test_state_payload_and_leak()
    test_chronic_drug()
    test_constitution()
    test_intake_gaps()
    test_missing_state_input()
    test_no_rule_item_declared()
    test_read_only()
    print(f"\n{'=' * 56}")
    print(f"MCP 纯函数层回归：通过 {OK} 项，失败 {BAD} 项")
    return 1 if BAD else 0


if __name__ == "__main__":
    raise SystemExit(main())
