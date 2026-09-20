# -*- coding: utf-8 -*-
"""安全层单元回归：在服用判定 + 停药识别。

这两条都是**实测暴露出来的真 bug**，必须留回归用例：
  1. 「朋友推荐我吃阿胶」曾被判成 taking=True（"朋友推荐"误列在 _TAKING 里），
     话术从"是否适合"升级成"请暂停"，还会被写进档案污染后续所有轮次；
  2. 档案"只增不减"——用户说"我停了附子理中丸"，档案里仍记着在服。

用法：python -m tools.test_safety_rules
"""
from __future__ import annotations

from app import intake, storage
from app.safety import scan_text

SUPP = "在服中药与食疗"
RESET = "附子、薏米（薏苡仁）、桂圆（龙眼肉）、生姜红枣茶"

# 2026-09-16「记忆按会话隔离」后，档案存在 session_profile(conv_id, key) 里。
# 测试用一个**专属作用域 id**（不是真实会话，也不是 0 / -1 两个保留值），
# 跑完直接 clear_profile 清掉——比快照还原干净，也不会碰到任何真实对话。
TEST_CONV = 990001


def _reset() -> None:
    storage.set_profile(SUPP, RESET, TEST_CONV)


def test_taking() -> bool:
    print("一、「正在服用」判定")
    text = ("朋友推荐我吃阿胶，可我自己天天喝红豆薏米茶、生姜红枣茶，"
            "还吃桂圆，早晚各一颗附子理中丸")
    hits = {h.name: h.taking for h in scan_text(text)}
    expect = {
        "阿胶": False,          # 只是被推荐，不是在用
        "薏米（薏苡仁）": True,   # 天天喝
        "生姜红枣茶": True,
        "桂圆（龙眼肉）": True,   # 还吃
        "附子": True,           # 早晚各一颗
    }
    ok = True
    for name, want in expect.items():
        got = hits.get(name)
        good = got is want
        ok &= good
        print(f"  {'✅' if good else '❌'} {name:<12} taking={got}（期望 {want}）")

    # 2026-09-16 补：口语「在吃」必须算在服用。
    # 真机事故：「我45岁，女的，有高血压在吃氨氯地平」+ 同句「早晚各一颗
    # 附子理中丸」两项都判成 taking=False，话术从「请暂停并咨询」降级为
    # 「是否适合你」——而这两味恰恰是最需要升级话术的。
    print("  口语说法（在吃 / 正在吃）")
    colloquial = [
        ("有高血压在吃氨氯地平", {"降压药": True}),
        ("我高血压在吃氨氯地平，另外早晚各一颗附子理中丸",
         {"附子": True, "降压药": True}),
        ("我正在服用氨氯地平", {"降压药": True}),
        # 反向守卫：「朋友推荐…吃」不得因新增「在吃」被误判
        ("朋友推荐我吃阿胶", {"阿胶": False}),
    ]
    for text, want in colloquial:
        got = {h.name: h.taking for h in scan_text(text)}
        for name, w in want.items():
            good = got.get(name) is w
            ok &= good
            print(f"  {'✅' if good else '❌'} {text[:16]:<18} {name} "
                  f"taking={got.get(name)}（期望 {w}）")
    return ok


def test_stops() -> bool:
    print("\n二、停药识别（档案要能减）")
    cases = [
        ("附子理中丸我上周已经停了，薏米还在喝", ["附子"], "药名在前 + 时间状语"),
        ("我停了附子了", ["附子"], "停用词在前"),
        ("不吃桂圆了，太甜", ["桂圆"], "不吃 X 了"),
        ("我停了降压药，但还在吃附子理中丸", [], "★跨句不误判（关键反例）"),
        ("最近没什么变化", [], "无停用词"),
    ]
    ok = True
    for text, expect, why in cases:
        _reset()
        got = intake.note_stops(text, TEST_CONV)
        good = (len(got) == len(expect)
                and all(expect[i][:2] in got[i] for i in range(len(expect))))
        ok &= good
        print(f"  {'✅' if good else '❌'} {why:<24} → 摘掉 {got}")
        print(f"       剩余：{storage.get_profile(TEST_CONV).get(SUPP, '')}")
    return ok


def test_child_guard() -> bool:
    """上线清单③：儿童规则的年龄守卫（模板泄漏修复）。"""
    from app.safety import tag_from_text
    print("\n三、儿童规则年龄守卫（「我女儿25岁」不得触发儿童话术）")
    ok = True
    # 成年亲属 → 儿童规则与 child tag 一律不触发
    t1 = "我女儿25岁，口苦、苔黄腻，红豆薏米茶能天天喝吗"
    h1 = [h for h in scan_text(t1) if h.kind == "population"]
    g1 = tag_from_text(t1)
    good = not h1 and "child" not in g1
    ok &= good
    print(f"  {'✅' if good else '❌'} 成年亲属不触发儿童规则 "
          f"（population 命中={[h.name for h in h1]}，child tag={'child' in g1}）")
    # 真儿童语境 → 照常触发（守卫不能把儿童安全提醒也拦掉）
    t2 = "我女儿5岁，能喝金银花露吗"
    h2 = [h for h in scan_text(t2) if h.kind == "population"]
    good = any(h.key == "pop:child" for h in h2)
    ok &= good
    print(f"  {'✅' if good else '❌'} 真儿童语境照常触发（命中={[h.name for h in h2]}）")
    # 无年龄信息但明确提儿童 → 宽松放行
    t3 = "宝宝能喝苦丁茶吗"
    h3 = [h for h in scan_text(t3) if h.kind == "population"]
    good = any(h.key == "pop:child" for h in h3)
    ok &= good
    print(f"  {'✅' if good else '❌'} 提儿童无年龄照常提醒（命中={[h.name for h in h3]}）")
    return ok


def test_new_herbs() -> bool:
    """上线清单④：滋腻/温燥/偏凉补录条目 + 人参分品种。"""
    print("\n四、补录条目覆盖（熟地/鹿茸/苦丁茶 + 人参分品种）")
    cases = [
        ("熟地泡水喝补补血行吗", "熟地黄"),
        ("父亲托人买了鹿茸片，我能吃点吗", "鹿茸"),
        ("我天天喝苦丁茶降火", "苦丁茶"),
        ("人参、红参、西洋参有什么区别", "人参类（红参/生晒参/西洋参）"),
    ]
    ok = True
    for text, name in cases:
        hits = {h.name for h in scan_text(text)}
        good = name in hits
        ok &= good
        print(f"  {'✅' if good else '❌'} 「{text[:16]}…」→ {name}"
              f"{'' if good else f'（实际命中：{hits}）'}")
    # 分品种条目的 why 必须真的讲清三参差异（不能只是名字换了）
    from app.safety import rules as R
    r = R.HERB_BY_KEY["rensen"]
    good = all(w in r.why for w in ("红参", "生晒参", "西洋参"))
    ok &= good
    print(f"  {'✅' if good else '❌'} 人参规则 why 含红参/生晒参/西洋参三品种说明")
    return ok


def test_denial_guard() -> bool:
    """否定/停用句不得写进档案（2026-09-16 真机挖出的档案级污染）。

    实测现场：用户第 2 轮说「没有高血压糖尿病」，档案却变成
    `慢病 ＝ 孕产/备孕、高血压、糖尿病`——因为抽取用的是朴素的 `k in t`，
    而安全层早就有否定守卫、**只有档案抽取侧没有**。
    后果：用户明确否认的病成了他的长期事实，此后每轮卡片都印「命中：高血压」，
    且档案是 CSV 只增不减的合并 → 错到用户删会话为止。
    回答错只错一轮；档案错错一整个会话。
    """
    print("\n五、否定 / 停用句不得写进档案")
    ok = True
    cases = [
        ("我25岁男，舌淡红苔薄白，没有高血压糖尿病", [], [], "否认慢病"),
        ("我没有高血压", [], [], "否认单项"),
        ("我没吃降压药", [], [], "否认在服"),
        ("降压药已经停了", [], [], "停用（不是否认，但同样不该记在服）"),
        ("我停了降压药，但现在改吃氨氯地平", [], ["氨氯地平（降压药）"], "停旧换新"),
        ("我45岁女，有高血压在吃氨氯地平", ["高血压"], ["氨氯地平（降压药）"], "肯定句照常抽取"),
        ("朋友推荐我吃阿胶", [], [], "推荐 ≠ 在服"),
    ]
    for text, want_ch, want_drug, why in cases:
        ch = intake.extract_conditions(text)
        dr = intake.extract_drugs(text)
        good = (ch == want_ch and dr == want_drug)
        ok &= good
        print(f"  {'✅' if good else '❌'} {why}：「{text[:22]}」→ "
              f"慢病{ch} 西药{dr}"
              + ("" if good else f"（期望 慢病{want_ch} 西药{want_drug}）"))
    # 落库路径也要验：否认句不能改动档案
    # （注意 API：delete_profile(key, conv) 删单键；clear_profile(conv) 才是清空。
    #   时间线表对 conversations 有外键——落库路径必须用**真实会话**，
    #   用完即删（测试卫生：只写自己新建的会话，删会话即还原）。
    #   ⚠️ 2026-09-16 修复记录：此处曾误用 clear_profile(key, conv) 两参签名 +
    #   TEST_CONV 虚拟会话 id，先后炸在 TypeError 与外键上——
    #   这条断言实际上从未跑通过，"全过"是假通过。）
    cid = storage.create_conversation("safety-rules 落库验")["id"]
    try:
        intake.update_from_message("我25岁男，没有高血压糖尿病", cid)
        after = storage.get_profile(cid).get(intake.F_CHRONIC, "")
        good = after == ""
        ok &= good
        print(f"  {'✅' if good else '❌'} 走落库路径也不写档案：慢病＝{after!r}")
    finally:
        storage.delete_conversation(cid)
    return ok


def main() -> int:
    # 本测试会写档案（record_herbs / note_stops 的落库路径）。
    # 档案自 2026-09-16 起按会话隔离，测试用专属作用域 TEST_CONV，
    # 跑完清空即可——污染不了任何真实对话（旧版全局面板事故的根治）。
    try:
        a, b, c, d, e = (test_taking(), test_stops(), test_child_guard(),
                         test_new_herbs(), test_denial_guard())
    finally:
        storage.clear_profile(TEST_CONV)
    all_ok = a and b and c and d and e
    print("\n" + "=" * 60)
    print("安全层回归：", "全部通过" if all_ok else "存在失败")
    print(f"（测试作用域 {TEST_CONV} 的档案已清空）")
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
