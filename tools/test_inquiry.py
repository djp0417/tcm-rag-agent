# -*- coding: utf-8 -*-
"""追问引擎回归（2026-09-16）。

背景（用户原话）：
> "我们是养生的项目，这个一定要有追问的功能，用户给的信息不是特别全的时候，
>  一定要学会主动追问，不然这个智能体就是废物。"

所以追问是**产品主干能力**，必须像安全规则一样有回归。本测试全是纯规则、
零 API、秒级完成，验这些事：

  一、意图闸门：**求建议**要追问；**问知识**不能追问
      （旧实现对所有问题挂 gap_block，导致"阳虚体质有什么表现"这种
        纯知识题被一句"信息不足"卡住，答不完整——这是真实缺陷）；
  二、缺口优先级与上限：一次最多 3 问，且决定安全边界的项排前面；
  三、会话感知：本会话已经说过的项不再重复问；
  四、程序化兜底：模型没问 → 代码补上；模型问了 → 不啰嗦重复；
  五、记忆存取闸门（与「记忆按会话隔离」配套）；
  六、**第四轮 P0 追问验**：第三人场景不重复问已知信息、问句切到
      咨询对象、主诉关联项（周期排卵/时间线/卒中影像）优先、
      换主体后"已知"同样隔离；
  七、**第四轮 P1 输出卫生**：内部流程语言不外泄、外泄守卫不复述药名、
      引用支持性自检、库外排查项标注常识性来源。

用法：python -m tools.test_inquiry
"""
from __future__ import annotations

from app import inquiry as IQ
from app import intake, storage


def test_intent() -> bool:
    print("一、意图闸门（该问 vs 不该问）")
    cases = [
        ("我最近老是累，怎么调理啊", "advice", "求建议 → 追问"),
        ("我最近老是累", "advice", "我 + 症状自述 → 追问"),
        ("45岁女，高血压在吃氨氯地平，能吃阿胶吗", "advice", "能不能吃 → 追问"),
        ("阳虚体质有什么表现？", "knowledge", "问知识 → 不追问"),
        ("《黄帝内经》里四气调神是什么意思", "knowledge", "问典籍 → 不追问"),
        ("红豆薏米茶有什么功效", "knowledge", "问功效 → 不追问"),
        ("红豆薏米茶能吃吗", "knowledge", "纯食材常识（无个人语境）→ 不追问"),
    ]
    ok = True
    for text, want, why in cases:
        got = IQ.intent_of(text)
        good = got == want
        ok &= good
        print(f"  {'✅' if good else '❌'} {why:<28} intent={got}"
              f"{'' if good else f'（期望 {want}）'}")
    return ok


def test_knowledge_not_asked(conv_b: int) -> bool:
    """纯知识问题必须**一个追问都不带**（否则等于不回答）。"""
    print("\n二、纯知识问题不追问")
    ok = True
    for text in ("阳虚体质有什么表现？", "红豆薏米茶有什么功效"):
        fups = IQ.followups(text, conv_id=conv_b)
        good = fups == []
        ok &= good
        print(f"  {'✅' if good else '❌'} 「{text}」→ 追问 {len(fups)} 条")
    return ok


def test_advice_asks(conv_a: int) -> bool:
    """求建议 + 信息稀薄 → 必须给出追问，且 ≤ 3 条、优先问安全边界项。"""
    print("\n三、求建议时的追问（有问、有序、有上限）")
    text = "我最近老是累，怎么调理啊"
    fups = IQ.followups(text, conv_id=conv_a)
    ok = True
    good = bool(fups)
    ok &= good
    print(f"  {'✅' if good else '❌'} 生成了 {len(fups)} 条追问："
          f"{[f['key'] for f in fups]}")

    good = len(fups) <= IQ.MAX_QUESTIONS
    ok &= good
    print(f"  {'✅' if good else '❌'} 不超过 {IQ.MAX_QUESTIONS} 条"
          f"（一次问太多用户会跑）")

    # 排序：在服药物/慢病 这类"决定安全边界"的项必须排在舌象/二便之前
    keys = [f["key"] for f in fups]
    if "tongue" in keys and "medication" in keys:
        good = keys.index("medication") < keys.index("tongue")
    else:
        good = True          # 截断到 3 条时可能没轮到舌象，不做强断言
    ok &= good
    print(f"  {'✅' if good else '❌'} 安全边界项（在服药物）优先于细节项（舌象）")

    # 话术必须是"用户能照着答"的具体问题，不是"请补充更多信息"这种空话
    good = all(len(f["ask"]) >= 12 for f in fups)
    ok &= good
    print(f"  {'✅' if good else '❌'} 每条追问都是具体问题（不是空泛的「请补充信息」）")

    # 提示词块要写清"不许在追问前下证型/开方"
    blk = IQ.followup_block(fups)
    good = "不要" in blk and "证型" in blk and "剂量" in blk
    ok &= good
    print(f"  {'✅' if good else '❌'} 提示词块含「追问前不下证型结论/不给剂量」约束")
    return ok


def test_session_aware(conv_a: int) -> bool:
    """本会话说过的项不再问（会话感知）。"""
    print("\n四、会话感知（说过的别再问）")
    first = IQ.followups("我最近老是累，怎么调理", conv_id=conv_a)
    # 用户把"在服药物/慢病/年龄性别"都交代了
    intake.update_from_message(
        "我今年45岁，女，有高血压，在吃氨氯地平", conv_a)
    second = IQ.followups("那饮食上我该怎么调", conv_id=conv_a)
    ok = True
    k1, k2 = {f["key"] for f in first}, {f["key"] for f in second}
    good = "medication" not in k2 and "chronic" not in k2 and "age_sex" not in k2
    ok &= good
    print(f"  {'✅' if good else '❌'} 已交代的项不再追问（首次 {sorted(k1)} → "
          f"交代后 {sorted(k2)}）")

    good = "tongue" in k2 or "stool_urine" in k2 or "cold_heat" in k2
    ok &= good
    print(f"  {'✅' if good else '❌'} 仍未交代的细节项继续追问（{sorted(k2)}）")
    return ok


def test_ensure_questions() -> bool:
    """程序化兜底：模型忘了问 → 代码补；模型问了 → 不重复。"""
    print("\n五、追问兜底（关键动作不依赖模型自觉）")
    items = [{"key": "medication", "label": "在服用的药物",
              "ask": "现在有没有在吃西药或中成药？", "signals": ("在吃", "西药")},
             {"key": "tongue", "label": "舌象",
              "ask": "舌苔是白是黄、厚不厚？", "signals": ("舌", "齿痕")}]
    ok = True

    silent = "你这是湿气重，建议少熬夜、多吃薏米。"
    out, added = IQ.ensure_questions(silent, items)
    good = len(added) == 2 and "现在有没有在吃西药" in out and "舌苔是白是黄" in out
    ok &= good
    print(f"  {'✅' if good else '❌'} 模型没问 → 代码补上 {len(added)} 条，"
          f"并给出通用安全建议"
          f"（{'通用建议在' if '通用的' in out or '作息' in out else '缺通用建议'}）")

    asked = ("先说通用的：作息规律、别久坐。\n"
             "1. 你现在有没有在吃西药或中成药？\n"
             "2. 方便看看舌苔是白是黄吗？")
    out2, added2 = IQ.ensure_questions(asked, items)
    good = added2 == [] and out2.strip() == asked.strip()
    ok &= good
    print(f"  {'✅' if good else '❌'} 模型已问 → 不追加（avoid 啰嗦重复）")

    # 只问了一条（<2）也视为没问够 → 补齐
    half = "1. 你现在在吃什么药吗？"
    _out3, added3 = IQ.ensure_questions(half, items)
    good = bool(added3)
    ok &= good
    print(f"  {'✅' if good else '❌'} 只问了 1 条（不足 2 条）→ 仍补齐")
    return ok


def test_subject_aware(conv_c: int) -> bool:
    """第四轮 P0：第三人场景的已知比对 + 追问对象跟随 + 主诉关联优先。"""
    print("\n七、追问验（第四轮 P0：不重复问、不问错对象、主诉关联）")
    ok = True

    # —— 场景 1（反馈原话）：爸爸 68 岁 / 男 / 在吃药 三项俱在 + 认知主诉
    q = ("我爸爸68岁，有高血压和糖尿病，一直在吃药，"
         "最近记性越来越差，经常忘事，该怎么调理？")
    fups = IQ.followups(q, conv_id=conv_c)
    keys = [f["key"] for f in fups]
    good = bool(fups)
    ok &= good
    print(f"  {'✅' if good else '❌'} 生成了追问：{keys}")

    for k in ("age_sex", "medication", "chronic"):
        good = k not in keys
        ok &= good
        print(f"  {'✅' if good else '❌'} 已明确的「{k}」不再追问"
              f"（反馈失效①：68岁/男/在吃药仍被问）")

    good = all("您父亲" in f["ask"] or "他" in f["ask"] for f in fups)
    ok &= good
    print(f"  {'✅' if good else '❌'} 问句切到咨询对象（您父亲），不再问「你的」"
          f"（反馈失效②）")

    good = any(k in keys for k in ("cog_timeline", "stroke_imaging",
                                   "living_care", "drug_names"))
    ok &= good
    print(f"  {'✅' if good else '❌'} 主诉关联项在问（时间线/卒中影像/照护/药名）"
          f"（反馈失效③：该问的没问）")

    blk = IQ.followup_block(fups)
    good = "咨询对象" in blk and "顶替" in blk
    ok &= good
    print(f"  {'✅' if good else '❌'} 提示词块声明了对象边界（不得用本人信息顶替）")

    # —— 场景 1b：信源用法不是换主体（真机挖出：e2e 三轮的"朋友推荐我吃阿胶"
    #    曾被识别成咨询对象「朋友」→ 本人档案被整体排除 → 已知全部失忆 →
    #    附子/降压药多挂出 chronic 缺口 → "挂了缺口却没说挂起"）
    for t in ("朋友推荐我吃阿胶，可我自己天天喝红豆薏米茶",
              "我朋友推荐我吃阿胶"):
        good = intake.subject_label(t) == ""
        ok &= good
        print(f"  {'✅' if good else '❌'} 信源用法不算换主体：「{t[:14]}…」→ 本人")

    # —— 场景 2：47 岁月经量少——年龄性别已明确，且不得硬贴抗凝问询
    q2 = "我47岁女性，月经量少，想调理一下，该吃什么？"
    fups2 = IQ.followups(q2, conv_id=conv_c)
    keys2 = [f["key"] for f in fups2]
    good = "age_sex" not in keys2
    ok &= good
    print(f"  {'✅' if good else '❌'} 「47岁女性」已交代 → 不再问年龄性别"
          f"（反馈：正文按女性辨证、结尾仍列年龄性别）")
    prof = storage.get_profile(conv_c)
    adv = intake.advisory_gaps(intake._tags_of(q2), prof, q2, conv_id=conv_c)
    good = not ({"bleeding", "cardiac_history"} & set(adv))
    ok &= good
    print(f"  {'✅' if good else '❌'} 无抗凝语境 → 不问支架/出血"
          f"（反馈：抗凝模板硬贴，advisory={adv}）")

    # —— 场景 3：备孕三年 → 周期排卵优先于通用三件套
    q3 = "备孕三年了还没怀上，月经总是往后推，想用中药调理，该怎么调？"
    fups3 = IQ.followups(q3, conv_id=conv_c)
    keys3 = [f["key"] for f in fups3]
    good = bool(keys3) and keys3[0] == "fertility_cycle"
    ok &= good
    print(f"  {'✅' if good else '❌'} 备孕场景首问「月经周期与排卵」"
          f"（实际 {keys3}）")
    scr = intake.screening_keys(intake._tags_of(q3), text=q3)
    good = "fertility_workup" in scr
    ok &= good
    print(f"  {'✅' if good else '❌'} 排查项挂上主诉关联（性激素/AMH/排卵/男方精液）"
          f"（实际 {scr}）")
    q4 = "我爸爸最近记性越来越差，经常忘事"
    scr4 = intake.screening_keys(intake._tags_of(q4), text=q4)
    good = "cognitive_workup" in scr4
    ok &= good
    print(f"  {'✅' if good else '❌'} 认知下降挂上认知评估/头颅影像（实际 {scr4}）")

    # —— 场景 4：换主体后"已知"也要隔离（女儿的事实不得顶替爸爸）
    # 构造：先说女儿 25 岁，再问爸爸（**不给年龄**）——若隔离失效，
    # 女儿的"25 岁"会被当成已知 → 不再问年龄；正确行为是继续问。
    storage.delete_conversation(conv_c)
    conv_c2 = storage.create_conversation("追问测试C2")["id"]
    try:
        intake.update_from_message("我女儿25岁，女，湿热体质，爱吃辛辣", conv_c2)
        fups5 = IQ.followups(
            "我爸爸最近老是头晕，该怎么调理", conv_id=conv_c2)
        keys5 = [f["key"] for f in fups5]
        good = "age_sex" in keys5
        ok &= good
        print(f"  {'✅' if good else '❌'} 换主体后女儿的信息不算爸爸的已知"
              f"（追问含年龄性别：{keys5}）")
        subj = {f.get("subject") for f in fups5}
        good = subj == {"父亲"}
        ok &= good
        print(f"  {'✅' if good else '❌'} 追问主体识别为「父亲」（{subj}）")
    finally:
        storage.delete_conversation(conv_c2)
    return ok


def test_output_hygiene() -> bool:
    """第四轮 P1 + 第五轮：内部语言不外泄 + **外泄药名概化** + 引用支持性。"""
    print("\n八、输出卫生（内部语言不外泄 / 外泄药名概化 / 引用支持性自检）")
    ok = True
    from app import contract as CT
    good = ("完整性检查" not in CT._HEAD and "系统" not in CT._HEAD
            and "再补充几点" not in CT._HEAD and "前面没有展开" not in CT._HEAD)
    ok &= good
    print(f"  {'✅' if good else '❌'} 补全段**无流程语言**（第五轮：「再补充几点/"
          f"前面没有展开」也算流程语言，已删；旧版残留会被剥掉）")

    ans = "丹参活血，红参、川芎也属活血类，机制类似。"
    ctx = {"hits": [{"kind": "herb", "name": "丹参", "key": "danshen",
                     "origin": "message"}],
           "question": "丹参粉能长期吃吗", "profile": {}}
    out, mods = CT.enforce(ans, ctx)
    good = ("deleak" in mods and "红参" not in out and "川芎" not in out
            and "丹参" in out)
    ok &= good
    print(f"  {'✅' if good else '❌'} 外泄药名被**概化**（第五轮：不再只补一句澄清，"
          f"而是直接把名字换掉）：{out.replace(chr(10), ' ')[-52:]}")

    # 流程语言残留（旧版历史被模型复述）→ 必须被剥掉
    legacy = ("正文。\n\n---\n\n**再补充几点（前面没有展开，但和你的情况直接相关）：**\n"
              "- 用药纪律：不能自行停药。")
    good = "再补充几点" not in CT._strip_flow(legacy)
    ok &= good
    print(f"  {'✅' if good else '❌'} 旧版流程语言引子在输出前被剥掉")

    from app.safety import scripts as SG
    good = "支持" in SG.CITATION_JUDGMENT and "镀金" in SG.CITATION_JUDGMENT
    ok &= good
    print(f"  {'✅' if good else '❌'} 引用判读含「支持性自检」要求"
          f"（防'高龄衰退'论证'非正常衰老'式牵强引用）")

    good = "完整性检查" not in SG.screening_block([])
    ok &= good
    scr_blk = SG.screening_block(["cognitive_workup"])
    good = "常识性" in scr_blk and "资料库" in scr_blk
    ok &= good
    print(f"  {'✅' if good else '❌'} 库外排查项（MMSE/影像）标注常识性来源，"
          f"不得说成资料记载")

    # 第五轮：状态文案必须是"用户版"（不含内部举例药名）
    from app import constraints as CN
    states = CN.detect("我在备孕，想调理一下", {}, None)
    good = (bool(states) and all(s.get("user_note") for s in states)
            and not any(bad in states[0].get("user_note", "")
                        for bad in ("红花", "桃仁", "水蛭", "巴豆", "麝香")))
    ok &= good
    print(f"  {'✅' if good else '❌'} 状态约束有**用户版**文案，内部举例药名不进输出"
          f"（{states[0].get('user_note','')[:28] if states else '—'}…）")
    return ok


def test_recall_gate() -> bool:
    """跨会话记忆的两道闸门（存：明说记住 / 取：主动问起）。"""
    print("\n九、记忆的存取闸门（与「记忆按会话隔离」配套）")
    from app.memory import wants_longterm, wants_recall
    ok = True
    for text, want in (("记住我长期熬夜到凌晨两点", True),
                       ("帮我记一下我不吃羊肉", True),
                       ("我最近老是累", False),
                       ("以后都按这个来", True)):
        got = wants_longterm(text)
        good = got == want
        ok &= good
        print(f"  {'✅' if good else '❌'} 长期记忆通道「{text[:12]}…」→ {got}")
    for text, want in (("我上次跟你说的阿胶还能吃吗", True),
                       ("还记得我的体质吗", True),
                       ("我最近老是累", False),
                       ("阳虚体质有什么表现", False)):
        got = wants_recall(text)
        good = got == want
        ok &= good
        print(f"  {'✅' if good else '❌'} 显式回忆通道「{text[:12]}…」→ {got}")
    return ok


def test_light_tone() -> bool:
    """第六轮：日常轻问题不被「清点式声明」压场（2026-09-16 晚实测）。

    用户只问了"秋招求职压力大怎么缓解"，回答却以「你这次只提到…没有说过
    体质/舌象/大便/用药…所以只能给不分型的安全建议」开场，正文逐节复读
    「资料库无直接依据」×4，结尾列 5 个追问——内部指令被模型说了出口。
    修法锁进提示词层：缺口清单只内部掌握（≤3 条）、禁清点式开场、
    库外标注全文最多一次、追问语气自然像聊天。
    """
    print("\n十、轻量问题不被「清点式声明」压场")
    from app import framework as FW
    from app.safety import scripts as SG
    ok = True

    # ① 框架渲染：缺口清单只内部掌握，且截断到追问引擎同一上限（3 条）
    plan = FW.Plan(category=FW.CAT_ADVICE, dimensions=list(FW.ACTION_DIMS),
                   need_info=["medication", "chronic", "age_sex",
                              "cold_heat", "stool_urine"])
    txt = FW.render(plan)
    good = "严禁向用户声明或罗列" in txt
    ok &= good
    print(f"  {'✅' if good else '❌'} 缺口清单标为「内部掌握」，不许向用户声明或罗列")
    good = ("cold_heat" not in txt and "stool_urine" not in txt
            and "medication" in txt)
    ok &= good
    print(f"  {'✅' if good else '❌'} 缺口渲染截断到 3 条（5 项只列前 3）")

    # ② 禁清点式开场 + 库外标注去重，写进框架提示词
    good = ("开场直接回答问题本身" in txt and "清点" in txt
            and "最多出现一次" in txt)
    ok &= good
    print(f"  {'✅' if good else '❌'} 框架层：禁清点式开场 + 库外标注全文最多一次")

    # ③ 问答系统提示词同步（QA 路径的主提示词也要有同一约束）
    from app.rag import ANSWER_SYSTEM
    good = ("清点" in ANSWER_SYSTEM and "最多出现一次" in ANSWER_SYSTEM
            and "开场直接回答问题本身" in ANSWER_SYSTEM)
    ok &= good
    print(f"  {'✅' if good else '❌'} ANSWER_SYSTEM：反清点开场 + 标注去重双规则在位")

    # ④ 追问块：禁止「等你补充后再判断」式临床收尾，语气自然
    items = [{"key": k, "label": SG.GAP_ITEMS[k]["label"],
              "ask": SG.GAP_ITEMS[k]["ask"], "subject": "",
              "signals": list(SG.GAP_ITEMS[k].get("signals") or ())}
             for k in ("medication", "cold_heat")]
    blk = IQ.followup_block(items)
    good = ("临床式声明" in blk and "清点" in blk and "不像问诊表" in blk)
    ok &= good
    print(f"  {'✅' if good else '❌'} 追问块：禁临床式收尾声明、语气像聊天不像问诊表")
    return ok


def main() -> int:
    # 用两个**真实会话**跑（时间线表对 conversations 有外键约束），
    # 结束把它们删掉——会话删除会连带清掉它的档案与记忆，
    # 所以测试不会在库里留下任何痕迹（这是「记忆按会话隔离」的附带好处）。
    ca = storage.create_conversation("追问测试A")["id"]
    cb = storage.create_conversation("追问测试B")["id"]
    cc = storage.create_conversation("追问测试C")["id"]
    try:
        r = [test_intent(), test_knowledge_not_asked(cb), test_advice_asks(ca),
             test_session_aware(ca), test_ensure_questions(), test_recall_gate(),
             test_subject_aware(cc), test_output_hygiene(), test_light_tone()]
    finally:
        storage.delete_conversation(ca)
        storage.delete_conversation(cb)
        storage.delete_conversation(cc)
    print("\n" + "=" * 60)
    print("追问引擎回归：", "全部通过" if all(r) else "存在失败")
    print(f"（测试会话 {ca}/{cb} 及其档案/记忆已删除）")
    return 0 if all(r) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
