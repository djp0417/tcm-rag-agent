# -*- coding: utf-8 -*-
"""记忆隔离回归（2026-09-15 上线清单②）。

验收场景（用户清单原文）：
  A. **反向切换**：先以"女儿"人设咨询，再切回"我"——档案里父亲的年龄/
     慢病不得被女儿的年龄/症状污染，且回答侧必须触发「咨询对象切换确认」；
  B. **新开会话**：新会话里时间线为空、跨轮材料为空——旧会话的
     人设细节（黄腻苔、25 岁）不会糊到新会话脸上；
  C. **主体标注**：替家人问出的事实落库时 subject=third_party，
     注入 prompt 时显式标注「关于家人」，且提示词层面禁止套用到本人。

用法：python -m tools.test_memory_isolation
"""
from __future__ import annotations

from app import intake, storage
from app.memory import MemoryManager, _subject_of


def _cleanup(conv_ids: list[int], mem_ids: list[int]) -> None:
    """删掉测试会话（会话删除会连带清掉它的会话档案与 session 记忆），
    再删掉测试期间新增的 global 记忆。**不再快照/还原全局档案**——
    2026-09-16 起档案按会话存，测试根本碰不到真实数据。"""
    for cid in conv_ids:
        try:
            storage.delete_conversation(cid)
        except Exception:
            pass
    for mid in mem_ids:
        try:
            storage.delete_memory(mid)
        except Exception:
            pass


def test_reverse_switch(conv_a: int) -> bool:
    """场景 A：女儿 → 我 的反向切换。"""
    print("一、反向切换（女儿 → 我）")
    ok = True

    # 第 1 轮：替女儿问——亲属分句不得写进本人档案
    intake.update_from_message(
        "我女儿25岁，口苦、苔黄腻，平时爱吃辛辣，她能喝红豆薏米茶吗", conv_a)
    prof = storage.get_profile(conv_a)
    good = prof.get(intake.F_AGE) != "25"
    ok &= good
    print(f"  {'✅' if good else '❌'} 「女儿25岁」未写进本人档案年龄"
          f"（当前档案年龄：{prof.get(intake.F_AGE, '（空）')}）")

    # 第 2 轮：切回本人——本人档案应正确建立，女儿的湿热不得混入
    intake.update_from_message("切回我自己：今年45岁，有高血压，在吃氨氯地平", conv_a)
    prof = storage.get_profile(conv_a)
    good = prof.get(intake.F_AGE) == "45"
    ok &= good
    print(f"  {'✅' if good else '❌'} 本人年龄正确写入 45（实际："
          f"{prof.get(intake.F_AGE, '（空）')}）")
    chronic = prof.get(intake.F_CHRONIC, "")
    good = "高血压" in chronic and "湿热" not in chronic
    ok &= good
    print(f"  {'✅' if good else '❌'} 本人慢病={chronic}（含高血压、无湿热混入）")

    # 第 3 轮：回答侧材料——必须出现「咨询对象切换确认」
    brief = intake.cross_turn_brief(conv_a, "我最近总是畏寒怕冷，手脚冰凉")
    good = "咨询对象疑似切换" in brief
    ok &= good
    print(f"  {'✅' if good else '❌'} 跨轮材料含「咨询对象切换确认」"
          f"（{'命中' if good else '未命中，材料如下↓'}）")
    if not good:
        print("  ---- " + brief[:400].replace("\n", "\n  ---- "))
    return ok


def test_fresh_session(conv_a: int, conv_b: int) -> bool:
    """场景 B：新开会话不串——**档案/时间线/记忆三者都必须为空**。"""
    print("\n二、新会话隔离（2026-09-16 强化：连档案也必须是空的）")
    ok = True
    good = storage.get_timeline(conv_b) == ""
    ok &= good
    print(f"  {'✅' if good else '❌'} 新会话时间线为空")

    prof = storage.get_profile(conv_b)
    good = not prof
    ok &= good
    print(f"  {'✅' if good else '❌'} 新会话档案为空（实际 keys={list(prof)}）")

    # 会话间互不可见：A 里记的东西，B 一点也看不到
    # （"重新开的对话应该没有记忆"的核心断言）
    prof_a = storage.get_profile(conv_a)
    good = bool(prof_a) and not prof
    ok &= good
    print(f"  {'✅' if good else '❌'} 会话 A 的档案（{len(prof_a)} 项："
          f"{list(prof_a)}）在会话 B 完全不可见")

    ctx = MemoryManager().build_context(conv_b, "我最近老是累")
    good = not ctx.profile_text
    ok &= good
    print(f"  {'✅' if good else '❌'} 注入提示词的档案为空"
          f"（profile_keys={ctx.stats.get('profile_keys')}）")

    # 跨会话记忆：默认（用户没问起往事）一条都不带
    good = ctx.memories == [] and ctx.stats.get("recall_past") is False
    ok &= good
    print(f"  {'✅' if good else '❌'} 默认不注入任何跨会话记忆"
          f"（recall_past={ctx.stats.get('recall_past')}，条数={len(ctx.memories)}）")

    # 用户主动问起往事 → recall_past 打开（真正的"允许回忆"信号）
    ctx2 = MemoryManager().build_context(conv_b, "我上次跟你说的那些，阿胶还能吃吗")
    good = ctx2.stats.get("recall_past") is True
    ok &= good
    print(f"  {'✅' if good else '❌'} 「我上次跟你说的…」触发显式回忆通道"
          f"（recall_past={ctx2.stats.get('recall_past')}）")

    brief = intake.cross_turn_brief(conv_b, "我上次说的那些，阿胶还能吃吗")
    good = brief == ""
    ok &= good
    print(f"  {'✅' if good else '❌'} 新会话跨轮材料为空（不足两轮不整合，"
          f"旧会话原话不得出现）")

    # 新会话里"阿胶还能吃吗"的命中只来自本轮文本，与旧人设无关
    from app.safety import scan_text
    hits = [h.name for h in scan_text("我上次说的那些，阿胶还能吃吗")]
    good = hits == ["阿胶"]
    ok &= good
    print(f"  {'✅' if good else '❌'} 本轮扫描只命中本轮提到的阿胶（{hits}）")
    return ok


def test_subject_labeling() -> bool:
    """场景 C：主体判定与家人标注。"""
    print("\n三、记忆主体判定与家人标注")
    ok = True
    good = _subject_of("女儿25岁，湿热体质，爱吃辛辣") == "third_party"
    ok &= good
    good &= _subject_of("长期熬夜到凌晨两点，常年坐办公室") == "self"
    ok &= good
    print(f"  {'✅' if good else '❌'} _subject_of 区分家人/本人")

    m = {"text": "女儿25岁，湿热体质，爱吃辛辣",
         "created_at": 0, "subject": "third_party"}
    label = MemoryManager._label(m)
    good = "关于家人" in label and "非用户本人情况" in label
    ok &= good
    print(f"  {'✅' if good else '❌'} 家人条目标注：{label}")

    ctx = MemoryContextStub()
    block = ctx.block(["女儿25岁，湿热体质（关于家人，记录于 2026-09-15；"
                       "非用户本人情况）"])
    good = "严禁" in block and "套用" in block
    ok &= good
    print(f"  {'✅' if good else '❌'} prompt 区块含「严禁套用到本人」守卫语")
    return ok


class MemoryContextStub:
    """绕开 LLM/向量链路，只验证 as_prompt_block 的文案。"""
    @staticmethod
    def block(memories: list[str]) -> str:
        from app.memory import MemoryContext
        ctx = MemoryContext(memories=memories)
        return ctx.as_prompt_block()


def main() -> int:
    conv_ids: list[int] = []
    mem_ids: list[int] = []
    try:
        conv_a = storage.create_conversation("隔离测试A")["id"]
        conv_b = storage.create_conversation("隔离测试B")["id"]
        conv_ids = [conv_a, conv_b]
        a = test_reverse_switch(conv_a)
        b = test_fresh_session(conv_a, conv_b)
        c = test_subject_labeling()
    finally:
        _cleanup(conv_ids, mem_ids)
    print("\n" + "=" * 60)
    print("记忆隔离回归：", "全部通过" if (a and b and c) else "存在失败")
    print("（测试会话及其档案/记忆已删除，未触碰任何真实数据）")
    return 0 if (a and b and c) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
