# -*- coding: utf-8 -*-
"""命中来源分流（annotate_origins）的回归。

背景（真实事故）：端到端测试把「附子理中丸 / 氨氯地平 / 阿胶」写进了
**全局**长期档案；用户 afterwards 新开会话只问「累、想补补」，
档案扫描无差别命中 → 六张判读卡糊脸，体验是"突兀、答非所问"。
修复 = 来源标记 + 分流渲染：本轮提到的完整展开，档案带出的一行带过。

运行：python -m tools.test_origin_gating
"""
from __future__ import annotations

from app.safety import (annotate_origins, build_block, scan_profile,
                        scan_text)

MSG = "我最近老是觉得累，睡够了也没精神，是不是虚啊？该吃点啥补补？"

PROFILE = {
    "慢病": "高血压",
    "西药": "氨氯地平（降压药）",
    "在服中药与食疗": "附子、薏米（薏苡仁）、桂圆（龙眼肉）、生姜红枣茶",
}


def _names(hits, origin=None):
    return sorted(h.name for h in hits if origin is None or h.origin == origin)


def case_clean_profile():
    """干净档案 + 只问累 → 不应产生任何命中（此前是 6 项糊脸）。"""
    hits = annotate_origins(scan_text(MSG), scan_profile({}), MSG)
    assert not hits, f"干净档案不应命中，实际：{_names(hits)}"
    print("✅ 干净档案 + 只问累 → 0 命中（不再糊脸）")


def case_profile_compact():
    """真实档案 + 只问累 → 命中都来自档案，且渲染为一行式提醒。"""
    hits = annotate_origins(scan_text(MSG), scan_profile(PROFILE), MSG)
    assert hits, "真实档案应有命中"
    assert all(h.origin == "profile" for h in hits), \
        f"本轮没提任何药，不应有 message 来源：{[(h.name, h.origin) for h in hits]}"
    assert all(h.taking for h in hits if h.kind in ("herb", "drug")), \
        "档案里的药材/西药应视为在服"
    block = build_block(hits)
    assert "【档案提醒" in block, "档案命中应渲染成一行式提醒段"
    assert "1. 【" not in block, "档案命中不应展开成完整条目"
    print(f"✅ 真实档案 + 只问累 → {len(hits)} 项全部为档案提醒（一行式），无整卡")


def case_mentioned_this_turn():
    """档案里有薏米，本轮也提到薏米 → 升级为完整卡。"""
    text = MSG + "另外我天天喝红豆薏米茶行不行？"
    hits = annotate_origins(scan_text(text), scan_profile(PROFILE), text)
    ym = [h for h in hits if "薏米" in h.name]
    assert ym and ym[0].origin == "message", "本轮提到的应升级为 message 来源"
    others = [h for h in hits if h.origin == "profile"]
    assert others, "其余档案命中仍应保持 profile 来源"
    block = build_block(hits)
    assert "1. 【" in block and "【档案提醒" in block, "两种渲染段应同时存在"
    print(f"✅ 本轮提到薏米 → 升级完整卡；其余 {len(others)} 项保持一行提醒")


def case_alias_upgrade():
    """档案写「薏米（薏苡仁）」，本轮说「红豆薏米茶」（茶名不含全名但含别名）→ 升级。"""
    text = "红豆薏米茶能天天喝吗？"
    hits = annotate_origins(scan_text(text), scan_profile(PROFILE), text)
    ym = [h for h in hits if "薏米" in h.name]
    assert ym and ym[0].origin == "message", \
        f"别名应触发升级，实际 origin={ym[0].origin if ym else '未命中'}"
    print("✅ 别名（薏米⊂红豆薏米茶）也能触发升级为完整卡")


def case_merge_keeps_message():
    """同 key 合并时 origin 保留 message 优先。"""
    prof = {"在服中药与食疗": "薏米（薏苡仁）"}
    text = "我天天喝薏米水"
    hits = annotate_origins(scan_text(text), scan_profile(prof), text)
    ym = [h for h in hits if "薏米" in h.name]
    assert len(ym) == 1 and ym[0].origin == "message", "合并后应保留 message"
    assert ym[0].taking, "合并后 taking 应保留 True"
    print("✅ 同一命中合并：origin=message 优先，taking 保留")


def main() -> int:
    case_clean_profile()
    case_profile_compact()
    case_mentioned_this_turn()
    case_alias_upgrade()
    case_merge_keeps_message()
    print("\n来源分流回归：5/5 通过")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
