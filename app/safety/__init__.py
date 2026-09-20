# -*- coding: utf-8 -*-
"""安全层：独立于 RAG 的硬编码安全规则库 + 强制话术模板。

分四块：
    rules.py    规则数据（毒性中药 / 慢病西药冲突 / 特殊人群 / 食药同源）
    scan.py     扫描器（文本与档案 → SafetyHit）+ 元信息过滤 + 强制区块渲染
    tiers.py    分级引擎（五档处置：明确禁止 / 需专业确认 / 不建议(无适应症)
                / 对证但有条件 / 可执行）
    scripts.py  话术模板（引用判读 / 叫停用药 / 抗施压 / 排查追问 / 分层结论）

对外的稳定入口就下面这些，调用方不必关心内部结构：

    from app.safety import scan_text, scan_profile, merge_hits, build_block
    from app.safety import active_tags, needs_medication_stop
    from app.safety import scripts
    from app.safety import strip_meta_info
"""
from app.safety.scan import (SafetyHit, active_tags, annotate_origins,
                             build_block, has_meta_info, merge_hits,
                             needs_medication_stop, scan_profile, scan_text,
                             strip_meta_info, tag_from_text)
from app.safety import rules, scripts, tiers


__all__ = [
    "SafetyHit", "scan_text", "scan_profile", "merge_hits", "build_block",
    "annotate_origins",
    "active_tags", "needs_medication_stop", "tag_from_text",
    "strip_meta_info", "has_meta_info", "rules", "scripts", "tiers",
]
