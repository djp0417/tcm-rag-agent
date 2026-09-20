# -*- coding: utf-8 -*-
"""MCP 只读资源（阶段 1）。

资源与工具的区别（也是要跟别人讲清的一点）：
    · **Tool = 做事**（有参数、模型决定何时调用）；
    · **Resource = 读数据**（被客户端直接展示，通常是"背景知识"性质）。
所以「语料清单」「能力边界」「会话档案」应当是 Resource 而不是硬塞成 Tool——
它们不产生动作，只提供上下文。

【红线】资源内容同样是对外可见的，所以**绝不暴露内部规则原文**
（`states.note`、规则 `verdict`、具体药名清单）。这里只给"边界与清单"这类元信息。
"""
from __future__ import annotations

import json
from typing import Any

from app.mcp import _boot

_boot.ensure_project_root()

__all__ = ["kb_manifest", "guide_scope", "session_profile"]


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


# ---------------------------------------------------------------------------
# tcm://kb/manifest —— 语料清单
# ---------------------------------------------------------------------------
def kb_manifest() -> str:
    """收录了哪些书 / 多少块 / 什么时候建的 / 用什么管线建的（只读文件，不碰网络）。"""
    try:
        m = json.loads(_manifest_path().read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        _boot.log(f"[mcp:kb_manifest] 读清单失败: {type(e).__name__}: {e}")
        return _json({"ok": False, "error": f"读语料清单失败: {e}"})

    files = m.get("files") or {}
    rows = []
    for name, meta in sorted(files.items()):
        meta = meta or {}
        rows.append({"name": name,
                     "chunks": meta.get("chunks"),
                     "chars": meta.get("chars")})

    up = m.get("updated_at")
    return _json({
        "ok": True,
        "collection": "tcm_health",
        "built_at": up,
        "pipeline": m.get("pipeline"),
        "file_count": len(rows),
        "total_chunks": sum(int(r.get("chunks") or 0) for r in rows),
        "total_chars": sum(int(r.get("chars") or 0) for r in rows),
        "corpus": rows,
        "note": "检索用 tcm_search；这里只是清单，不返回正文。",
    })


def _manifest_path():
    from pathlib import Path
    return Path(_boot.PROJECT_ROOT) / "store" / "ingest_manifest.json"


# ---------------------------------------------------------------------------
# tcm://guide/scope —— 能力边界
# ---------------------------------------------------------------------------
def guide_scope() -> str:
    """能力边界说明（静态文本）。

    为什么值得做一个 Resource：模型很容易"顺手答一下"超出知识库范围的问题
    （尤其被追问时）。把边界放在一个可读资源里，客户端可以在答之前先读一次，
    比散落在每个工具的 description 里更省 token、也更不容易漏。
    """
    return """\
# 中医养生知识库 · 能力边界

## 能回答什么
- 中药 / 食材 / 保健品的**性味、宜忌与风险**（含毒性药材、峻下逐水类、活血化瘀类）；
- **特殊人群**（备孕 / 妊娠 / 哺乳 / 儿童 / 高龄）与**慢病、在服西药**情况下的用药食养冲突；
- 中医**理论问题**：脏腑、气血津液、六淫、体质等；
- 检索**典籍原文**：《灵枢》《难经》《神农本草经》《食疗本草》《饮膳正要》
  《千金方》《抱朴子》《医学衷中参西录》《寿世保元》《圆运动的古中医学》等；
- **体质辨识**（九分法量表）与分体质的调养方向。

## 不能做什么
- **不做诊断**：不给疾病诊断结论，不替代医生面诊；
- **不开方**：不给"你自己抓这几味药煮着喝"的处方级用法与剂量；
- **不解释检查单**，不做西医治疗决策；
- **不预测**疾病走向或疗效。

## 遇到这些情况，直接建议就医（不要用工具硬答）
胸痛、呼吸困难、意识改变、突然的剧烈头痛、大出血或黑便、高热不退、
持续呕吐、体重短期内明显下降、孕期腹痛或阴道出血、儿童精神萎靡。

## 输出纪律（重要）
1. `tcm_safety_check` 返回的 `tier` 是**规则库的确定性结论**，可以组织语言，
   **不得弱化**（"明确禁止"不能表述成"建议谨慎"，反之亦然）；
2. 引用典籍必须**逐字**来自 `tcm_search` 返回的 `text`，不得凭记忆补典籍原文；
3. 检索没覆盖到，就如实说"资料库里没有这方面的内容"，
   但**安全的确定性事实（毒性、禁忌、相互作用）不受语料覆盖影响，必须照常提示**；
4. 涉及个人体质的结论，要基于已确认的信息；没确认的要点，
   如实说"这一点还没确认"，不要替用户假设。
"""


# ---------------------------------------------------------------------------
# tcm://session/{conv_id}/profile —— 会话档案（只读）
# ---------------------------------------------------------------------------
def session_profile(conv_id: str) -> str:
    """某个会话已确认的档案 / 长期记忆 / 时间线（**只读**）。"""
    from app import storage
    from app import constraints as CN

    try:
        cid = int(str(conv_id).strip())
    except (TypeError, ValueError):
        return _json({"ok": False, "error": f"conv_id 必须是整数，收到 {conv_id!r}"})

    out: dict[str, Any] = {"ok": True, "conv_id": cid}
    try:
        prof = storage.get_profile(cid) or {}
        out["profile"] = prof
        # 状态识别：让客户端知道"这个人的特殊状态"，比原始字段更好用
        try:
            states = CN.detect("", prof, cid)
            out["states"] = {"labels": list(CN.labels(states) or ()),
                             "notes": [str(s.get("user_note") or "").strip()
                                       for s in (states or ())
                                       if str(s.get("user_note") or "").strip()]}
        except Exception as e:  # noqa: BLE001
            _boot.log(f"[mcp:session_profile] 状态识别失败: {e}")
            out["states"] = {}
    except Exception as e:  # noqa: BLE001
        _boot.log(f"[mcp:session_profile] 读档案失败: {e}")
        out["profile"] = {}
        out.setdefault("warnings", []).append(f"读档案失败: {e}")

    try:
        out["memories"] = storage.list_memories(limit=50, conv_id=cid)
    except Exception as e:  # noqa: BLE001
        _boot.log(f"[mcp:session_profile] 读记忆失败: {e}")
        out.setdefault("warnings", []).append(f"读记忆失败: {e}")

    try:
        out["timeline"] = storage.get_timeline(cid)
    except Exception as e:  # noqa: BLE001
        _boot.log(f"[mcp:session_profile] 读时间线失败: {e}")
        out.setdefault("warnings", []).append(f"读时间线失败: {e}")

    out["read_only"] = True
    out["note"] = ("本服务阶段 1 只读：不提供写档案的工具。"
                   "若客户端要记住信息，请由客户端自己保存。")
    return _json(out)
