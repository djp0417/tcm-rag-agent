# -*- coding: utf-8 -*-
"""组装 MCP Server：注册工具与资源。

这一层刻意做得极薄 —— 逻辑都在 `tools.py` / `resources.py`，文案都在 `schemas.py`，
本文件只负责"把契约接到协议上"。好处是：改文案不碰逻辑，改逻辑不碰协议。

【启动顺序很重要】`_boot.ensure_project_root()` 必须在导入任何业务模块之前执行
（chroma 开库依赖相对路径，见 `_boot.py` 的说明）。
"""
from __future__ import annotations

# ① 先摆正工作目录，再导入业务模块
from app.mcp import _boot

_boot.ensure_project_root()

# ② 协议层
from mcp.server.mcpserver import MCPServer  # noqa: E402

# ③ 本项目
from app.mcp import resources as RS  # noqa: E402
from app.mcp import schemas as S  # noqa: E402
from app.mcp import tools as TL  # noqa: E402

mcp = MCPServer(
    name=S.SERVER_NAME,
    title=S.SERVER_TITLE,
    instructions=S.SERVER_INSTRUCTIONS,
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Tools —— P0 是 tcm_safety_check：纯确定性、零大模型成本、毫秒级，
# 最适合被任何客户端当成"动手前的安全检查"来调。
# ---------------------------------------------------------------------------
mcp.tool(name="tcm_search",
         description=S.TOOL_TCM_SEARCH)(TL.tcm_search)
mcp.tool(name="tcm_safety_check",
         description=S.TOOL_TCM_SAFETY_CHECK)(TL.tcm_safety_check)
mcp.tool(name="tcm_constitution",
         description=S.TOOL_TCM_CONSTITUTION)(TL.tcm_constitution)
mcp.tool(name="tcm_intake_gaps",
         description=S.TOOL_TCM_INTAKE_GAPS)(TL.tcm_intake_gaps)

# ---------------------------------------------------------------------------
# Resources —— 只读，不产生动作
# ---------------------------------------------------------------------------
mcp.resource("tcm://kb/manifest", name="语料清单",
             description=S.RES_KB_MANIFEST)(RS.kb_manifest)
mcp.resource("tcm://guide/scope", name="能力边界",
             description=S.RES_GUIDE_SCOPE)(RS.guide_scope)
mcp.resource("tcm://session/{conv_id}/profile", name="会话档案",
             description=S.RES_SESSION_PROFILE)(RS.session_profile)

__all__ = ["mcp"]
