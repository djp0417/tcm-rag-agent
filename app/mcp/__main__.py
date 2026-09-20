# -*- coding: utf-8 -*-
"""MCP Server 命令行入口。

    python -m app.mcp                      # stdio（本地客户端：WorkBuddy / Claude Desktop / Cursor）
    python -m app.mcp --http               # Streamable HTTP，默认 127.0.0.1:7863（第二阶段）
    python -m app.mcp --http --port 7863   # 指定端口

【为什么 stdio 是默认】本地单用户场景下 stdio 最省事：没有端口、没有鉴权、
由宿主按配置拉起并长期复用同一条连接。HTTP 模式留给"远程 / 多客户端"场景。

【日志纪律】stdio 模式下 **stdout 是 JSON-RPC 协议流**，所以任何诊断输出都必须走
stderr（本项目统一用 `_boot.log`）。日志若跑到 stdout，客户端报的是
"JSON 解析错误"——现象离原因很远，极难查。
"""
from __future__ import annotations

import sys

from app.mcp import _boot

_boot.ensure_project_root(verbose=True)

from app.mcp.server import mcp  # noqa: E402

_DEFAULT_PORT = 7863


def _opt(argv: list[str], flag: str) -> str | None:
    """取 `--flag value` 形式的值（没有返回 None）。"""
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv):
            return argv[i + 1]
    return None


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if "--http" in args:
        host = _opt(args, "--host") or "127.0.0.1"   # 默认只绑本机
        port = int(_opt(args, "--port") or _DEFAULT_PORT)
        _boot.log(f"[mcp] Streamable HTTP 模式：http://{host}:{port}/mcp")
        if host not in ("127.0.0.1", "localhost"):
            # 提醒而不是阻止：绑非本机地址等于把这个能力和 API Key 一起暴露出去
            _boot.log("[mcp][警告] 绑定到非本机地址，请确认已加鉴权与访问控制。")
        mcp.run("streamable-http", host=host, port=port)
    else:
        _boot.log("[mcp] stdio 模式已启动（stdout 为协议流，日志一律走 stderr）")
        mcp.run("stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
