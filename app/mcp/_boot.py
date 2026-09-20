# -*- coding: utf-8 -*-
"""MCP 入口的启动引导：**把工作目录切到项目根**，并把日志钉死在 stderr。

为什么必须要有这个文件（2026-09-17 实测结论，不是推测）：
    chroma 开库**必须走相对路径**（见 `app/paths.py`：中文绝对路径会让
    HNSW 索引报 "Error loading hnsw index"），所以进程的工作目录必须是项目根。
    而 MCP 是宿主（WorkBuddy / Claude Desktop / Cursor…）**按配置拉起子进程**的，
    启动目录由宿主决定。实测 WorkBuddy 的 `mcpServers` 配置项里
    **没有 `cwd` 字段**（只有 command / args / env / runtime / staticEnv）
    —— 也就是说「在配置里写 cwd」这条路在新宿主上**根本走不通**。

    结论：**不依赖宿主的实现细节**，自己在入口第一行把 cwd 摆正。
    （支持的宿主仍可在配置里再写一遍 `cwd` 作双保险，但不能只靠它。）

第二个职责是日志纪律：stdio 模式下 **stdout 是 JSON-RPC 协议流**，
一个 `print` 就会污染它，客户端报的是「JSON 解析错误」——
现象离原因很远，极难查。所以本模块的 `log()` 一律写 stderr，
新代码请用它而不是 `print`。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# app/mcp/_boot.py → app/mcp → app → 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]

_booted = False


def ensure_project_root(verbose: bool = False) -> Path:
    """切到项目根并把项目根放进 sys.path（幂等）。"""
    global _booted
    if not _booted:
        os.chdir(PROJECT_ROOT)
        root = str(PROJECT_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        _booted = True
    if verbose:
        log(f"[mcp] 工作目录 = {PROJECT_ROOT}")
    return PROJECT_ROOT


def log(*args) -> None:
    """所有诊断输出走 **stderr**（stdout 留给协议流）。"""
    print(*args, file=sys.stderr, flush=True)
