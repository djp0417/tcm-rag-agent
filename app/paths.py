# -*- coding: utf-8 -*-
"""路径工具：绕过 chromadb 1.5.9 在 Windows 中文路径下的 HNSW 加载 bug。

⚠️ 第三条铁律（2026-09-09 实测定位）：
chromadb 1.5.9 的 Rust 底层（bindings）在 Windows 上无法从【绝对路径含非
ASCII 字符】的目录加载 HNSW 索引，查询必报：
    Error sending backfill request to compactor: ... Error loading hnsw index
而【相对路径】不受影响——Rust 按进程工作目录自行解析，绕过了该 bug。
对照实验（同一进程、同一个库）：
    绝对路径 D:\\24控制科学与工程\\...\\store\\chroma  → 必失败
    绝对路径 D:/24控制科学与工程/.../store/chroma     → 必失败（与斜杠无关）
    相对路径 store/chroma、store\\chroma、./store/chroma → 全部成功

本项目路径含中文，因此所有打开 chroma 库的地方（建库/查询）都必须通过
chroma_store_path() 拿路径：优先返回相对项目根目录的相对路径；若当前
工作目录不在项目根附近导致相对路径仍含非 ASCII 字符，则把工作目录切到
项目根目录后再返回（python -m app.xxx 本来就要求在项目根目录运行）。

写入不受此 bug 影响（绝对路径建库能正常落盘），但为一致性统一走本模块。
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
STORE_DIR = BASE_DIR / "store" / "chroma"


def chroma_store_path() -> str:
    """返回一个可被 chromadb Rust 层正常加载的库路径（相对项目根目录）。"""
    if STORE_DIR.exists():
        try:
            rel = os.path.relpath(STORE_DIR)
            if rel.isascii():
                return rel
        except ValueError:            # Windows 跨盘符时 relpath 抛异常
            pass
        os.chdir(BASE_DIR)            # 工作目录切到项目根，再取相对路径
        return os.path.relpath(STORE_DIR)
    return str(STORE_DIR)             # 不存在时返回绝对路径，由调用方提示建库
