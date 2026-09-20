# -*- coding: utf-8 -*-
"""图片型 PDF 的 OCR 兜底：pypdfium2 渲染页面 → RapidOCR 识别中文 → 文本。

为什么需要本模块
----------------
pypdf 只能抽取 PDF 的【文字层】。扫描件 / 影印本整页就是一张图片，文字层为空，
提取结果是空串——这类书在语料里会直接消失（大量古籍影印本正是这种情况）。
本模块把每页渲染成位图再识别文字，让影印本也能入库。

技术选型（为什么不直接用 PaddleOCR / Tesseract）
------------------------------------------------
- **pypdfium2** 负责渲染：纯 pip 安装、自带 pdfium 二进制，不需要 poppler 等
  系统级依赖（pdf2image 需要另外装 poppler，Windows 上很折腾）。
- **rapidocr_onnxruntime** 负责识别：用的是 PaddleOCR 的 PP-OCR 模型转成的
  ONNX，中文识别质量与 PaddleOCR 同级，但**不需要安装 paddlepaddle**——
  省掉数百 MB 依赖，也不会引入 paddle 与 transformers 的版本冲突。
- Tesseract 需要单独装系统程序 + 中文语言包，可移植性最差，故不选。

缓存策略（逐页断点续跑）
------------------------
OCR 很慢（200 dpi 实测约 6~8 秒/页，一本 552 页的影印本要跑近 1 小时）。
所以做了两级缓存：
  · **逐页进度文件** `<hash>_<dpi>.partial.jsonl`：每识别完一页就追加一行并 flush。
    中途 Ctrl+C、断网、关电脑，下次重跑会**从断点继续**，不白跑已完成的页。
  · **完整结果文件** `<hash>_<dpi>.txt`：全书跑完后落盘，之后直接读它，零耗时。
    （早期版本只在整本结束时写缓存，一旦中断几十页白算——这是实测踩出来的坑。）

已知局限（面试可讲，也是后续优化点）
------------------------------------
- 竖排繁体古籍的 OCR 质量明显低于横排简体：PP-OCR 以横排为训练主体，
  竖排需要按列切分后再识别，本模块暂用默认方向检测，效果一般的书建议人工校对；
- 识别结果**没有排版结构**（无标题层级），所以 OCR 出来的书只能走「纯文本切块」，
  拿不到篇章级元数据（detect 不到 `##` 时就自动降级，属预期行为）；
- dpi 的取舍：150 dpi 快约一倍，但中医罕见字（如人名「余靖」被认成「佘靖」）
  准确率下降；故默认取 200 dpi，宁可慢一点。

用法
----
    from app.ocr import ocr_pdf, ocr_available
    if ocr_available():
        text = ocr_pdf(Path("data/影印本.pdf"))
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.paths import BASE_DIR

CACHE_DIR = BASE_DIR / "store" / "ocr_cache"
DEFAULT_DPI = 200          # 200 dpi 是中文小四号字的舒适区间；调高更准但更慢

_engine = None             # RapidOCR 单例（模型加载约 1~2 秒，复用）


def ocr_available() -> tuple[bool, str]:
    """探测 OCR 依赖是否齐备。返回 (是否可用, 不可用原因)。"""
    missing = []
    for mod, pip_name in (("pypdfium2", "pypdfium2"),
                          ("rapidocr_onnxruntime", "rapidocr_onnxruntime")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pip_name)
    if missing:
        return False, ("缺少依赖 " + " / ".join(missing)
                       + "，请运行: pip install " + " ".join(missing))
    return True, ""


def _get_engine():
    """RapidOCR 懒加载单例。"""
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _engine = RapidOCR()
    return _engine


def _content_hash(f: Path) -> str:
    """按文件内容算 sha1（分块读，避免大文件占内存）。"""
    h = hashlib.sha1()
    with f.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _render_page(page, dpi: int):
    """把 pdfium 的 page 渲染成 PIL Image（RGB）。"""
    scale = dpi / 72.0                      # pdfium 以 72dpi 为基准
    bitmap = page.render(scale=scale)
    return bitmap.to_pil().convert("RGB")


def ocr_pdf(f: Path, dpi: int = DEFAULT_DPI, max_pages: int | None = None,
            verbose: bool = True, progress=None) -> str:
    """对 PDF 逐页 OCR，返回合并后的纯文本（行以换行分隔，页间空一行）。

    支持**断点续跑**：每页识别完立即追加写进度文件，中断后重跑会跳过已完成的页。

    Args:
        f:         PDF 路径
        dpi:       渲染分辨率，越高越准越慢（默认 200）
        max_pages: 只识别前 N 页（调试用），None 表示全书
        verbose:   是否打印进度
        progress:  回调 (已完成页数, 总页数) → None，供上层显示进度

    Returns:
        识别出的文本；一页都没识别出内容时返回空串。
    """
    ok, reason = ocr_available()
    if not ok:
        raise RuntimeError(reason)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # 缓存键必须带上"页数范围"：否则调试时跑 3 页（max_pages=3）的结果
    # 会被当成全书缓存命中，导致正式建库时静默只用前 3 页——这是实测踩到的坑。
    scope = f"_p{max_pages}" if max_pages else ""
    cid = f"{_content_hash(f)}_{dpi}{scope}"
    cache_file = CACHE_DIR / f"{cid}.txt"
    partial_file = CACHE_DIR / f"{cid}.partial.jsonl"

    # ① 完整缓存命中 → 直接返回（重复建库零耗时）
    if cache_file.exists():
        if verbose:
            print(f"[ocr ] {f.name} → 命中完整缓存 ({cache_file.name})")
        return cache_file.read_text(encoding="utf-8")

    # ② 读断点进度：已完成页的结果
    done: dict[int, list[str]] = {}
    if partial_file.exists():
        for ln in partial_file.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                rec = json.loads(ln)
                done[int(rec["page"])] = list(rec["lines"])
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        if done and verbose:
            print(f"[ocr ] {f.name} 发现断点，已完成 {len(done)} 页，继续剩余部分")

    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(f))
    total = len(doc)
    limit = min(total, max_pages) if max_pages else total
    todo = [i for i in range(limit) if i not in done]

    engine = _get_engine() if todo else None     # 无待办页时不必加载模型
    import numpy as np

    with partial_file.open("a", encoding="utf-8") as pf:
        for i in todo:
            img = _render_page(doc[i], dpi)
            # RapidOCR 接受 numpy 数组；PIL → numpy 避免写临时文件
            result, _elapse = engine(np.asarray(img))
            page_lines = [item[1].strip() for item in (result or []) if item[1].strip()]
            done[i] = page_lines
            # 逐页落盘：中断也不丢进度（这是"要跑一小时"的任务的必备设计）
            pf.write(json.dumps({"page": i, "lines": page_lines},
                                ensure_ascii=False) + "\n")
            pf.flush()
            if verbose:
                print(f"[ocr ] {f.name} 第 {i + 1}/{limit} 页 → {len(page_lines)} 行")
            if progress:
                progress(len(done), limit)
    doc.close()

    # ③ 按页序拼装（断点续跑时页可能乱序完成，这里统一排序）
    lines: list[str] = []
    for i in range(limit):
        lines.extend(done.get(i, []))
        lines.append("")                         # 页间空行
    text = "\n".join(lines).strip()

    # ④ 只有真正跑完全书才写「完整缓存」（局部调试结果不污染正式缓存）
    if limit == total:
        cache_file.write_text(text, encoding="utf-8")
        partial_file.unlink(missing_ok=True)     # 完整结果落盘后清掉进度文件
    return text
