# -*- coding: utf-8 -*-
"""离线建库：data/*.md/*.txt/*.pdf → 章节/段落感知切块 → bge-m3 向量化 → Chroma 持久化。

本文件有两套机制必须同时成立：**增量入库**（只处理变更文件）与
**两阶段写入**（先算向量再写库）。前者省时间省算力，后者保库不坏。

增量入库（`store/ingest_manifest.json`）
---------------------------------------
早期版本每次 `python -m app.index` 都把全部语料重新抽取 + 重新向量化——
27 份 PDF（约 1GB，含多本影印本）要跑几十分钟并烧掉几万次 embedding 调用，
而绝大多数文件根本没变。现在做**文件级增量**：

1. 每份文件记录 `(size, mtime, sha1)`；size+mtime 都没变就直接复用（不算 sha1），
   否则算 sha1 精确比对。未变更的文件**连文本抽取都跳过**——OCR 结果与文本层
   内容由文件内容唯一决定，文件没变则切块必然相同。
2. 变更/新增的文件才走「抽取 → 归一 → 切块 → 向量化」。
3. 写库时**按 source 精准替换**：`col.delete(where={"source": 文件名})` 后只
   upsert 这批文件的块，其余文件的数据原地不动。语料被删除时其块同步清除。
4. 全部无变更 → 直接返回，**0 次 API 调用**。
5. 流水线签名（`pipeline`）变了（切块参数 / 归一逻辑 / embedding 模型 /
   manifest schema）→ 自动全量重建，避免"文件没变但切法变了"导致的脏库。

⚠️ 因此**改动切块或归一逻辑后，必须把 `EXTRACTOR_VERSION` +1**，否则老块不会更新。

`--rebuild` 强制全量重建；`--report-only` 只切块出报告不写库。

⚠️ 重要工程坑（chromadb 1.5.9 / Windows，实测复现，两条铁律）：
1) 【先算向量再写库】：若在 add_texts 内部边算 embedding 边写库（langchain-chroma
   默认行为，写入流程中发起 HTTP 请求），chromadb 1.5.9 后台 HNSW 落盘线程
   受干扰，导致"向量进了 sqlite、索引 bin 文件从未落盘"——建库进程内一切
   正常，跨进程打开即报 Error loading hnsw index。
   对照实验（2026-09-13，1157 块/1024 维/余弦空间）：
       add_texts 内实时调 API  → bin 缺失，跨进程必坏；
       预计算向量后零 API 写入 → bin 齐全，跨进程正常。
   → 本脚本两阶段：先一次性算完全部向量，再零网络请求写库。
2) 【不要在 Python 里 shutil.rmtree 删库目录】：本机沙箱的安全删除机制会在
   批量删除(>50 文件)时中止进程，留下半删除的残缺目录（表现为同样的
   hnsw 加载错误）。→ 重建库时用 chroma 的 delete_collection（逻辑删除），
   或在命令行手动 rm -rf。
3) 【单批 upsert】：多批小 upsert 会触发 chroma compaction，Windows 1.5.9
   上有丢索引风险；千级规模单批直写实测稳定（约 5MB payload）。
4) 【打开库必须用相对路径】：chromadb 1.5.9 的 Rust 层在 Windows 上无法
   从含非 ASCII 字符（如中文）的绝对路径加载 HNSW 索引——建库（写入）
   不受影响，但任何新进程查询必报 Error loading hnsw index。相对路径
   由 Rust 按进程工作目录解析，不受影响。→ 统一经 app/paths.py 的
   chroma_store_path() 拿路径，不要直接传 str(STORE_DIR)。

切块策略（两档，按文档结构自动选择）：
1) 结构化文档（含 `##` 标题，如清洗后的《素问》、或提取出篇章标题的 PDF）：
   先按标题切章节 → 章节内按中文句读递归二切；每个 chunk 继承"书名/篇章"
   元数据，可溯源。
2) 纯文本文档：直接按中文语义边界（句号/问号/逗号）递归切。

PDF 支持（pypdf 抽文字层 + OCR 自动兜底）：
- 逐页提取文本，行内若命中"篇章标题"模式（如"上古天真论篇第一""第十二章"
  "第一节"等独立短行），改写为 `## 标题` —— 复用结构化切块流水线，
  PDF 语料同样获得篇章级溯源元数据；
- 【影印本 / 扫描件自动 OCR】：文字层字数低于阈值即判定为图片型 PDF，
  自动转 app/ocr.py 逐页渲染识别（pypdfium2 + RapidOCR，结果带磁盘缓存，
  重复建库不重算）。加 --no-ocr 可关闭该行为、直接跳过这类文件；
- 一页常常含多个小节、或一节跨多页，逐页切会割裂语义，所以采用
  "全本合并 → 标题改写 → 统一切块"而不是按页切块。

PDF 抽取的日志噪声（2026-09-14）
--------------------------------
pypdf 对每个字体都会发 WARNING 级日志，一本影印本能刷上千行：
- `fontTools is required to fully parse the encoding of a CFF Type1 font ...`
  → **根因是缺 fontTools**，装上即可（已列入 requirements.txt）；
- `Ignoring wrong pointing object 37 0 (offset 0)`
  → PDF 里 xref 指针损坏，pypdf 已自行容错，无行动价值。
两者都走 pypdf 的 logging（`logging.getLogger("pypdf")`），批量建库时
默认压到 ERROR 级（`--verbose` 原样放出），并在缺 fontTools 时只提示一次。

Unicode 归一（app/textfix.py，2026-09-14 由评估体系发现）：
部分中文 PDF 的字体缺标准 Unicode 表，文本层把汉字映到了「部首码位」
（`⼈` U+2F08 而非 `人`，`⻩` 而非 `黄`），实测 5 本 PDF 共 880/2597 块、
183 种字符、51260 次出现受影响。后果是这批书的召回几乎全废
（关键词串不等、embedding 词表里是罕见 token），且 LLM 读到的上下文
字形异常、幻觉概率上升。入库前统一归一，并在报告中打印修复量；
`find_artifacts()` 可用于"入库后污染残留应为 0"的自检。

建库完成后打印《语料构成报告》：逐文件块数/字符数 + 块长分布 +
元数据（book/chapter）覆盖率自检 —— 语料质量是 RAG 的上限，
把"这次到底喂进去多少、什么结构"量化出来，问题才可定位。

chunk_size 取 500 字的理由（中文科普/古籍的经验区间 400~600）：
- 太小（<200）：语义被切碎，单块信息量不足；
- 太大（>800）：一块混多个主题，embedding 表征被"稀释"，召回精度下降；
- 500 字 + 80 重叠 ≈ 中文舒适区，也匹配 reranker 的输入长度。

注意：全量重建用 chroma 的 delete_collection 逻辑删除旧集合（不动文件系统）；
若库目录本身已损坏，请在命令行手动 `rm -rf store/chroma` 后再重跑。

用法（须在项目根目录以模块方式运行）：
    python -m app.index                 # 增量：只处理新增/变更的语料
    python -m app.index --rebuild       # 全量重建（换切块参数后用）
    python -m app.index --report-only   # 只出语料报告，不写库（零 API）
    python -m app.index --no-ocr        # 跳过影印本（建库更快）
    python -m app.index --verbose       # 放出 pypdf 的字体级告警
"""
import hashlib
import json
import logging
import re
import time
from pathlib import Path

import chromadb
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.embed import get_embeddings
from app.paths import BASE_DIR, STORE_DIR, chroma_store_path
from app.textfix import describe_fix, find_artifacts, normalize_text

DATA_DIR = BASE_DIR / "data"
COLLECTION = "tcm_health"
MANIFEST_PATH = BASE_DIR / "store" / "ingest_manifest.json"

# data/ 下这些文件是**说明文档**而不是语料：它们描述语料来源与获取途径，
# 被当成语料切块会污染检索（问"语料从哪来"反而可能召回自己的说明文）。
# 单独列出来而不是换成别的扩展名——GitHub 打开 data/ 目录时会自动渲染 README.md，
# 目录"自带说明书"是想要的效果。
NON_CORPUS_NAMES = frozenset({"README.md"})

CHUNK_SIZE = 500                 # 中文单块目标长度（字符）
CHUNK_OVERLAP = 80               # 相邻块重叠，避免关键句被拦腰截断

# 入库流水线签名：以上任何一项变化都会让"文件没变"的旧块失效，
# 需要全量重建。**改切块/归一/抽取逻辑后必须手动 +1。**
SCHEMA = 1
EXTRACTOR_VERSION = 1
EMBED_MODEL = "BAAI/bge-m3"

# 判定「图片型 PDF（影印本）」的文字量阈值：总字数低于
# max(MIN_TEXT_CHARS, MIN_CHARS_PER_PAGE × 页数) 即认为没有可用文字层，转 OCR。
# 取 50 字/页 是因为文字版 PDF 实测普遍在 300~800 字/页，50 已是极保守下限，
# 既不会把正常文字版误判为影印本，又能抓住"整页只有页眉页码"的扫描件。
MIN_TEXT_CHARS = 200
MIN_CHARS_PER_PAGE = 50

# 句内递归切分：优先段落 → 换行 → 中文句末标点 → 逗号 → 词
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
)
# 标题感知切分：# → book（书名），## → chapter（篇章）
HEADER_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "book"), ("##", "chapter")],
    strip_headers=True,          # 标题不进正文，进 metadata
)


# ---------------------------------------------------------------------------
# PDF 抽取的日志噪声收敛
# ---------------------------------------------------------------------------
_PDF_LOGGERS = ("pypdf", "pypdf._reader", "pypdf.generic", "pypdf._cmap")


def configure_pdf_logging(verbose: bool = False) -> None:
    """把 pypdf 的字体级 WARNING 压到 ERROR（详见文件头「PDF 抽取的日志噪声」）。

    这些信息按字体重复打印，一本影印本能刷上千行，且没有行动价值；
    真正的解决办法是把 fontTools 装上（见 _check_fonttools）。
    `--verbose` 时恢复原级别，便于排查抽取问题。
    """
    level = logging.INFO if verbose else logging.ERROR
    for name in _PDF_LOGGERS:
        logging.getLogger(name).setLevel(level)


def _check_fonttools() -> None:
    """缺 fontTools 时只提示一次（否则 pypdf 会按字体刷几百条告警）。"""
    try:
        import fontTools  # noqa: F401
    except ImportError:
        print("[hint] 未检测到 fontTools：部分 PDF 的 CFF Type1 字体编码无法完整解析，"
              "文本层可能缺字。建议执行：pip install fonttools（相关告警已静默）")


# 导入即静音，避免任何调用路径漏掉；main() 里再按 --verbose 调整一次
configure_pdf_logging(False)


# ---------------------------------------------------------------------------
# 增量状态（ingest_manifest.json）
# ---------------------------------------------------------------------------
def _pipeline_sig() -> str:
    """入库流水线指纹：任一项变化都必须全量重建，否则新旧块会混在一起。"""
    return (f"schema{SCHEMA}|ext{EXTRACTOR_VERSION}|"
            f"chunk{CHUNK_SIZE}/{CHUNK_OVERLAP}|emb{EMBED_MODEL}|col{COLLECTION}")


def _src_key(name: str) -> str:
    """文件名 → 稳定的块 id 前缀（避免中文/书名号直接进 id）。"""
    return hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]


def _file_sha1(f: Path) -> str:
    """按文件内容算 sha1（1MB 分块读，大 PDF 不占内存）。"""
    h = hashlib.sha1()
    with f.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest() -> dict:
    """读取增量状态；文件缺失/损坏时返回空状态（视作首次建库）。

    结构：
      files    —— 与 data/ 下文件一一对应的记录（可增量的主体）
      orphans  —— 库里有块、但 data/ 下已找不到对应文件的来源。
                  这类块**默认保留**（可能是文件被误删/改名，静默丢弃语料太危险），
                  只在显式 --prune 时才清除。
    """
    if not MANIFEST_PATH.exists():
        return {"schema": SCHEMA, "pipeline": "", "files": {}, "orphans": {}}
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("files"), dict):
            data.setdefault("orphans", {})
            return data
    except (json.JSONDecodeError, OSError):
        print("[warn] ingest_manifest.json 损坏，按首次建库处理")
    return {"schema": SCHEMA, "pipeline": "", "files": {}, "orphans": {}}


def save_manifest(files: dict, orphans: dict | None = None) -> None:
    """写回增量状态（只在整个入库流程成功之后调用）。"""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(
        {"schema": SCHEMA, "pipeline": _pipeline_sig(),
         "updated_at": time.time(), "files": files, "orphans": orphans or {}},
        ensure_ascii=False, indent=2), encoding="utf-8")


def library_mtime() -> float:
    """库的写入时间（chroma.sqlite3 的 mtime，作代理）。

    必须在使用 PersistentClient 之前取——打开库可能触发 WAL checkpoint
    而刷新 mtime，那样就分不清"文件改过"还是"库刚被打开过"了。
    """
    sq = STORE_DIR / "chroma.sqlite3"
    if sq.exists():
        return sq.stat().st_mtime
    times = [f.stat().st_mtime for f in STORE_DIR.rglob("*") if f.is_file()]
    return max(times, default=0.0)


def adopt_existing_library(client, lib_mtime: float,
                           will_prune: bool = False) -> tuple[dict, dict]:
    """库里已有数据、但 manifest 缺失时，按「文件在库写入后没被改过」补建增量状态。

    这是**升级到增量版的一次性迁移**：老库里的块是好的，不该为了补一张状态表
    就把它全部重算——本机 27 份 PDF 里近 10 本是影印本，重 OCR 要好几个小时。

    判定依据（两个都满足才认）：
      1. 集合里确实存在该 source 的块；
      2. 文件 mtime <= 库写入时间 —— 即文件在建库之后没被改动过，
         库里的块就是它当前的版本。
    块长统计直接从库里的 document 现算（无需重新抽取文本）。

    Args:
        will_prune: 调用方本次是否带 --prune / --prune-only（只影响提示文案）

    Returns:
        (可认领的记录, 孤儿来源 {来源: 块数})
        孤儿 = 库里有块、但 data/ 下**确实已无同名文件**。**不自动删除**——
        可能是文件被误删或改名，静默丢掉语料比留着脏块危险得多；
        由调用方保留并在报告里提示，用户确认后再 --prune。
    """
    from collections import defaultdict

    col = client.get_collection(COLLECTION)
    data = col.get(include=["documents", "metadatas"])
    by_src: dict[str, list[int]] = defaultdict(list)
    for doc, meta in zip(data["documents"], data["metadatas"]):
        by_src[meta.get("source", "")].append(len(doc))

    # data/ 下现存的语料文件名 —— 判定孤儿的唯一、也是最保守的依据：
    # 只有「库里有块、磁盘上连同名文件都没有」才算孤儿。
    # 不能用「未被认领」来判孤儿：mtime 晚于建库时间的文件只是「待重新处理」，
    # 它还在 data/ 下，删掉就是丢数据。
    present = {f.name for f in DATA_DIR.iterdir()
               if f.is_file() and f.suffix.lower() in (".md", ".txt", ".pdf")
               and f.name not in NON_CORPUS_NAMES}

    records: dict = {}
    for f in DATA_DIR.iterdir():
        if not f.is_file() or f.suffix.lower() not in (".md", ".txt", ".pdf"):
            continue
        if f.name in NON_CORPUS_NAMES:
            continue
        lens = by_src.get(f.name)
        if not lens:
            continue
        st = f.stat()
        if st.st_mtime > lib_mtime:            # 建库之后改过 → 不认领，走重建
            print(f"[adopt] {f.name} 在库写入后被改动过，本次将重新处理")
            continue
        records[f.name] = {
            "sha1": _file_sha1(f), "size": st.st_size, "mtime": st.st_mtime,
            "how": "adopted",                   # 沿用旧库，未重新判定抽取方式
            "chars": sum(lens), "chunks": len(lens),
            "avg": round(sum(lens) / len(lens)), "max": max(lens), "min": min(lens),
            "over": sum(1 for L in lens if L > CHUNK_SIZE * 1.5),
        }

    orphans = {name: len(lens) for name, lens in by_src.items()
               if name and name not in present}
    if records or orphans:
        print(f"[adopt] 沿用已在库中的 {len(records)} 份语料"
              f"（{sum(r['chunks'] for r in records.values())} 块），本次只处理其余文件")
    if orphans:
        detail = "、".join(f"{k}（{v} 块）" for k, v in sorted(orphans.items()))
        if will_prune:
            print(f"[prune] 将清除 {len(orphans)} 个孤儿来源的块：{detail}")
        else:
            print(f"[warn] 库中有 {len(orphans)} 个来源在 data/ 下找不到对应文件：{detail}")
            print("       —— 已保留这些块不动。若确认语料已废弃，执行 "
                  "python -m app.index --prune-only 清除；"
                  "若只是挪了位置，请把文件放回 data/")
    return records, orphans


# ---------------------------------------------------------------------------
# 抽取 / 切块
# ---------------------------------------------------------------------------
def split_doc(text: str, source: str) -> list[dict]:
    """把一份文档切成 [{text, source, book, chapter}] 的字典列表。"""
    pieces: list[dict] = []
    if re.search(r"^##\s", text, re.M):
        # 结构化文档：标题切分（拿元数据）→ 章节内递归二切（控长度）
        for sec in HEADER_SPLITTER.split_text(text):
            meta = {"source": source, **sec.metadata}
            for piece in TEXT_SPLITTER.split_text(sec.page_content):
                pieces.append({"text": piece, **meta})
    else:
        # 纯文本文档：以首行（一般是 # 标题）作章节名
        first_line = next((ln for ln in text.splitlines() if ln.strip()), source)
        title = first_line.lstrip("# ").strip()
        for piece in TEXT_SPLITTER.split_text(text):
            pieces.append({"text": piece, "source": source,
                           "book": title, "chapter": title})
    return pieces


# PDF 篇章标题模式：独立成行的短标题，如"上古天真论篇第一""第十二章 体质"
# "第一节 饮食调养""九、常用食材" 等。命中则改写为 ## 标题供结构化切分。
PDF_CHAPTER_RE = re.compile(
    r"^\s*(?:"
    r"第[一二三四五六七八九十百零〇0-9]{1,4}[章节篇]"      # 第十二章 / 第三篇
    r"|[一二三四五六七八九十百零〇0-9]{1,4}[、.．]"          # 九、常用食材
    r"|\S{2,20}(?:篇|论)第[一二三四五六七八九十百零〇0-9]{1,4}"  # 咳论篇第三十八
    r")\s*\S{0,20}\s*$"
)


def _extract_text_layer(f: Path) -> tuple[str, int]:
    """用 pypdf 抽取 PDF 文字层，返回 (原始文本, 页数)。不做标题改写。"""
    from pypdf import PdfReader

    reader = PdfReader(str(f))
    lines: list[str] = []
    for page in reader.pages:
        lines.extend((page.extract_text() or "").splitlines())
        lines.append("")                      # 页与页之间留空行
    return "\n".join(lines), len(reader.pages)


def _upgrade_chapter_lines(text: str) -> str:
    """把「独立成行的短篇章标题」改写为 `## 标题`，供结构化切块复用。

    PDF 文字层与 OCR 结果都走这一步：OCR 出来的书同样只有文字流，
    靠这个规则尽力恢复章节结构；恢复不了也无妨，会自然降级为纯文本切块。
    """
    out: list[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if s and len(s) <= 25 and PDF_CHAPTER_RE.match(s):
            out.append("## " + s)
        else:
            out.append(ln)
    return "\n".join(out)


def load_pdf(f: Path, allow_ocr: bool = True) -> tuple[str, str]:
    """提取 PDF 文本，返回 (文本, 来源方式)。

    来源方式：'pdf-text'（文字层）/ 'pdf-ocr'（走 OCR）/ ''（提取失败，调用方跳过）。

    判定逻辑：先抽文字层，若「总字数 < max(200, 50 × 页数)」则判定为图片型 PDF
    （影印本/扫描件），此时自动降级 OCR。--no-ocr 时不降级、直接返回空。
    """
    body, n_pages = _extract_text_layer(f)
    threshold = max(MIN_TEXT_CHARS, MIN_CHARS_PER_PAGE * n_pages)
    if len(body.strip()) >= threshold:
        return _upgrade_chapter_lines(body), "pdf-text"

    if not allow_ocr:
        print(f"[skip] {f.name} 文字层不足（{len(body.strip())} 字 / {n_pages} 页），"
              f"已按 --no-ocr 跳过")
        return "", ""

    from app.ocr import ocr_available, ocr_pdf
    ok, reason = ocr_available()
    if not ok:
        print(f"[warn] {f.name} 疑似图片型 PDF，但无法 OCR：{reason}")
        return "", ""

    print(f"[ocr ] {f.name} 文字层不足（{len(body.strip())} 字 / {n_pages} 页），转 OCR ...")
    try:
        text = ocr_pdf(f)
    except Exception as e:                     # 单份文件失败不应中断整个建库
        print(f"[warn] {f.name} OCR 失败：{type(e).__name__}: {e}")
        return "", ""
    if not text.strip():
        print(f"[warn] {f.name} OCR 未识别出任何文字")
        return "", ""
    return _upgrade_chapter_lines(text), "pdf-ocr"


def _read_text_smart(path: Path) -> tuple[str, str]:
    """自动识别文本文件编码并读取。古籍/繁体语料常见 utf-8 / gbk / gb18030 / big5。

    返回 (文本内容, 识别到的编码名)。五个候选全部失败时抛异常，
    并附上前 100 字节的十六进制 dump 便于排查（出现这种情况往往是
    二进制文件被误放进来，或文件被 BOM 损坏）。
    """
    raw = path.read_bytes()
    # 优先按 BOM 区分（utf-8-sig 能吃 BOM；之后才是真正的 utf-8 / 中文常见编码）
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk", "big5"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "smart_read",
        raw,
        0,
        len(raw),
        f"无法识别文件 {path.name} 的编码（已尝试 utf-8-sig/utf-8/gb18030/gbk/big5）。"
        f"前 100 字节十六进制: {raw[:100].hex()}",
    )


def _file_stat(pieces: list[dict], chars: int) -> dict:
    """单个文件的块统计（复用文件时用这些数直接出报告，不必重新切块）。"""
    lens = [len(p["text"]) for p in pieces]
    if not lens:
        return {"chars": chars, "chunks": 0, "avg": 0, "max": 0, "min": 0, "over": 0}
    over = sum(1 for L in lens if L > CHUNK_SIZE * 1.5)
    return {"chars": chars, "chunks": len(pieces),
            "avg": round(sum(lens) / len(lens)),
            "max": max(lens), "min": min(lens), "over": over}


def _expected_count(records: dict, orphans: dict) -> int:
    """对账用：库中应有的块数 = 各文件记录之和 + 保留的孤儿块。"""
    return (sum(r.get("chunks", 0) for r in records.values())
            + sum(orphans.values()))


def load_all(allow_ocr: bool = True, manifest: dict | None = None,
             reuse: bool = True) -> tuple[list[dict], list[dict], dict]:
    """读取 data/ 下所有 .md / .txt / .pdf，**只对新增/变更的文件**做切块。

    文本文件自动识别编码（utf-8 / gbk / gb18030 / big5 古籍通吃）；
    PDF 先抽文字层，影印本（图片型）自动降级 OCR。
    所有文本在切块前统一过一遍 Unicode 归一（app/textfix.py），修复部分 PDF
    把汉字错映射到「部首码位」的污染（如 `⼈参` → `人参`），并统计修复量。

    Args:
        allow_ocr: 是否允许对影印本走 OCR
        manifest:  上次入库状态（None 时自行读取）
        reuse:     True=增量（未变更的文件连抽取都跳过）；False=全量重建

    Returns:
        (待向量化的块, 每份文件的统计, 执行计划)
        执行计划 {"todo": [需重建的文件名], "removed": [已删除的文件名],
                  "records": {文件名: 新的 manifest 记录}}
    """
    manifest = manifest if manifest is not None else load_manifest()
    known: dict = manifest.get("files", {})

    all_pieces: list[dict] = []
    infos: list[dict] = []
    todo: list[str] = []          # 需要重新向量化并入库的文件
    records: dict = {}            # 本次运行后的最新 manifest 记录
    files = (
        sorted(DATA_DIR.glob("*.md"))
        + sorted(DATA_DIR.glob("*.txt"))
        + sorted(DATA_DIR.glob("*.pdf"))
    )
    files = [f for f in files if f.name not in NON_CORPUS_NAMES]

    for f in files:
        name = f.name
        st = f.stat()
        rec = known.get(name)

        # ---- 未变更 → 连抽取都跳过（OCR/抽取结果由文件内容唯一决定）----
        # 先用 (size, mtime) 做廉价预判，两者都一致就不必算 sha1（约 1GB 语料
        # 全量哈希要好几秒）；任一不一致再算 sha1 精确比对。
        if reuse and rec:
            cheap_hit = rec.get("size") == st.st_size and rec.get("mtime") == st.st_mtime
            if cheap_hit or rec.get("sha1") == _file_sha1(f):
                records[name] = rec
                infos.append({**rec, "name": name, "how": rec.get("how", "?"),
                              "action": "reuse"})
                print(f"[keep] {name} → 未变更，复用 {rec.get('chunks', 0)} 块"
                      f"（跳过抽取与向量化）")
                continue

        # ---- 新增 / 变更 → 完整走一遍 ----
        action = "changed" if rec else "new"
        if f.suffix.lower() == ".pdf":
            text, how = load_pdf(f, allow_ocr=allow_ocr)
            if not text:
                if rec and not allow_ocr:
                    # --no-ocr 跳过影印本时，不要清掉上一次的 OCR 成果
                    records[name] = rec
                    infos.append({**rec, "name": name, "how": rec.get("how", "?"),
                                  "action": "reuse"})
                    print(f"[keep] {name} 已按 --no-ocr 跳过，库中保留上次结果")
                    continue
                infos.append({"name": name, "how": "skipped", "action": "skip",
                              "chunks": 0, "chars": 0, "fixed": 0,
                              "avg": 0, "max": 0, "min": 0, "over": 0})
                continue
        else:
            text, how = _read_text_smart(f)

        # Unicode 归一：修复 PDF 文本层的「部首码位」污染（见 app/textfix.py）
        n_bad = find_artifacts(text)[0]
        if n_bad:
            fixed_text = normalize_text(text)
            print(f"[fix ] {name} 修复 {n_bad} 处 Unicode 部首污染："
                  f"{describe_fix(text, fixed_text)}")
            text = fixed_text

        # 元信息过滤：剔除「我们下回分解」「详见下节」这类**写作性元信息**。
        # 它们不是知识结论，留着只会在检索后被模型照引出来（实测出现过
        # "资料里说有个简单有效的方法，但原文卖了个关子"这种回答）。
        # 注意：这里**不 bump EXTRACTOR_VERSION** —— 已在库的语料不重新抽取
        # （重嵌入上万块的代价不值得），存量数据由**取用侧**
        # （rag.py / nodes.py 的 strip_meta_info）兜住；
        # 新增与变更的文件则从入库起就是干净的。
        from app.safety import strip_meta_info

        n_meta = len(text) - len(strip_meta_info(text))
        if n_meta:
            text = strip_meta_info(text)
            print(f"[meta ] {name} 剔除 {n_meta} 字的「下回分解」类元信息")

        pieces = split_doc(text, name)
        key = _src_key(name)
        for i, p in enumerate(pieces):          # 确定性块 id：同名文件重跑 id 不变
            p["_id"] = f"{key}-{i}"
        all_pieces.extend(pieces)
        todo.append(name)

        stat = _file_stat(pieces, len(text))
        records[name] = {"sha1": _file_sha1(f), "size": st.st_size,
                         "mtime": st.st_mtime, "how": how, **stat}
        infos.append({**stat, "name": name, "how": how,
                      "fixed": n_bad, "action": action})
        print(f"[{action:<7}] {name} → {len(pieces)} 块 (来源: {how})")

    # 从 data/ 里被删掉的语料：其块要同步清除
    removed = [name for name in known if name not in records]

    if not all_pieces and not records:
        raise SystemExit(f"data/ 下没有找到 .md/.txt/.pdf 语料（目录：{DATA_DIR}）")
    return all_pieces, infos, {"todo": todo, "removed": removed, "records": records}


def report_corpus(infos: list[dict], pieces: list[dict],
                  orphans: dict | None = None) -> None:
    """打印《语料构成报告》：逐文件明细 + 块长分布 + 元数据覆盖率自检。

    复用文件的块长统计直接取 manifest 里记录的值，所以增量运行时
    这份报告依然是"全库总量"，不会因为跳过了抽取而缩水。
    """
    orphans = orphans or {}
    total_chunks = sum(it.get("chunks", 0) for it in infos)
    total_chars = sum(it.get("chars", 0) for it in infos)
    skipped = [it for it in infos if it["how"] == "skipped"]

    print("\n" + "=" * 74)
    print("语料构成报告")
    print("=" * 74)
    print(f"{'文件':<34}{'动作':<8}{'来源':<10}{'块数':>7}{'字符':>10}{'修复':>8}")
    print("-" * 74)
    for it in infos:
        name = it["name"]
        if len(name) > 32:
            name = name[:31] + "…"
        if it["how"] == "skipped":
            chunks = chars = fixed = "-"
        else:
            chunks, chars = str(it.get("chunks", 0)), str(it.get("chars", 0))
            fixed = str(it.get("fixed", 0) or 0)
        print(f"{name:<34}{it.get('action', '?'):<8}{it['how']:<10}"
              f"{chunks:>7}{chars:>10}{fixed:>8}")
    print("-" * 74)
    total_fixed = sum(it.get("fixed", 0) or 0 for it in infos)
    print(f"{'合计':<34}{'':<8}{'':<10}{total_chunks:>7}{total_chars:>10}"
          f"{total_fixed:>8}")
    if orphans:
        print(f"{'（另保留孤儿来源）':<34}{'orphan':<8}{'':<10}"
              f"{sum(orphans.values()):>7}")
        print(f"   孤儿：{'、'.join(f'{k}（{v} 块）' for k, v in sorted(orphans.items()))}")

    # 块长分布：复用文件只有聚合值（avg/max/min/over），按块数加权合并
    live = [it for it in infos if it.get("chunks")]
    if live and total_chunks:
        weighted = sum(it["chars"] for it in live) / total_chunks
        print(f"\n块长：平均 {weighted:.0f} 字，"
              f"最长 {max(it['max'] for it in live)} 字，"
              f"最短 {min(it['min'] for it in live)} 字，"
              f"超长块(>{int(CHUNK_SIZE * 1.5)} 字) "
              f"{sum(it['over'] for it in live)} 个")

    # 元数据覆盖率只能用本次真正切过的块来算（复用文件不再保留原文）
    if pieces:
        n = len(pieces)
        with_book = sum(1 for p in pieces if p.get("book"))
        with_chapter = sum(1 for p in pieces if p.get("chapter"))
        print(f"元数据（本次新切 {n} 块）：book 覆盖 {with_book}/{n} "
              f"({with_book / n:.0%})，chapter 覆盖 {with_chapter}/{n} "
              f"({with_chapter / n:.0%})")
    print(f"库内来源 {len({it['name'] for it in infos if it.get('chunks')})} 份")

    ocr_files = [it["name"] for it in infos if it["how"] == "pdf-ocr"]
    if ocr_files:
        print(f"OCR 来源 {len(ocr_files)} 份：{', '.join(ocr_files)}"
              f"（无排版结构，走纯文本切块）")
    if skipped:
        print(f"⚠️ 跳过 {len(skipped)} 份：{', '.join(it['name'] for it in skipped)}")
    print("=" * 74 + "\n")


def _embed_in_batches(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """分批向量化 + 429 限流退避。

    为什么不能一次发全部：几千块 × 每块几百字 ≈ 数百万 token，作为一个请求
    打出去必撞服务端 TPM（每分钟 token 数）配额 → openai.RateLimitError 429，
    且此时前面所有 API 花费都已消耗。所以按批切片、撞限流就指数退避重试。
    """
    from openai import RateLimitError

    emb = get_embeddings()
    out: list[list[float]] = []
    total = len(texts)
    t0 = time.time()
    for i in range(0, total, batch_size):
        chunk = texts[i:i + batch_size]
        wait = 15.0
        for attempt in range(1, 7):            # 最多 5 次退避重试
            try:
                out.extend(emb.embed_documents(chunk))
                break
            except RateLimitError:
                if attempt == 6:
                    raise                      # 重试耗尽，如实抛出（进度已打印）
                print(f"\n[embed] 第 {i // batch_size + 1} 批撞到限流（429 TPM），"
                      f"等待 {wait:.0f}s 后重试（{attempt}/5）...")
                time.sleep(wait)
                wait = min(wait * 2, 120)      # 15 → 30 → 60 → 120s 封顶
        print(f"\r[embed] 进度 {min(i + batch_size, total)}/{total} 块"
              f"（累计 {time.time() - t0:.0f}s）", end="", flush=True)
    print()
    return out


# ---- 本地向量缓存：崩溃恢复 / 重复运行不烧第二次 embedding API ----
EMBED_CACHE_PATH = STORE_DIR / "embed_cache.npz"


def _load_vector_cache() -> dict:
    import numpy as np
    if not EMBED_CACHE_PATH.exists():
        return {}
    try:
        z = np.load(EMBED_CACHE_PATH)
        return dict(zip(z["ids"].tolist(), z["vectors"]))
    except Exception:
        print("[warn] 向量缓存文件损坏，忽略后重建")
        return {}


def _save_vector_cache(cache: dict) -> None:
    import numpy as np
    ids = list(cache)
    np.savez_compressed(EMBED_CACHE_PATH,
                        ids=np.array(ids),
                        vectors=np.stack([cache[k] for k in ids]))


def _text_key(text: str) -> str:
    """块的**内容指纹**（sha1 前 12 位），用于给向量缓存加"内容维度"。

    为什么缓存键必须带内容指纹（2026-09-16 修）
    -----------------------------------------
    块 id 是 `{sha1(文件名)}-{序号}`，**只绑位置、不绑内容**。若某文件的块数
    不变而某块文字变了（例如 textfix 手工表新增一个部首映射、只改掉一个字），
    新块的 id 与上一版完全相同 —— 纯按 id 查缓存会把**上一版的旧向量**原样
    喂回 Chroma：库里存的文字是新的、向量是旧的，检索从此静默错位，不报错、
    不告警。这正是最难查的那类缺陷。
    加上内容指纹后，文字一变缓存必 miss，宁可多算一个向量。
    """
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _cache_key(piece: dict) -> str:
    """向量缓存的键：`{块id}:{内容指纹}`。"""
    return f"{piece['_id']}:{_text_key(piece['text'])}"


def _embed_with_cache(pieces: list[dict]) -> list[list[float]]:
    """优先命中本地向量缓存，只对缺失的块调 embedding API。

    缓存键是 `{块id}:{内容指纹}`：文件没变 → id 与内容都没变 → 直接复用；
    **文字变了（哪怕只改一个字）→ 内容指纹变 → 自动重算**。上次 429/写库
    崩溃这类中途失败，重跑时已算好的向量一分钱不重花。
    """
    import numpy as np

    texts = [p["text"] for p in pieces]
    keys = [_cache_key(p) for p in pieces]
    cache = _load_vector_cache()
    todo = [i for i, k in enumerate(keys) if k not in cache]
    if todo:
        print(f"[embed] 向量缓存命中 {len(keys) - len(todo)} 块，"
              f"向量化缺失的 {len(todo)} 块 ...")
        new_vecs = _embed_in_batches([texts[i] for i in todo])
        for i, v in zip(todo, new_vecs):
            cache[keys[i]] = np.asarray(v, dtype=np.float32)
        _save_vector_cache(cache)
    else:
        print(f"[embed] 全部 {len(keys)} 块命中本地向量缓存（0 次 API 调用）")
    return [cache[k].tolist() for k in keys]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="建向量库：data/*.md|txt|pdf → 切块 → 向量化 → Chroma（默认增量）")
    ap.add_argument("--no-ocr", action="store_true",
                    help="关闭 OCR（图片型 PDF 直接跳过，建库更快）")
    ap.add_argument("--report-only", action="store_true",
                    help="只切块并打印《语料构成报告》，不写库（零 API 消耗）")
    ap.add_argument("--rebuild", action="store_true",
                    help="全量重建（忽略增量状态，换切块参数后用）")
    ap.add_argument("--prune", action="store_true",
                    help="清除孤儿块（库里有块但 data/ 下已无对应文件），随完整入库流程一起做")
    ap.add_argument("--prune-only", dest="prune_only", action="store_true",
                    help="只清除孤儿块：零抽取 / 零向量化 / 零 OCR，秒级完成")
    ap.add_argument("--verbose", action="store_true",
                    help="放出 pypdf 的字体级告警（默认已静默）")
    args = ap.parse_args()

    configure_pdf_logging(args.verbose)
    _check_fonttools()

    t0 = time.time()
    # 库写入时间必须在打开库之前取（见 library_mtime 的说明）
    lib_mtime = library_mtime()

    client = chromadb.PersistentClient(path=chroma_store_path())
    col_exists = any(c.name == COLLECTION for c in client.list_collections())

    manifest = load_manifest()
    has_state = bool(manifest.get("files"))
    sig = _pipeline_sig()
    sig_changed = has_state and manifest.get("pipeline") != sig
    if sig_changed:
        print(f"[info] 入库流水线已变更，自动全量重建：\n"
              f"       旧 {manifest.get('pipeline')}\n       新 {sig}")

    # 升级迁移：没有增量状态、但库里已有数据 → 认领未改动的文件，
    # 只处理新增/变更的部分（否则会为了补一张状态表重 OCR 近 10 本影印本）
    orphans: dict = manifest.get("orphans", {})
    if not args.rebuild and not has_state and col_exists:
        adopted, orphans = adopt_existing_library(
            client, lib_mtime, will_prune=args.prune or args.prune_only)
        if adopted or orphans:
            manifest = {"schema": SCHEMA, "pipeline": sig,
                        "files": adopted, "orphans": orphans}
            has_state = True

    # ---- --prune-only：只清孤儿块，不抽取 / 不向量化 / 不 OCR ----
    # 与 --prune 的区别：--prune 挂在完整入库流程末尾（要先跑完 OCR 才轮到删除），
    # 这里则跳过其余一切，秒级完成，专门用于「确认某些语料已废弃」的场景。
    if args.prune_only:
        files = manifest.get("files", {})
        if not orphans:
            print("[done] 没有孤儿块，无需清除")
            return
        if not has_state:
            print("[warn] 库里没有可安全认领的语料，未做任何删除。"
                  "请先正常跑一次 python -m app.index")
            return
        col = client.get_collection(COLLECTION)
        removed = sorted(orphans)
        for name in removed:                    # 精准删除：只按 source 删这批
            col.delete(where={"source": name})
        left = col.count()
        expected = sum(r.get("chunks", 0) for r in files.values())
        print(f"[prune] 已清除 {len(removed)} 个孤儿来源、共 "
              f"{sum(orphans.values())} 块：{'、'.join(removed)}")
        print(f"[ok   ] 集合现有 {left} 块（预期 {expected}）→ {STORE_DIR}")
        if left == expected:
            save_manifest(files, {})            # 孤儿归零后落盘
            print(f"[ok   ] 增量状态已更新（{len(files)} 份文件，孤儿 0）")
        else:
            print("[warn] 块数与预期不一致，未更新增量状态；"
                  "建议执行 python -m app.index --rebuild 全量重建")
        print(f"[done] 耗时 {time.time() - t0:.1f}s")
        return

    full_rebuild = args.rebuild or sig_changed or not has_state
    if full_rebuild and not args.rebuild and not sig_changed:
        print("[info] 库为空或状态缺失，执行全量建库")

    pieces, infos, plan = load_all(allow_ocr=not args.no_ocr,
                                   manifest=manifest, reuse=not full_rebuild)
    if full_rebuild:
        orphans = {}                        # 集合已重建，孤儿概念不再存在
    if args.prune and orphans:
        plan["removed"] += sorted(orphans)
        print(f"[prune] 清除 {len(orphans)} 个孤儿来源的块：{', '.join(sorted(orphans))}")
        orphans = {}
    report_corpus(infos, pieces, orphans)

    if args.report_only:
        print("[done] --report-only：已出报告，未写向量库")
        return

    # ---- 无变更快速通道：一次 API 都不调 ----
    if not full_rebuild and not plan["todo"] and not plan["removed"]:
        expected = _expected_count(plan["records"], orphans)
        if plan["records"] and expected == client.get_collection(COLLECTION).count():
            if manifest.get("files") != plan["records"]:
                save_manifest(plan["records"], orphans)     # 迁移后首次落盘
        print(f"[done] 语料无变更，向量库无需更新（0 次 API 调用，"
              f"耗时 {time.time() - t0:.1f}s）")
        return

    # ---- 阶段一：一次性预计算全部向量（API 调用全部前置，写入期零网络）----
    vectors: list[list[float]] = []
    if pieces:
        vectors = _embed_with_cache(pieces)
        assert len(vectors) == len(pieces), "向量数量与块数量不一致"
        print(f"[embed] 完成，维度 {len(vectors[0])}")
    else:
        print("[embed] 无需向量化（本次只涉及删除）")

    # ---- 阶段二：零网络请求写库 ----
    if full_rebuild:
        try:                                # 逻辑删除旧集合（不动文件系统）
            client.delete_collection(COLLECTION)
        except Exception:                   # 不存在则忽略
            pass
        col = client.create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},  # bge-m3 用余弦相似度
        )
        print("[db   ] 已重建集合")
    else:
        col = client.get_collection(COLLECTION)
        # 精准替换：只删"本次要重建/已删除"的文件的旧块，其余文件原地不动
        for name in plan["todo"] + plan["removed"]:
            col.delete(where={"source": name})
        if plan["removed"]:
            print(f"[db   ] 已清除 {len(plan['removed'])} 份被删语料的块："
                  f"{', '.join(plan['removed'])}")

    if pieces:
        # chromadb 有单批 upsert 上限（本机实测 5461），全量一把梭会 InternalError
        try:
            max_bs = int(client.get_max_batch_size())
        except Exception:
            max_bs = 4096
        step = max(1, min(4096, max_bs))
        metas = [{"source": p["source"], "book": p.get("book", ""),
                  "chapter": p.get("chapter", "")} for p in pieces]
        for i in range(0, len(pieces), step):
            col.upsert(
                ids=[p["_id"] for p in pieces[i:i + step]],
                documents=[p["text"] for p in pieces[i:i + step]],
                embeddings=vectors[i:i + step],
                metadatas=metas[i:i + step],
            )
        n_batches = (len(pieces) + step - 1) // step
        print(f"[db   ] 已写入 {len(pieces)} 块（{n_batches} 批，单批上限 {step}）")

    # ---- 阶段三：落盘自检 + 与 manifest 对账 ----
    bins = sorted(f.name for f in STORE_DIR.rglob("*.bin"))
    if not bins:
        raise SystemExit("[fail] HNSW 索引文件未落盘，库不可用！"
                         "请确认按本文件注释的两条铁律执行")
    count = col.count()
    expected = _expected_count(plan["records"], orphans)
    print(f"[ok   ] 本次写入 {len(pieces)} 块，集合现有 {count} 块 → {STORE_DIR}")
    print(f"[ok   ] 索引文件落盘自检通过: {bins}")

    if count == expected:
        save_manifest(plan["records"], orphans)
        print(f"[ok   ] 增量状态已更新（{len(plan['records'])} 份文件"
              f"{f'，另保留 {len(orphans)} 个孤儿来源' if orphans else ''}）")
    else:
        print(f"[warn] 集合块数 {count} 与增量记录 {expected} 不一致，"
              f"未更新增量状态。建议执行 python -m app.index --rebuild 全量重建")
    print(f"[done] 耗时 {time.time() - t0:.1f}s（新增/变更 {len(plan['todo'])} 份，"
          f"复用 {sum(1 for it in infos if it.get('action') == 'reuse')} 份，"
          f"删除 {len(plan['removed'])} 份）")
    print("[next] 向量库已更新，重启 Web 服务（python -m app.server）才会加载新库")


if __name__ == "__main__":
    main()
