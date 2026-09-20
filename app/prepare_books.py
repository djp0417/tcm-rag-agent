# -*- coding: utf-8 -*-
"""把古籍原始电子文本清洗成结构化 markdown 语料（一次性脚本，语料更新时重跑）。

处理对象：殆知阁版《黄帝内经素问》（公版古籍，无版权问题）。
  原始文件：data/suwen_dl.md（含 YAML frontmatter，无标题结构）
  输出文件：data/03_黄帝内经素问.md（每篇一章：`## 篇名` + 段落）

为什么要清洗：
- 原文 42 万字是"一整块"文本，没有 markdown 标题；
- 篇名行（如"四气调神大论篇第二"）是天然的章节边界，
  转成 `##` 标题后，index.py 就能做"章节感知切块"，
  每个 chunk 都带上篇章名元数据，检索结果可解释、可溯源。

用法：
    python -m app.prepare_books
"""
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# 章节行特征："XXX篇第X"（X 为汉字数字），如"上古天真论篇第一"
CHAPTER_RE = re.compile(r"^(.{2,25}篇第[一二三四五六七八九十百零]+)\s*$")


def clean_suwen(src: Path, dst: Path) -> int:
    """清洗素问：拆 YAML → 按篇名切章节 → 清全角空格 → 写结构化 md。"""
    text = src.read_text(encoding="utf-8")
    # 1) 去掉 YAML frontmatter（--- 包裹的元信息块）
    if text.startswith("---"):
        text = re.sub(r"\A---.*?---\s*", "", text, flags=re.S)

    chapters: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        m = CHAPTER_RE.match(line.strip())
        if m:                                  # 新的一篇开始
            chapters.append((m.group(1), []))
        elif chapters:                         # 篇内正文行
            para = line.replace("\u3000", "").strip()   # 去全角缩进空格
            if para:
                chapters[-1][1].append(para)

    if not chapters:
        raise SystemExit("未识别到任何篇名，请检查原始文件格式")

    parts = ["# 黄帝内经素问\n"]
    for title, paras in chapters:
        parts.append(f"\n## {title}\n")
        parts.append("\n\n".join(paras))
        parts.append("\n")
    dst.write_text("".join(parts), encoding="utf-8")
    return len(chapters)


def main() -> None:
    src, dst = DATA_DIR / "suwen_dl.md", DATA_DIR / "03_黄帝内经素问.md"
    if not src.exists():
        raise SystemExit(f"原始文件不存在：{src}（先运行下载命令，见 README）")
    n = clean_suwen(src, dst)
    src.unlink()                               # 删除原始文件，避免 index.py 重复入库
    print(f"[ok] 《黄帝内经素问》清洗完成：{n} 篇 → {dst.name}")
    print(f"     字符数：{dst.read_text(encoding='utf-8').__len__():,}")


if __name__ == "__main__":
    main()
