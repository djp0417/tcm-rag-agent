# -*- coding: utf-8 -*-
"""Unicode 归一化：修复 PDF 文本层把汉字错映射成「部首码位」的污染。

问题现象（2026-09-14 由评估体系发现）
------------------------------------
对语料做关键词检索时，"人参""大枣""独取寸口"等词【一个都查不到】，
但"上品""延年"却能命中——说明不是内容缺失，而是**字符本身不对**。
逐字符统计后发现：2597 个块里有 **880 块（33.9%）** 含非法汉字，
共 **183 种字符、51260 次出现**，全部集中在 5 本 PDF。

根因：部分中文 PDF 的字体没有标准 Unicode 编码表，文本层提取时把字形
映射到了 Unicode 的「部首区」而非「汉字区」：

    ⼈ U+2F08 KANGXI RADICAL MAN        ≠ 人 U+4EBA
    ⽓ U+2F53 KANGXI RADICAL STEAM      ≠ 气 U+6C14
    ⻝ U+2EDD CJK RADICAL EAT ONE       ≠ 食 U+98DF
    ⻩ U+2EE9 CJK RADICAL SIMPLIFIED YELLOW ≠ 黄 U+9EC4

两类污染源，需要两种修法：

  ① **康熙部首块** U+2F00–U+2FD5：这些码位自带 NFKC 兼容分解，
     `unicodedata.normalize("NFKC", …)` 可直接还原（⼈→人、⽓→气）。
  ② **CJK 部首补充块** U+2E80–U+2EFF：**没有**兼容分解，NFKC 无能为力
     （实测 ⻝ 归一化后还是 ⻝），必须手工建表映射。

为什么影响检索远不止"字打得不对"
----------------------------------
- 向量检索：`⼈`(U+2F08) 在 bge-m3 词表里是极罕见 token，与"人"的向量
  相距很远，导致 5 本书几乎召不回来；
- 关键词/BM25 类召回：字符串不相等，直接漏掉；
- 生成质量：LLM 读到的上下文是「⽯⻙ 味苦平。主劳热邪⽓」这种字形，
  字形与常用字不同，会显著抬高误读与幻觉概率。

也就是说：**这是一处语料层的静默数据缺陷，靠肉眼看原文很难发现**
（终端/浏览器里 部首字形和正字字形非常接近），只有量化评估能暴露它。

设计取舍：为什么不用全局 NFKC
------------------------------
全局 NFKC 还会把全角 ASCII、罗马数字、圈号等一并改写，对古籍正文是
不必要的副作用。这里只对「兼容/部首区」做定点归一，正文其余字符原样
保留。修复是**幂等**的：normalize_text 跑两遍结果相同。

用法：
    from app.textfix import normalize_text, find_artifacts
    clean = normalize_text(raw)          # 入库前调用
    n_chars, kinds, examples = find_artifacts(raw)   # 体检用
"""
import unicodedata

# ---------------------------------------------------------------------------
# ① CJK 部首补充块（U+2E80–U+2EFF）：无 NFKC 分解，手工映射
#    键值来自语料中出现过的 35 个字符；名字取自 unicodedata.name()，
#    "C-SIMPLIFIED" 表示该字形本身是简体部首形，直接对应简体正字。
# ---------------------------------------------------------------------------
_CJK_RADICAL_MAP = {
    "\u2e85": "亻",   # CJK RADICAL PERSON           （拆字描述："（⺅耳耳耳）"）
    "\u2e89": "刂",   # CJK RADICAL KNIFE TWO        （"左刘去⺉右⻦"）
    "\u2e8e": "兀",   # CJK RADICAL LAME ONE
    "\u2e90": "尤",   # CJK RADICAL LAME THREE       （"⺐当急服此散"→尤当急服此散）
    "\u2e92": "巳",   # CJK RADICAL SNAKE            （"⾦⽣于⺒"→金生于巳）
    "\u2e93": "糸",   # CJK RADICAL THREAD
    "\u2e9f": "母",   # CJK RADICAL MOTHER
    "\u2ea0": "民",   # CJK RADICAL CIVILIAN
    "\u2ea1": "氵",   # CJK RADICAL WATER ONE        （"左⺡右毒"）
    "\u2ebe": "艹",   # CJK RADICAL GRASS ONE        （"上⺾下左⼯右⻖"）
    "\u2ec1": "虎",   # CJK RADICAL TIGER
    "\u2ec5": "见",   # CJK RADICAL C-SIMPLIFIED SEE
    "\u2ec6": "角",   # CJK RADICAL SIMPLIFIED HORN
    "\u2ec9": "贝",   # CJK RADICAL C-SIMPLIFIED SHELL
    "\u2ecb": "车",   # CJK RADICAL C-SIMPLIFIED CART
    "\u2ed3": "长",   # CJK RADICAL C-SIMPLIFIED LONG
    "\u2ed4": "门",   # CJK RADICAL C-SIMPLIFIED GATE
    "\u2ed6": "阝",   # CJK RADICAL MOUND TWO        （"左⼯右⻖"）
    "\u2ed8": "青",   # CJK RADICAL BLUE
    "\u2ed9": "韦",   # CJK RADICAL TANNED LEATHER   （"⽯⻙"→石韦）
    "\u2eda": "页",   # CJK RADICAL C-SIMPLIFIED LEAF（"左吉右⻚"）
    "\u2edb": "风",   # CJK RADICAL C-SIMPLIFIED WIND
    "\u2edc": "飞",   # CJK RADICAL C-SIMPLIFIED FLY
    "\u2edd": "食",   # CJK RADICAL EAT ONE
    "\u2ee2": "马",   # CJK RADICAL C-SIMPLIFIED HORSE
    "\u2ee3": "骨",   # CJK RADICAL BONE
    "\u2ee4": "鬼",   # CJK RADICAL GHOST
    "\u2ee5": "鱼",   # CJK RADICAL C-SIMPLIFIED FISH
    "\u2ee6": "鸟",   # CJK RADICAL C-SIMPLIFIED BIRD
    "\u2ee7": "卤",   # CJK RADICAL C-SIMPLIFIED SALT（"⻧盐"→卤盐）
    "\u2ee8": "麦",   # CJK RADICAL SIMPLIFIED WHEAT
    "\u2ee9": "黄",   # CJK RADICAL SIMPLIFIED YELLOW
    "\u2eec": "齐",   # CJK RADICAL C-SIMPLIFIED EVEN
    "\u2eee": "齿",   # CJK RADICAL C-SIMPLIFIED TOOTH
    "\u2ef0": "龙",   # CJK RADICAL C-SIMPLIFIED DRAGON
    "\u2ef3": "龟",   # CJK RADICAL C-SIMPLIFIED TURTLE（"⻳甲"→龟甲）
}

# 需要检测/归一化的 Unicode 区段（均为"看着像汉字但不是汉字"的兼容区）
ARTIFACT_RANGES = (
    (0x2E80, 0x2EFF),   # CJK 部首补充
    (0x2F00, 0x2FDF),   # 康熙部首
    (0xF900, 0xFAFF),   # CJK 兼容汉字
    (0x2F800, 0x2FA1F),  # CJK 兼容汉字补充
)


def _build_table() -> dict[int, str]:
    """构造 str.translate 用的映射表：仅覆盖兼容/部首区，正文其余字符不动。

    康熙部首块与兼容汉字块都自带 NFKC 兼容分解，逐个算出来填表；
    CJK 部首补充块无分解，用手工表补上。
    """
    table: dict[int, str] = {}
    for lo, hi in ARTIFACT_RANGES:
        for cp in range(lo, hi + 1):
            ch = chr(cp)
            fixed = unicodedata.normalize("NFKC", ch)
            if fixed != ch:
                table[cp] = fixed
    for ch, fixed in _CJK_RADICAL_MAP.items():
        table[ord(ch)] = fixed
    return table


_TABLE = _build_table()


def normalize_text(text: str) -> str:
    """把部首/兼容码位还原成正字；幂等，可安全重复调用。"""
    if not text:
        return text
    return text.translate(_TABLE)


def find_artifacts(text: str) -> tuple[int, int, list[str]]:
    """体检：返回 (出现次数, 字符种类数, 样例字符列表)。

    入库存档时用它做"入库后污染残留"的自检断言；正常语料应返回 0。
    """
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        for lo, hi in ARTIFACT_RANGES:
            if lo <= cp <= hi:
                counts[ch] = counts.get(ch, 0) + 1
                break
    if not counts:
        return 0, 0, []
    ordered = sorted(counts.items(), key=lambda x: -x[1])
    return sum(counts.values()), len(counts), [c for c, _ in ordered[:10]]


def describe_fix(before: str, after: str, limit: int = 6) -> str:
    """给日志用：列出前几个被修掉的字符，形如 `⼈(U+2F08)→人`。"""
    seen: list[str] = []
    for a, b in zip(before, after):
        if a != b:
            item = f"{a}(U+{ord(a):04X})→{b}"
            if item not in seen:
                seen.append(item)
        if len(seen) >= limit:
            break
    return "、".join(seen) if seen else "（无）"
