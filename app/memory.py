# -*- coding: utf-8 -*-
"""分层长期记忆：把「只手 6 轮记忆」升级成三层记忆架构。

问题从哪来
----------
早期版本把"最近 6 条消息"硬编码进 prompt。用几轮就会暴露：说到第 4 轮，
模型已经忘了第 1 轮说过的体质；聊到 20 轮，前面全部丢失。
用户在真实使用中明确反馈了这个问题——**养生咨询恰恰是长程对话**：
先聊体质，再聊饮食，再聊睡眠，最后还要把建议串起来，中间断了链条就废了。

关键认知：**这不是模型能力问题，是上下文组装策略问题**。
DeepSeek-chat 的上下文窗口有 64K token，装几十轮对话绰绰有余；
"只有 6 轮"纯粹是因为我们只往里塞了 6 条。所以解法不是换模型，
而是**重新设计"每一轮到底把什么塞进 prompt"**。

三层记忆架构
------------
    ┌─ 第 3 层 长期记忆（跨会话，永久）─────────────────────┐
    │  · 用户档案 profile：结构化 KV（主体质/兼夹/忌口/偏好）  │
    │  · 记忆条目 memories：自然语言事实，向量召回相关项        │
    ├─ 第 2 层 会话摘要（会话内，压缩留存）──────────────────┤
    │  · 早期对话被 LLM 压成「对话纪要」，一直携带               │
    ├─ 第 1 层 近期原文（会话内，滑动窗口）───────────────────┤
    │  · 最近若干轮**完整原文**，保证细节不失真                  │
    └────────────────────────────────────────────────────┘

为什么保留"近期原文"而不是全部摘要化
------------------------------------
摘要必然丢信息（"他提到晚上 12 点睡"可能被压成"睡眠偏晚"）。
而多轮追问恰恰极度依赖近期细节。所以策略是：
**近期保真（原文）+ 远期保量（摘要）+ 跨会话保值（档案与条目）**。

预算控制（用字符数近似 token：中文约 1.5 字/token）
---------------------------------------------------
    RECENT_BUDGET_CHARS = 6000   ≈ 4000 token  近期原文
    SUMMARY_BUDGET_CHARS = 2500  ≈ 1700 token  摘要纪要
    MEMORY_TOPK = 5                              长期记忆召回条数
总计约 7~8K token，只占 64K 窗口的 1/8，给"参考资料 + 回答"留足空间。
超过 RECENT_BUDGET_CHARS × 1.5 就触发压缩：把最老的一段折进摘要。

用法：
    from app.memory import MemoryManager
    m = MemoryManager()
    ctx = m.build_context(conv_id, query="那饮食上注意什么")
    ctx.summary_text / ctx.recent / ctx.profile_text / ctx.memories
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app import storage

RECENT_BUDGET_CHARS = 6000      # 近期原文预算（字符）
SUMMARY_BUDGET_CHARS = 2500     # 摘要总预算（字符）
COMPRESS_TRIGGER = 9000         # 未压缩消息总字数超过此值 → 触发压缩
RECENT_KEEP_CHARS = 5000        # 压缩时至少保留这么久的近期原文
MEMORY_TOPK = 5                 # 长期记忆召回条数（仅在条目数超过 MEMORY_ALL_IF_LESS 时生效）
MEMORY_ALL_IF_LESS = 30         # 条目数不超过此值时全量注入，不做 top-k 截断
PROFILE_MAX_CHARS = 800         # 档案注入 prompt 的上限
DUP_DISTANCE = 0.15             # 余弦距离 ≤ 此值视为"语义重复"（相似度 ≥ 0.85）

# 归一化：去空白与常见标点，用于"同一句话不同标点/空格"的硬去重
_PUNCT = "　 \t\n，。、；：！？.,;:!?\"'“”‘’()（）[]【】"


def _norm(text: str) -> str:
    return "".join(ch for ch in text if ch not in _PUNCT)


# ---------------------------------------------------------------------------
# 主体识别（2026-09-15 上线清单②：记忆隔离的根因加固）
# ---------------------------------------------------------------------------
# 事故复盘：50 岁高血压的父亲人设与 25 岁湿热的女儿人设先后测试，
# 系统把**不同的人**当成了**同一个人的档案变更**——上一轮还在追问
# 「您之前提到有糖尿病、在吃二甲双胍，现在还在吃吗」。
# 治本分三处：① 落库时判主体（本文件）；② 注入时显式标注「关于家人」
# 并禁止套用到本人；③ intake 侧不让亲属分句的事实写进本人档案。
_KIN_WORDS = ("女儿", "儿子", "母亲", "父亲", "妈妈", "爸爸", "我妈", "我爸",
              "老公", "老婆", "妻子", "丈夫", "我姐", "我妹", "我哥", "我弟",
              "爷爷", "奶奶", "外公", "外婆", "岳母", "岳父")


def _subject_of(text: str) -> str:
    """一句话的记忆主体：提到亲属 → 替家人问的（third_party）。"""
    return "third_party" if any(k in (text or "") for k in _KIN_WORDS) else "self"


# 家人条目的守卫语（无论记忆是会话内的还是用户主动问起的，都必须带上）：
# 事故——"女儿25岁湿热"被当成用户本人的情况，跨会话追问他"现在还在吃吗"。
_FAMILY_GUARD = (
    "（标注「关于家人」的条目，记录的是用户**家人**的情况——"
    "**严禁**把其中的年龄、病史、症状、体质套用到用户本人身上，"
    "也不得反过来把用户本人的情况套到家人头上。）")


# ---------------------------------------------------------------------------
# 显式回忆（2026-09-16「记忆按会话隔离」的配套闸门）
# ---------------------------------------------------------------------------
# 默认：新开的对话没有记忆（档案与记忆条目都按 conv_id 隔离）。
# 唯一允许跨会话取记忆的入口是**用户主动问起往事**。判据用窄词表，
# 宁可漏（顶多这一次想不起来）也不要宽（动不动就把旧信息翻出来，
# 那就是用户抱怨的"记忆很乱"）。
_RECALL_CUES = (
    "上次", "上回", "上一回", "之前跟你说", "之前跟你提", "以前跟你说",
    "跟你说过", "跟你提过", "我提过", "我讲过", "还记得我吗", "你记得我",
    "还记得我", "我之前说", "我早前说", "前面说过", "以前提过",
    "我们之前聊", "之前聊过", "上次聊",
)


def wants_recall(text: str) -> bool:
    """用户本轮是否在**主动索取往事**（决定要不要跨会话取记忆）。"""
    t = text or ""
    return any(c in t for c in _RECALL_CUES)


# 用户明说"要我一直记住"的判据——只有命中它，记忆才升格为跨会话（global）。
# 与 _RECALL_CUES 成对：一个管"取"（显式回忆），一个管"存"（显式长期）。
_LONGTERM_CUES = (
    "记住", "记一下", "记下来", "记下我", "帮我记", "帮我记住", "别忘了",
    "以后都", "以后一直", "长期记得", "要记得", "请记", "你记着",
    "存下来", "以后按这个",
)


def wants_longterm(text: str) -> bool:
    """用户是否明确要求**长期记住**（决定新记忆写 session 还是 global）。"""
    t = text or ""
    return any(c in t for c in _LONGTERM_CUES)


def scope_label(conv_id: int | None) -> str:
    """给人看的作用域标签（前端"记忆状态"一行用）。"""
    cid = storage.scope_of(conv_id)
    if cid == storage.SESSION_SCOPE:
        return "cli"
    return f"conv:{cid}"

# 压缩提示词：明确要求保留"会反复用到"的信息，而不是泛泛概括
COMPRESS_PROMPT = (
    "下面是养生咨询对话的一段早期记录，请压缩成简明的「对话纪要」，供后续继续"
    "对话时参考。必须保留以下信息（有则留，无则略）：\n"
    "1. 用户自述的体质、症状、身体状况；\n"
    "2. 用户的偏好与忌口、生活习惯（作息/饮食/运动）；\n"
    "3. 已经给出过的具体建议（要点即可，不必展开）；\n"
    "4. 用户提出但尚未解决的问题。\n"
    "要求：第三人称陈述用户情况；不要评价、不要建议、不要客套；"
    "条目化，控制在 300 字以内。\n\n"
    "对话记录：\n{history}\n\n"
    "对话纪要："
)

# 记忆抽取提示词：从一轮对话里挑出"以后几轮还会用到"的事实
# 注意"以后几轮"而不是"跨会话"——2026-09-16 起记忆默认只在本会话生效
# （scope='session'），只有用户明说"记住"才升格为跨会话（scope='global'）。
EXTRACT_PROMPT = (
    "从下面这轮对话中抽取**值得在本对话后续轮次继续用上**的用户信息。\n"
    "只抽取以下三类，没有就输出「无」：\n"
    "- 体质与健康状况（如：阳虚体质、畏寒、偶尔失眠）\n"
    "- 偏好与忌口（如：不爱吃羊肉、忌辛辣）\n"
    "- 生活习惯与个人信息（如：作息偏晚、在杭州、常年坐办公室）\n"
    "要求：每条一行，以「- 」开头，一句话说清，不要编号、不要解释；"
    "**不要记录寒暄、不要记录本次回答的内容**。\n\n"
    "用户：{question}\n"
    "助手：{answer}\n\n"
    "抽取结果："
)


@dataclass
class MemoryContext:
    """组装好的上下文材料（交给 rag / agent 拼进 prompt）。"""
    summary_text: str = ""              # 会话纪要（第 2 层）
    recent: list[dict] = field(default_factory=list)   # 近期原文消息（第 1 层）
    profile_text: str = ""              # 本次对话的档案（第 3 层·结构化）
    memories: list[str] = field(default_factory=list)  # 记忆条目（第 3 层）
    recall_requested: bool = False      # 用户本轮是否主动问起往事（决定跨会话是否注入）
    stats: dict = field(default_factory=dict)          # 观测数据（前端展示"记忆状态"）

    def as_prompt_block(self, recall_requested: bool = False) -> str:
        """拼成一段可直接放进 System prompt 的「本轮背景」区块。

        2026-09-16「记忆按会话隔离」后的语言约定（很重要）：
        - 档案与记忆**只来自本次对话**（除非用户主动问起往事），所以标题写
          「本次对话中了解到的情况」而不是「已知用户档案」——标题本身就
          在告诉模型"这是这次聊出来的"，从根上掐掉"这是这个人一贯的事实"
          这种误解（旧标题配旧全局档案，正是"新会话被旧信息糊脸"的来源）。
        - 跨会话条目单独成块，并明确要求用「你之前（X月X日）提到过」的口吻。
        """
        parts: list[str] = []
        if self.profile_text:
            parts.append(
                "【本次对话中了解到的情况（只包括这次对话里用户说过的；"
                "本次对话没提到的，一律视为「不知道」，不得从别处推断）】\n"
                + self.profile_text)
        if self.memories:
            if recall_requested:
                head = ("【用户主动问起的既往记录（跨会话；每条末尾是记录日期）】\n"
                        "用户这一轮明确提到了「以前/上次」，所以把下面这些给他。"
                        "引用时必须带时间限定（「你之前（X月X日）提到过…」），"
                        "**不得**当作当前状态直接陈述；与本次对话里说的不一致时，"
                        "先指出不一致并询问以哪次为准。\n")
            else:
                head = "【本次对话中记下的要点】\n"
            parts.append(head + _FAMILY_GUARD + "\n"
                         + "\n".join(f"- {m}" for m in self.memories))
        if self.summary_text:
            parts.append("【本次对话早前纪要】\n" + self.summary_text)
        return "\n\n".join(parts)


class MemoryManager:
    """分层记忆的统一入口（无状态，方法都直接在 SQLite 上操作）。"""

    # ---------- 第 1 + 2 层：会话内上下文 ----------
    def build_context(self, conv_id: int, query: str = "",
                      recall_past: bool | None = None) -> MemoryContext:
        """组装三层记忆，构成本轮 prompt 的上下文材料。

        recall_past（2026-09-16）：用户是否**主动问起**往事（"我上次说的…"）。
        传 None 时**自动判定**（`wants_recall(query)`）——默认值不做成 False，
        是因为漏传参数的后果是"用户明明问了上次的事、系统却装失忆"，
        而自动判定最差也只是把旧信息带出来（还带日期与口吻约束）。
        判定结果默认只取 `scope='session'` 的条目；
        global（用户明说"记住"的）只在 recall_past 为真时进来。
        """
        if recall_past is None:
            recall_past = wants_recall(query)
        msgs = storage.get_messages(conv_id)

        # --- 取已压缩到哪条消息为止 ---
        summaries = storage.get_summaries(conv_id)
        compressed_upto = summaries[-1]["upto_id"] if summaries else 0

        # --- 第 2 层：摘要（多层摘要按预算取最近的若干段）---
        summary_text = self._join_summaries(summaries)

        # --- 第 1 层：压缩点之后的全部消息，按预算从新到旧保留 ---
        fresh = [m for m in msgs if m["id"] > compressed_upto]
        recent: list[dict] = []
        used = 0
        for m in reversed(fresh):                    # 从最新往回装
            L = len(m["content"])
            if used + L > RECENT_BUDGET_CHARS and recent:
                break
            recent.append({"role": m["role"], "content": m["content"]})
            used += L
        recent.reverse()                             # 恢复正序

        # --- 第 3 层：本会话档案 + 本会话记忆（+ 用户主动问起时的跨会话条目）---
        ctx = MemoryContext(
            summary_text=summary_text,
            recent=recent,
            profile_text=self.profile_text(conv_id),
            memories=self.recall(query, conv_id, recall_past) if query else [],
            recall_requested=recall_past,
        )
        ctx.stats = {
            "total_messages": len(msgs),
            "compressed_upto_id": compressed_upto,
            "recent_kept": len(recent),
            "summary_chars": len(summary_text),
            "profile_keys": len(storage.get_profile(conv_id)),
            "memories_recalled": len(ctx.memories),
            "scope": scope_label(conv_id),
            "recall_past": recall_past,
        }
        return ctx

    @staticmethod
    def _join_summaries(summaries: list[dict]) -> str:
        """把多段摘要合并，超预算时保留最近的若干段（远期摘要优先级最低）。"""
        if not summaries:
            return ""
        picked: list[str] = []
        used = 0
        for s in reversed(summaries):                # 从最近往前取
            if used + len(s["text"]) > SUMMARY_BUDGET_CHARS and picked:
                break
            picked.append(s["text"])
            used += len(s["text"])
        picked.reverse()
        return "\n\n".join(picked)

    def maybe_compress(self, conv_id: int, force: bool = False) -> dict:
        """若未压缩的早期消息过长，则把最老的一段压成摘要落库。

        返回 {"compressed": bool, "upto_id": int, "chars": int}。
        这是"无限记忆"的关键：历史不再是滑出窗口就丢失，而是**降密留存**。
        """
        msgs = storage.get_messages(conv_id)
        summaries = storage.get_summaries(conv_id)
        compressed_upto = summaries[-1]["upto_id"] if summaries else 0
        pending = [m for m in msgs if m["id"] > compressed_upto]

        pending_chars = sum(len(m["content"]) for m in pending)
        if not force and pending_chars <= COMPRESS_TRIGGER:
            return {"compressed": False, "reason": "未超过压缩阈值",
                    "pending_chars": pending_chars}

        # 找出压缩边界：从最老开始累积，直到剩余部分 ≤ RECENT_KEEP_CHARS
        keep = 0
        cut_idx = len(pending)
        for i in range(len(pending) - 1, -1, -1):
            keep += len(pending[i]["content"])
            if keep > RECENT_KEEP_CHARS:
                cut_idx = i + 1
                break
        to_compress = pending[:cut_idx]
        if not to_compress:
            return {"compressed": False, "reason": "可压缩区间为空"}

        history_text = "\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}"
            for m in to_compress)
        from app.llm import get_llm
        try:
            text = get_llm().invoke(
                COMPRESS_PROMPT.format(history=history_text)).content.strip()
        except Exception as e:
            return {"compressed": False, "reason": f"摘要生成失败：{e}"}

        # 若已有更早的摘要，把旧摘要与新纪要**串接**（保持沿革，而不是二选一）
        prev = self._join_summaries(summaries)
        merged = (prev + "\n\n" + text).strip() if prev else text
        upto_id = to_compress[-1]["id"]
        storage.add_summary(conv_id, upto_id, merged)

        return {"compressed": True, "upto_id": upto_id,
                "chars": len(merged), "compressed_messages": len(to_compress)}

    # ---------- 第 3 层：本会话档案 ----------
    @staticmethod
    def set_profile(key: str, value: str, conv_id: int | None = None) -> None:
        storage.set_profile(key, value, conv_id)

    @staticmethod
    def profile_text(conv_id: int | None = None) -> str:
        """把**本会话**档案渲染成 Prompt 友好的一行行文本；超长时截断。"""
        prof = storage.get_profile(conv_id)
        if not prof:
            return ""
        lines = [f"- {k}：{v}" for k, v in prof.items()]
        text = "\n".join(lines)
        return text[:PROFILE_MAX_CHARS]

    def save_constitution(self, primary: str, tendencies: list[str],
                          scores: dict[str, float],
                          conv_id: int | None = None) -> None:
        """问诊出结果后归档到**本会话**档案。

        2026-09-16 变化：以前这里是"下次开新会话助手还记得你的体质"，
        现在**只记在本会话**——体质辨识本身就是一个连续会话里的结果，
        新开的对话不该继承（用户明确要求"重新开的对话应该没有记忆"）。
        需要跨会话携带时，由用户说一句"记住我的体质是…"走 global 通道。
        """
        detail = "、".join(f"{k}{v:.0f}分" for k, v in
                          sorted(scores.items(), key=lambda kv: -kv[1]))
        self.set_profile("体质判定", primary +
                         (f"（兼夹：{'、'.join(tendencies)}）" if tendencies else ""),
                         conv_id)
        self.set_profile("体质转化分明细", detail, conv_id)

    # ---------- 第 3 层：长期记忆条目 ----------
    # 向量索引说明（两个必须注意的工程细节）
    # ① chroma 的 id 用**数据库主键**（mem<int>），不能用列表下标：
    #    列表按下标当 id 时，一旦删掉某条记忆后面的下标全部前移，
    #    向量与文本就会错位——检索出来的"记忆"张冠李戴。这是实测发现的设计缺陷。
    # ② 写入沿用"先算向量、再零网络写入"的铁律（chromadb 1.5.9 见 app/index.py）。

    @staticmethod
    def _memory_collection():
        import chromadb
        from app.paths import chroma_store_path
        client = chromadb.PersistentClient(path=chroma_store_path())
        return client.get_or_create_collection(
            name="user_memory", metadata={"hnsw:space": "cosine"})

    def _sync_index(self, memories: list[dict]):
        """把尚未入索引的记忆补进向量集合，返回集合对象。"""
        from app.embed import get_embeddings
        col = self._memory_collection()
        have = set(col.get(include=[]).get("ids", []))
        todo = [m for m in memories if f"mem{m['id']}" not in have]
        if todo:
            vecs = get_embeddings().embed_documents([m["text"] for m in todo])
            col.upsert(ids=[f"mem{m['id']}" for m in todo],
                       documents=[m["text"] for m in todo], embeddings=vecs)
        return col

    def recall(self, query: str, conv_id: int | None = None,
               include_global: bool | None = None,
               topk: int = MEMORY_TOPK) -> list[str]:
        """召回**本会话**的记忆条目；跨会话条目只在用户主动问起时才带上。

        2026-09-16「记忆按会话隔离」后的两道闸门：
        ① **作用域闸门**（本函数）：默认只取 `scope='session' AND conv_id=本会话`。
           跨会话的 global 条目只有两种情况下才进提示词——
           `include_global=True`（用户这轮说了"我上次说的…"，由调用方判定）
           或显式传入；否则**新开的对话一条旧记忆都拿不到**。
        ② **方向一致性门控**（`_consistent_memories`，沿用 2026-09-15 的修复）：
           与当前主诉病机方向相反的旧条目不注入（当前畏寒 vs 旧记忆潮热盗汗）。

        为什么小集合不做 top-k 截断（2026-09-14 由评估发现）：
        长期记忆天然是小集合，按 embedding 相似度截断成 5 条**只会引入抖动**
        （同一句问法有时能回忆起"长期熬夜"，有时不能）。条目数 ≤ 30 时全量注入。
        """
        if include_global is None:
            include_global = wants_recall(query)
        if include_global:
            allm = storage.list_memories(limit=500, conv_id=conv_id)
        else:
            allm = storage.list_memories(limit=500, conv_id=conv_id,
                                         scope=storage.MEM_SESSION)
        if not allm:
            return []
        picked = self._consistent_memories(allm, query)
        if len(picked) <= max(topk, MEMORY_ALL_IF_LESS):
            return [self._label(m) for m in picked]
        try:
            from app.embed import get_embeddings
            col = self._sync_index(allm)
            qv = get_embeddings().embed_query(query)
            res = col.query(query_embeddings=[qv], n_results=topk)
            by_id = {m["id"]: self._label(m) for m in picked}
            out: list[str] = []
            for cid_ in (res.get("ids") or [[]])[0]:
                mid = int(str(cid_).replace("mem", ""))
                if mid in by_id:                 # 只认本次作用域内的条目
                    out.append(by_id[mid])
            return out or [self._label(m) for m in picked[:topk]]
        except Exception:
            # 退路：向量检索不可用时不阻塞对话，按时间取最近的若干条
            return [self._label(m) for m in picked[:topk]]

    # 病机方向互斥对：两个方向同时出现，说明记录之间很可能不是同一状态
    _OPPOSITE = (("cold", "yin_deficiency"), ("cold", "damp_heat"))

    @staticmethod
    def _consistent_memories(allm: list[dict], query: str) -> list[dict]:
        """过滤掉与当前主诉方向相反的旧记忆（被滤掉的不注入，不是删除）。"""
        from app.safety import tag_from_text
        cur = tag_from_text(query or "")
        if not cur:
            return allm
        out = []
        for m in allm:
            tags = tag_from_text(m.get("text", ""))
            if any((a in cur and b in tags) or (b in cur and a in tags)
                   for a, b in MemoryManager._OPPOSITE):
                continue
            out.append(m)
        return out

    @staticmethod
    def _label(m: dict) -> str:
        """给记忆条目加来源标注——模型只能「之前提到过」地引用它。

        主体分流（2026-09-15 上线清单②）：
        - self：`事实（记录于 YYYY-MM-DD）`；
        - third_party：`事实（关于家人·YYYY-MM-DD）`——家人条目必须有
          独立于本人的标注，否则模型会把"女儿25岁湿热"当成用户本人的
          档案变更，跨会话追问时张冠李戴（实测事故）。
        """
        text = m.get("text", "")
        ts = m.get("created_at")
        try:
            import time as _t
            date = _t.strftime("%Y-%m-%d", _t.localtime(float(ts)))
        except Exception:
            return text
        if m.get("subject") == "third_party":
            return f"{text}（关于家人，记录于 {date}；非用户本人情况）"
        return f"{text}（记录于 {date}）"

    @staticmethod
    def _extract(question: str, answer: str) -> list[str]:
        """调 LLM 从一轮对话里抽取长期有效的事实（纯函数，不落库）。"""
        from app.llm import get_llm
        try:
            raw = get_llm().invoke(
                EXTRACT_PROMPT.format(question=question, answer=answer)).content
        except Exception:
            return []
        out: list[str] = []
        for ln in (raw or "").splitlines():
            ln = ln.strip()
            if not ln.startswith(("-", "•", "*")):
                continue
            fact = ln.lstrip("-•* ").strip()
            if not fact or fact == "无" or len(fact) < 4 or len(fact) > 120:
                continue
            out.append(fact)
        return out

    def remember_turn(self, conv_id: int, question: str,
                      answer: str) -> list[str]:
        """从一轮对话里抽取"以后还会用到"的事实并落库；返回新增条目。

        **作用域判定**（2026-09-16）：默认写 `scope='session'`（只属于本会话，
        新开对话看不到）；只有用户在问题里**明说**"记住／记一下／以后都"，
        才升格成 `scope='global'`（唯一允许跨会话的条目）。
        这条判据写在代码里而不是提示词里——"能不能跨会话"是产品语义，
        不能让模型自由发挥（否则又回到"记忆很乱"）。

        为什么要做语义去重：模型可能同时通过 remember_fact 工具显式记忆、
        又由这里的自动抽取再记一遍，同一件事会以不同措辞存两条
        （实测：『在杭州上班』与『用户在杭州工作，常年坐办公室』并存）。
        除"原文相同"的硬去重外，还用**向量余弦相似度**做语义去重；
        去重范围限定在本次作用域内（别的会话记过同一件事，不影响本会话再记）。
        """
        cands = self._extract(question, answer)
        if not cands:
            return []
        scope = (storage.MEM_GLOBAL if wants_longterm(question)
                 else storage.MEM_SESSION)

        allm = storage.list_memories(limit=500, conv_id=conv_id, scope=scope)
        known = {_norm(m["text"]) for m in allm}
        cands = [c for c in cands if _norm(c) not in known]
        if not cands:
            return []

        added: list[str] = []
        try:
            from app.embed import get_embeddings
            col = self._sync_index(allm)
            vecs = get_embeddings().embed_documents(cands)
            for fact, vec in zip(cands, vecs):
                subj = _subject_of(fact)          # 落库前判主体（家人 ≠ 本人）
                # 语义去重：只跟**本次作用域**里已有的条目比（跨会话的存量不参与，
                # 否则新会话记同一件事会被别的会话的旧条目拦掉）
                if self._max_sim(col, vec, [m["id"] for m in allm]) >= (
                        1.0 - DUP_DISTANCE):
                    continue
                r = storage.add_memory(fact, kind="fact", conv_id=conv_id,
                                       subject=subj, scope=scope)
                if not r.get("duplicated"):
                    added.append(fact)
                    allm.append({"id": r["id"], "text": fact})  # 后续候选也去重
                    col.upsert(ids=[f"mem{r['id']}"], documents=[fact],
                               embeddings=[vec])
        except Exception:
            # 向量链路不可用 → 退回"仅原文去重"，保证记忆功能不整体失效
            for fact in cands:
                r = storage.add_memory(fact, kind="fact", conv_id=conv_id,
                                       subject=_subject_of(fact), scope=scope)
                if not r.get("duplicated"):
                    added.append(fact)
        return added

    @staticmethod
    def _max_sim(col, vec, ids: list[int]) -> float:
        """候选向量与给定记忆条目的最大余弦相似度（纯本地计算，零网络）。

        为什么不用 col.query + where：chroma 的 where 只能过滤 metadata，
        过滤不了 id；而"只在本次作用域内去重"恰恰是 id 维度的过滤。
        条目数很小（几十条），取回向量本地算余弦更准也更省事。
        """
        ids = [i for i in ids if i]
        if not ids:
            return 0.0
        try:
            got = col.get(ids=[f"mem{i}" for i in ids], include=["embeddings"])
            embs = got.get("embeddings")
            if embs is None:
                return 0.0
            nv = sum(float(x) * float(x) for x in vec) ** 0.5 or 1.0
            best = 0.0
            for e in embs:
                if e is None:
                    continue
                ne = sum(float(x) * float(x) for x in e) ** 0.5 or 1.0
                s = sum(float(a) * float(b) for a, b in zip(vec, e)) / (nv * ne)
                best = max(best, s)
            return best
        except Exception:
            return 0.0
