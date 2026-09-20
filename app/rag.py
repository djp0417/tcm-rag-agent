# -*- coding: utf-8 -*-
"""RAG 会话核心：三层记忆 + 问题改写 + 自反思两段式检索。

一次 ask() 的完整流程（共 3 步）：
  ① 问题改写（condense）：结合**三层记忆**，把"那饮食上注意什么？"
     这类追问改写成独立问题"阳虚体质饮食上注意什么？"——否则拿追问
     直接去检索，向量库里根本没有"那/呢"这种指代的语义，必然检索失败；
  ② 自反思检索（app/selfrag.py）：embedding 召回 k_recall=16 → rerank 精排 →
     若 top1 分数低于阈值，自动改写查询再检索一轮，取历史最优；
  ③ 带记忆生成：System(用户档案 + 长期记忆 + 对话纪要 + 资料 + 边界)
     + 近期原文 + 当前问题 → DeepSeek。

关于「记忆」的演进（本文件最重要的变化）
--------------------------------------
早期版本只把「最近 6 条消息」塞进 prompt，几轮之后就"忘了前面说过什么"。
问题不在模型窗口不够（DeepSeek-chat 有 64K token），而在**上下文组装策略**
太粗糙。现在改为三层（详见 app/memory.py）：
    近期原文（保真） + 早期纪要（保量） + 跨会话档案与长期记忆（保值）
`_build_messages()` 不再硬编码任何条数上限，而由 MemoryManager 按预算组装。

设计取舍：
- 会话历史的"事实来源"是 SQLite（server 路径）而不是内存；CLI 单进程场景
  没有 conv_id，退回内存 history——两条路径共用 `_context()` 一个出口；
- 记忆写入（抽取长期事实、压缩早期历史）放在**回答推送完之后**执行，
  用户已看到完整答案，这部分延迟不落在体感上。

用法：
    from app.rag import RAGSession
    s = RAGSession()                      # CLI：内存历史
    s = RAGSession(conv_id=3)             # Web：SQLite 三层记忆
    r1 = s.ask("阳虚体质有什么表现？")
    r2 = s.ask("那饮食上注意什么")          # 追问也能正确检索
"""
import sys
from dataclasses import dataclass, field

from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app import credibility as CR
from app.embed import get_embeddings
from app.llm import get_llm
from app.memory import MemoryContext, MemoryManager, wants_recall
from app.paths import STORE_DIR, chroma_store_path
from app.selfrag import retrieve_with_reflection

K_RECALL = 16          # 第一段：embedding 召回候选数（粗排，宁多勿漏）
K_FINAL = 4            # 第二段：rerank 后真正喂给 LLM 的块数（精排）
CLI_HISTORY = 20       # CLI（无库）模式下的内存历史条数上限
COLLECTION = "tcm_health"

CONDENSE_PROMPT = (
    "结合下面的对话历史，把用户的最新问题改写成一个不依赖上下文、"
    "可以独立理解的完整问题。\n"
    "只输出改写后的问题本身，不要任何解释或引号。\n"
    "如果最新问题本身已经完整，原样输出即可。\n\n"
    "对话历史：\n{history}\n\n最新问题：{question}\n改写后的独立问题："
)

# System prompt：用户背景（记忆）+ 资料 + 行为边界。
# 行为边界经过三轮迭代收敛：
#   规则只约束模型行为、禁止复述给用户；回答里不出现任何免责话术
#   （合规提示由前端界面常驻展示，属产品层职责）。
ANSWER_SYSTEM = (
    "你是一名中医养生科普助手。请【只依据】下面提供的【参考资料】回答问题，"
    "用通俗易懂的现代汉语组织回答，可分点；若参考资料是文言文（如《黄帝内经》），"
    "请先准确理解，再用白话转述其含义。\n"
    "若参考资料中没有相关内容，请直接说『资料库中没有这方面的内容』，不要编造。\n\n"
    "以下是对你的行为要求（注意：这些规则用来约束你的行为，"
    "【禁止】在回答中向用户复述、引用或解释这些规则原文）：\n"
    "1. 你只做养生科普，不做疾病诊断或治疗建议，也不推荐具体药物。\n"
    "2. 【回答中绝对不要出现任何免责声明、健康提醒或“请及时就医”"
    "之类的话】——合规提示由产品界面常驻展示，不在回答文本里生成。"
    "无论用户是否描述症状，你的回答只管把养生科普内容本身讲好。\n"
    "3. 资料库没有的内容就坦诚说没有，可以补充常识性建议；"
    "『这是常识性建议，资料库中无直接依据』这类标注**整篇回答最多出现一次**"
    "（在第一次补常识的地方注明即可），其余常识内容自然带过，不要逐条标注。\n"
    "4. 用户的问题若明显不属于养生科普（写代码、查股票、看天气、买东西、"
    "代写文书等），不要作答，直接说『资料库中没有这方面的内容』。\n"
    "5. 【严禁凭自己的知识补写典籍原文】——你只能引用【参考资料】里实际出现的"
    "文字；参考资料里没有出现的原句一律不得编造，尤其是《黄帝内经》《抱朴子》"
    "等古籍的引文。回答中每一处引用都必须能在参考资料里逐字找到。\n"
    "6. 【开场直接回答问题本身】【严禁】用「你这次只提到…没有说过…所以只能"
    "给…」这类清点用户没提供过什么信息的声明开场，也不要在正文里逐条罗列"
    "用户没说过的项目（体质、舌象、大便、用药…）；需要用户补充什么，"
    "只在回答结尾自然地问一句。\n\n"
    "【本次对话的用户背景】以下信息只来自**这次对话**里用户自己说过的话。"
    "请自然地加以利用（例如他这次已确认自己是阳虚体质，就不必再让他重复描述）：\n"
    "注意：【用户背景】里有的内容【不属于】「资料库中没有」——用户问起自己的"
    "情况（作息、职业、体质、饮食偏好等）时，直接用【用户背景】回答，"
    "不要说「资料库中没有你个人的记录」。\n"
    "**但【用户背景】里没有的，就是这次对话里他没说过**：不要从常识、"
    "不要从别的对话去补全，也不要假设「他上次说过」；需要时在**回答结尾**"
    "自然地问他要（不要在开场清点他没说过什么）。\n"
    "{memory}\n\n"
    "参考资料：\n{context}"
)


# ---------------------------------------------------------------------------
# 向量库单例（RAGSession 与 Agent 工具共用，避免重复打开 HNSW 索引）
# ---------------------------------------------------------------------------
_db: Chroma | None = None


def get_db() -> Chroma:
    """全局向量库单例。

    必须用相对路径打开（见 app/paths.py 的 chromadb 中文绝对路径 bug），
    直接传 str(STORE_DIR) 会报 Error loading hnsw index。
    """
    global _db
    if _db is None:
        if not STORE_DIR.exists():
            raise SystemExit("找不到向量库，请先运行: python -m app.index")
        _db = Chroma(
            persist_directory=chroma_store_path(),
            embedding_function=get_embeddings(),
            collection_name=COLLECTION,
        )
    return _db


@dataclass
class Answer:
    """一次问答的完整结果（含中间过程，便于 UI 展示与调试）。"""
    question: str                       # 用户原始问题
    standalone: str                     # 改写后的独立问题
    answer: str                         # 最终回答
    sources: list = field(default_factory=list)     # [{source, chapter, score}]
    reflection: list = field(default_factory=list)  # 自反思各轮过程
    memory_used: dict = field(default_factory=dict)  # 本轮记忆使用情况
    context: str = ""                   # 实际喂给 LLM 的参考资料全文（评估/调试用）
    safety: dict = field(default_factory=dict)       # 硬规则安全预扫结果（命中/冲突/缺口）


class RAGSession:
    """一个会话 = 一段连续对话（共享检索器与记忆）。"""

    def __init__(self, conv_id: int | None = None,
                 k_recall: int = K_RECALL, k_final: int = K_FINAL,
                 reflect: bool = True, auto_remember: bool = True):
        self.conv_id = conv_id
        self.k_recall = k_recall
        self.k_final = k_final
        self.reflect = reflect
        self.auto_remember = auto_remember
        self.history: list[dict] = []   # 仅 CLI（无 conv_id）模式使用
        self.memory = MemoryManager()

    # ---------- 基础设施 ----------
    @property
    def db(self) -> Chroma:
        return get_db()

    def reset(self) -> None:
        """开启新话题：清空对话记忆（向量库与长期档案不动）。

        注意：Web 路径的历史来自 SQLite，"清空内存"没有意义——
        新话题请新建会话（server.py 的做法），长期档案按设计跨会话保留。
        """
        self.history.clear()

    # ---------- 记忆组装（两条路径的唯一出口） ----------
    def _context(self, query: str) -> MemoryContext:
        """组装本轮上下文：Web 走三层记忆，CLI 退回内存历史。"""
        if self.conv_id is not None:
            return self.memory.build_context(self.conv_id, query)
        recent = self.history[-CLI_HISTORY:]
        return MemoryContext(recent=list(recent),
                             stats={"mode": "cli", "recent_kept": len(recent)})

    # ---------- 三步主流程 ----------
    def _condense(self, question: str, ctx: MemoryContext) -> str:
        """① 有历史时把追问改写成独立问题；首轮直接返回原问题。"""
        if not ctx.recent:
            return question
        history_text = "\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}"
            for m in ctx.recent[-8:]
        )
        prompt = CONDENSE_PROMPT.format(history=history_text, question=question)
        standalone = get_llm().invoke(prompt).content.strip()
        return standalone or question

    def _retrieve(self, standalone: str):
        """② 自反思两段式检索，返回 (hits, attempts, reason)。"""
        if not self.reflect:
            from app.selfrag import retrieve
            return retrieve(self.db, standalone,
                            self.k_recall, self.k_final), [], ""
        res = retrieve_with_reflection(self.db, standalone,
                                       k_recall=self.k_recall, k_final=self.k_final)
        return res.hits, res.attempts, res.reason

    def _context_block(self, hits: list) -> str:
        """把检索命中拼成给 LLM 的参考资料区块（生成与评估共用同一份文本）。

        取用侧过两道过滤（都不重建索引——重建要重嵌入上万块，代价大而无必要）：

        ① `strip_meta_info`：语料里夹着「我们下回分解」「详见下节」这类
           **写作性元信息**，它们不是知识结论，被检索进来后模型会照引出来
           （实测出现过"资料里说有个简单有效的方法，但原文卖了个关子"这种回答），
           既是噪声也损害可信度；
        ② `credibility.filter_chunk`（P0-3）：引文不只要"相关"，还要"可采信"。
           实测把典籍里的传说性记载（鹿茸「中有小白虫，入人鼻必为虫颡」）
           当严肃建议引了出来。这里按句剔除**不可采信**的记载、
           给传说类句打「仅作文化背景」的标注。
        """
        from app import credibility as CR
        from app.safety import strip_meta_info

        return "\n\n---\n\n".join(
            f"[来源: {d.metadata.get('source', '?')} · "
            f"{d.metadata.get('chapter', '?')}]\n"
            f"{CR.filter_chunk(strip_meta_info(d.page_content))[0]}"
            for d, _score in hits
        ) or "（检索无结果）"

    def _safety_guard(self, question: str) -> tuple[str, dict | None, list[dict], dict]:
        """RAG 侧的硬规则安全预扫 + 追问清单 + **判断框架**（与问诊 Agent 同一套规则）。

        为什么问答路径也要有：用户实测那次对话就是在问答路径里发生——
        他问"附子理中丸 + 阿胶 + 红豆薏米 + 降压药能不能一起吃"，
        当时的回答逐条检索资料库后如实说"资料库里没有这方面的记载"。
        从"不编造"看没错，从**安全**看是失败的：
        附子有毒、甘草升血压、活血药加抗凝药出血风险——这些是确定的药学事实，
        不该因为语料没覆盖就不再提醒。

        返回值是四元组：`(guard 提示词块, safety 事件或 None, 追问清单, 上下文)`。
        追问清单单独返回，是因为**生成之后还要用它做程序化校验**
        （模型没问就由代码补上，见 app/inquiry.py::ensure_questions）；
        上下文交给 `contract.enforce` 做六模块完整性补全。
        """
        from app import constraints as CN
        from app import contract as CT
        from app import framework as FW
        from app import inquiry as IQ
        from app import intake as IL
        from app.safety import (active_tags, annotate_origins, build_block,
                                needs_medication_stop, scan_profile, scan_text)
        from app.safety import scripts as SG
        from app.safety import tiers as T

        try:
            cid = self.conv_id
            # 档案按会话隔离（2026-09-16）：这里读到的一切都只属于本次对话
            prof0 = IL.storage.get_profile(cid)
            hits = annotate_origins(scan_text(question),
                                    scan_profile(prof0), question)
            IL.update_from_message(question, cid)
            IL.record_herbs([h for h in hits if h.kind == "herb"], cid)
            prof = IL.storage.get_profile(cid)
            tags = active_tags(question, profile=prof)
            conflicts = IL.profile_conflicts(prof, question, conv_id=cid)
            # 追问：先判意图（求建议才问，纯知识题不问），再算缺口
            intent = IQ.intent_of(question)
            gap_list = IL.gaps(prof, question, require=IQ.require_slots(intent),
                               conv_id=cid)
            fups = IQ.followups(question, conv_id=cid, profile=prof,
                                intent=intent, gaps=gap_list,
                                hits=hits, tags=tags)
            # ---- 状态类约束（备孕/妊娠/哺乳/高龄…）：优先级高于药名规则（P1-3）----
            # 必须在 `apply_tiers` **之前**算出来：状态要**直接参与档位判定**
            # （2026-09-16 第五轮：识别到妊娠、档位却没跟上 = 版本退化）。
            states = CN.detect(question, prof, cid)
            # 五档分级：档位由「东西 × 他的条件 × 条件是否已评估 × 状态」共同决定
            T.apply_tiers(hits, tags, gap_list, states=states)
            advisory = IL.advisory_gaps(tags, prof, question, conv_id=cid)
            screen = IL.screening_keys(tags, prof, question, conv_id=cid)
            stop = needs_medication_stop(hits, tags)
            # 方向反转/换人设：既往记录在确认前不得作为判断依据
            switch = IL.subject_switch(question, cid, prof)
            # ---- 判断框架：先有框架，再用知识填充（框架不因检索缺失而消失）----
            hits_dict = [h.to_dict() for h in hits]
            plan = FW.build(question, intent=intent, tags=tags, hits=hits_dict,
                            screening=screen, gaps=gap_list)
            guard = SG.combine(
                build_block(hits, tags, with_stop=stop,
                            profile_ok=not switch, gaps=gap_list),
                # 状态约束块插在安全块之后、其它块之前——它是"先识别状态"那一步
                CN.render_block(states),
                IL.conflict_block(conflicts),
                IL.cross_turn_brief(cid, question, profile=prof),
                SG.CITATION_JUDGMENT,
                SG.TRANSLATE_CLASSICS,
                # 要追问时**不给**分层结论模板——"先给分层总括"与
                # "信息不足不得下证型结论"是互相打架的两条，必须二选一
                "" if fups else SG.LAYERED_CONCLUSION,
                SG.screening_block(screen),
                IQ.followup_block(fups),
                SG.advisory_block(advisory),
                FW.render(plan),
                CT.render_contract(),
            )
            ev = None
            if hits or conflicts or fups:
                ev = {"type": "safety", "hits": hits_dict,
                      "tags": sorted(tags), "conflicts": conflicts,
                      "stop": stop, "screening": screen,
                      "gaps": [f["key"] for f in fups],
                      "questions": IQ.to_event(fups),
                      "advisory": advisory,
                      # 布尔给逻辑判断用，detail 给界面说清"哪一轴反了"——
                      # 只报 true/false，用户看到"疑似换了对象"却不知道依据，等于没说
                      "subject_switch": bool(switch),
                      "subject_switch_detail": switch,
                      "states": states,
                      "plan": plan.to_dict()}
            ctx = {"hits": hits_dict, "tags": sorted(tags), "screening": screen,
                   "questions": IQ.to_event(fups), "plan": plan,
                   "gaps": gap_list, "stop": stop, "states": states,
                   "question": question, "profile": prof}
            return guard, ev, fups, ctx
        except Exception:
            # 兜底不能是"静默"：2026-09-16 出过一个隐蔽 bug ——
            # intake.screening_keys 漏声明 conv_id，三处调用点全部 TypeError，
            # 全被这里吞掉，于是"安全事件 + 追问清单"整条链路消失，
            # 而四个纯规则单测仍全绿（它们不经过本函数）。
            # 现在把异常打到 stderr，至少真机日志里看得见。
            import traceback
            print("[safety-guard] 预扫失败（本轮退化为无护栏回答）：",
                  file=sys.stderr, flush=True)
            traceback.print_exc()
            return "", None, [], {}

    def _build_messages(self, standalone: str, question: str,
                        hits: list, ctx: MemoryContext,
                        guard: str = "") -> list:
        """构造生成回答的消息列表（invoke 与 stream 共用）。"""
        context = self._context_block(hits)
        memory_block = ctx.as_prompt_block() or "（暂无历史信息）"
        system = ANSWER_SYSTEM.format(memory=memory_block, context=context)
        if guard:
            # 安全约束放在最后（大模型对末尾指令更敏感），并明确其优先级
            system += ("\n\n=== 安全与回答方式约束（由系统按本轮情况生成，"
                       "优先级高于以上所有要求）===\n" + guard)
        messages = [SystemMessage(content=system)]
        for m in ctx.recent:                     # 近期原文（保真）
            messages.append(
                HumanMessage(m["content"]) if m["role"] == "user"
                else AIMessage(m["content"])
            )
        messages.append(HumanMessage(question))  # 当前问题用原文（口语更自然）
        return messages

    def _post_turn(self, question: str, answer: str) -> None:
        """一轮结束后的记忆维护：抽长期事实 + 压缩早期历史。

        放在回答推送之后执行，用户已看到完整答案，延迟不落在体感上。
        """
        if self.conv_id is None:
            return
        try:
            if self.auto_remember:
                self.memory.remember_turn(self.conv_id, question, answer)
            self.memory.maybe_compress(self.conv_id)
        except Exception:                        # 记忆维护失败不应影响主流程
            pass

    # ---------- 对外唯一入口 ----------
    def ask(self, question: str) -> Answer:
        from app.contract import enforce
        from app.inquiry import ensure_questions
        question = question.strip()
        ctx = self._context(question)
        standalone = self._condense(question, ctx)
        hits, attempts, reason = self._retrieve(standalone)
        guard, ev, fups, gctx = self._safety_guard(question)
        answer = get_llm().invoke(
            self._build_messages(standalone, question, hits, ctx, guard)).content.strip()
        answer, _added = ensure_questions(answer, fups)     # 追问兜底（同流式路径）
        answer, _mods = enforce(answer, gctx)               # 输出契约：六模块自检
        if self.conv_id is None:                 # CLI 模式自己维护内存历史
            self.history.append({"role": "user", "content": question})
            self.history.append({"role": "assistant", "content": answer})
        self._post_turn(question, answer)
        return Answer(
            question=question, standalone=standalone, answer=answer,
            sources=[{"source": d.metadata.get("source", "?"),
                      "chapter": d.metadata.get("chapter", "?"),
                      "score": round(score, 3),
                      "credibility": CR.classify(d.page_content),
                      "credibility_label": CR.level_label(
                          CR.classify(d.page_content))}
                     for d, score in hits],
            reflection=[{"round": a.round, "query": a.query,
                         "top_score": a.top_score, "accepted": a.accepted}
                        for a in attempts],
            memory_used=ctx.stats,
            context=self._context_block(hits),
            safety=ev or {},
        )

    # ---------- 流式版主流程（Web SSE 用） ----------
    def ask_stream(self, question: str):
        """ask() 的流式版本：逐步 yield 事件 dict。

        事件类型（dict）：
          {"type": "memory", "stats": {...}}        本轮记忆使用情况
          {"type": "standalone", "text": ...}       问题改写完成
          {"type": "safety", ...}                   硬规则安全预扫（命中/冲突/追问）
          {"type": "reflection", "attempts": [...]} 自反思检索过程
          {"type": "sources", "sources": [...]}     检索完成（引用来源）
          {"type": "delta", "text": ...}            回答增量片段（多次）
          {"type": "questions", "items": [...]}     本轮需要向用户追问的点（可点选）
          {"type": "done", "answer": ...}           回答完成（全文）

        安全事件故意排在 sources/delta 之前：前端可先把「附子有毒/甘草升血压」
        这类判读横幅渲染出来，再让正文逐字流出——用户第一眼就该看到风险，
        而不是读完整段回答才在末尾发现。

        `questions` 事件排在 done 之前：追问由**代码**判定（app/inquiry.py），
        前端拿到后渲染成可点选的问题条——用户点一下就把信息补上，
        比让他在正文里找"你还没告诉我…"再手打一遍要顺得多。
        """
        from app.contract import enforce
        from app.inquiry import ensure_questions, to_event
        question = question.strip()
        ctx = self._context(question)
        yield {"type": "memory", "stats": ctx.stats}

        standalone = self._condense(question, ctx)
        yield {"type": "standalone", "text": standalone}

        guard, ev, fups, gctx = self._safety_guard(question)
        if ev:
            yield ev

        hits, attempts, reason = self._retrieve(standalone)
        if attempts:
            yield {"type": "reflection", "reason": reason,
                   "attempts": [{"round": a.round, "query": a.query,
                                 "top_score": a.top_score, "accepted": a.accepted}
                                for a in attempts]}
        yield {"type": "sources", "sources": [
            {"source": d.metadata.get("source", "?"),
             "chapter": d.metadata.get("chapter", "?"),
             "score": round(score, 3),
             # P0-3：来源要标"可采信性"。界面能据此把传说性段落标成
             # 「仅作文化背景」，而不是让用户以为它是依据。
             "credibility": CR.classify(d.page_content),
             "credibility_label": CR.level_label(CR.classify(d.page_content))}
            for d, score in hits]}

        messages = self._build_messages(standalone, question, hits, ctx, guard)
        parts: list[str] = []
        for chunk in get_llm().stream(messages):
            if chunk.content:
                parts.append(chunk.content)
                yield {"type": "delta", "text": chunk.content}
        raw = "".join(parts).strip()

        # ---- 追问兜底：模型没问就由代码补（关键动作不依赖模型自觉）----
        answer, added = ensure_questions(raw, fups)
        # ---- 输出契约：六模块完整性自检，缺哪个补哪个（同样不依赖模型自觉）----
        answer, mods = enforce(answer, gctx)
        if answer != raw:
            yield {"type": "delta", "text": answer[len(raw.rstrip()):]}
        if fups:
            yield {"type": "questions", "items": to_event(fups), "added": added}
        if mods:
            yield {"type": "contract", "filled": mods}

        if self.conv_id is None:
            self.history.append({"role": "user", "content": question})
            self.history.append({"role": "assistant", "content": answer})
        yield {"type": "done", "answer": answer}

        # 回答已推送完毕，再做记忆维护（不影响用户体感的耗时）
        self._post_turn(question, answer)
