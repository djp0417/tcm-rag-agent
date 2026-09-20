# -*- coding: utf-8 -*-
"""Web 服务：FastAPI 后端 + DeepSeek 风格静态前端（web/index.html / agent.html）。

两个模式共用同一套会话与持久化：
  · **问答模式**（`/`，web/index.html）—— RAG 问答：三层记忆 + 自反思检索 + 流式回答；
  · **问诊模式**（`/agent`，web/agent.html）—— 体质辨识 Agent：状态机 + function calling，
    可实时看到"工具调用过程 / 自反思检索 / 答题进度 / 判定卡片"。

面向「记忆」的几个接口把会话记忆暴露出来，便于观察与调试。
**⚠️ 2026-09-16 起记忆按会话隔离**：档案/时间线/记忆条目都只属于**当前会话**，
新开会话即归零；跨会话长期记忆（`scope=global`）只有用户**明说"记住"**才会写入、
且只有用户**主动问起往事**时才会被召回。所以这些接口的 `conv_id` **不是可选装饰**——
不传就是"无会话作用域"（CLI 用的那个），Web 端一律要传。
    GET /api/profile?conv_id=      本会话用户档案（体质判定/年龄性别/慢病/在服清单）
    GET /api/memories?conv_id=&scope=&include_archived=
                                   记忆条目（scope=session 只返回本会话的）
    会话内记忆统计随 SSE 的 memory 事件实时下发

API 一览：
    GET    /api/conversations                会话列表（置顶优先，其次最近活跃）
    POST   /api/conversations                新建会话 {title?}
    DELETE /api/conversations/{id}           删除会话（连带消息 + 本会话档案 + session 记忆）
    PATCH  /api/conversations/{id}           重命名 {title} / 置顶 {pinned}
    GET    /api/conversations/{id}/messages  全部消息
    POST   /api/conversations/{id}/chat      问答模式：发消息 {message} → SSE
    POST   /api/agent/{id}/chat              问诊模式：发消息 {message} → SSE
    GET    /api/agent/{id}/state             问诊状态（阶段/进度/判定结果）
    GET    /api/profile?conv_id=             本会话用户档案
    GET    /api/profile/panel?conv_id=       档案面板视图（字段/冲突/时间线/归档计数）
    PUT    /api/profile/{key}?conv_id=       行内修正某项
    DELETE /api/profile/{key}?conv_id=       删除某项
    GET    /api/memories?conv_id=&scope=     记忆条目
    DELETE /api/memories/{mid}               删除一条记忆
    GET    /                                 问答页
    GET    /agent                            问诊页

用法（项目根目录）：
    python -m app.server          # http://127.0.0.1:7860
"""
import json
import queue
import threading

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import storage
from app.agent.agent import ConsultAgent
from app.agent.state import ConsultState, STAGE_LABEL
from app.memory import MemoryManager
from app.paths import BASE_DIR
from app.rag import RAGSession

app = FastAPI(title="中医养生科普助手")

WEB_DIR = BASE_DIR / "web"

# 并发护栏：同一时刻只放行一个聊天请求（本地单用户，足够且避免
# 多线程同时写 chroma 连接 / SQLite）
_chat_lock = threading.Lock()

# ⚠️ 曾经的死锁事故（2026-09-14，实测复现）：
#   旧写法是 `def api_chat()` 同步端点 + 在【同步生成器 event_stream() 里】
#   `with _chat_lock:` 包住整个流。一旦客户端中途断开（关闭页面 / curl 被
#   `head -c` 截断 / 请求超时），Starlette 会取消对生成器的迭代——但
#   `anyio.to_thread.run_sync` 无法杀掉工作线程，生成器就被【永久挂在
#   yield 上】：`with` 块的 __exit__ 永不执行 → 锁永不释放 → 之后【所有】
#   聊天请求全部卡在获取锁，服务被焊死，只能重启。
#   实测表现：uvicorn 访问日志仍是 "200 OK"（它记在响应开始/结束时），
#   但 store/chat.db 里一条 assistant 消息都没有——这是最准的判据。
#
# 修法：把「干活」和「推流」解耦。
#   · 生产者：独立守护线程，持锁跑完整个 RAG 流程，事件塞进 Queue；
#     线程【一定会跑到底】（不受客户端影响），所以锁一定会被释放。
#   · 消费者：SSE 生成器只负责从 Queue 取事件推给前端；客户端断开时
#     只是没人取了，生产者照常跑完并释放锁。
#   · Queue.get 加超时兜底，防止生产者线程意外死亡时前端干等。
_CHAT_QUEUE_TIMEOUT = 180     # 单个事件最长等待（秒），超时即报错收尾
_END = object()               # 队列哨兵：生产者跑完（成功/异常）都会放进来

_memory = MemoryManager()


# ---------- 请求体模型 ----------
class ConvCreate(BaseModel):
    title: str | None = None


class ConvPatch(BaseModel):
    title: str | None = None
    pinned: bool | None = None


class QuickAnswer(BaseModel):
    """点选快速作答：分值由前端按钮直接给出，note 为可选补充说明。"""
    score: int
    note: str = ""


class ChatReq(BaseModel):
    message: str
    quick: QuickAnswer | None = None      # 仅问诊端点使用；点选作答时携带


# ---------- 会话 CRUD ----------
@app.get("/api/conversations")
def api_list_conversations():
    return storage.list_conversations()


@app.post("/api/conversations")
def api_create_conversation(req: ConvCreate | None = None):
    title = (req.title if req and req.title else "新对话")[:40]
    return storage.create_conversation(title)


@app.delete("/api/conversations/{conv_id}")
def api_delete_conversation(conv_id: int):
    storage.delete_conversation(conv_id)
    return {"ok": True}


class ConvBatchDelete(BaseModel):
    ids: list[int]


@app.post("/api/conversations/delete-batch")
def api_delete_conversations_batch(req: ConvBatchDelete):
    """批量删除会话（前端多选管理用；逐条循环删除，幂等）。"""
    n = 0
    for cid in req.ids:
        storage.delete_conversation(cid)
        n += 1
    return {"ok": True, "deleted": n}


@app.patch("/api/conversations/{conv_id}")
def api_patch_conversation(conv_id: int, req: ConvPatch):
    if req.title is not None:
        title = req.title.strip()[:40]
        if not title:
            raise HTTPException(400, "标题不能为空")
        storage.rename_conversation(conv_id, title)
    if req.pinned is not None:
        storage.set_pinned(conv_id, req.pinned)
    return {"ok": True}


@app.get("/api/conversations/{conv_id}/messages")
def api_get_messages(conv_id: int):
    return storage.get_messages(conv_id)


def _require_conv(conv_id: int) -> None:
    if not any(c["id"] == conv_id for c in storage.list_conversations()):
        raise HTTPException(404, "会话不存在")


def _auto_title(conv_id: int, message: str) -> None:
    """首条消息时自动生成会话标题（取问题前 20 字，DeepSeek 同款思路）。"""
    if len(storage.get_messages(conv_id)) == 1:
        storage.rename_conversation(conv_id, message[:20] + ("…" if len(message) > 20 else ""))


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _run_chat(sess, message: str, q: "queue.Queue") -> None:
    """生产者：持锁跑完整个 RAG 流程，事件写入队列（详见 _chat_lock 处注释）。

    跑在独立守护线程里，客户端是否断开都不影响它跑完，因此锁一定释放。
    """
    with _chat_lock:
        try:
            for event in sess.ask_stream(message):
                q.put(event)
        except Exception as e:                    # 网络/API 异常也要推给前端
            q.put({"type": "error", "text": str(e)})
        finally:
            q.put(_END)                           # 哨兵：无论成败都要收尾


def _run_stream(events, q: "queue.Queue") -> None:
    """同 _run_chat，但不加锁 —— 问诊 Agent 无共享可变检索状态
    （状态已按会话落 SQLite），只复用「生产者/消费者」的死锁防护结构。

    events 是已创建的事件生成器（ask_stream 或 ask_quick 的返回值，
    生成器惰性求值，跨线程传递安全）。
    """
    try:
        for event in events:
            q.put(event)
    except Exception as e:
        q.put({"type": "error", "text": str(e)})
    finally:
        q.put(_END)


def _sse_stream(q, on_done):
    """消费者：把队列里的事件转成 SSE；结束/出错时回调 on_done 落库。"""
    answer_out, sources_out, safety_out = "", [], {}
    while True:
        try:
            item = q.get(timeout=_CHAT_QUEUE_TIMEOUT)
        except queue.Empty:
            yield _sse({"type": "error",
                        "text": f"生成超时（{_CHAT_QUEUE_TIMEOUT}s 无新事件）"})
            break
        if item is _END:
            break
        if item["type"] == "sources":
            sources_out = item["sources"]
        elif item["type"] == "safety":
            # 安全判读随消息落库：否则用户刷新页面后提示就没了（安全信息不该
            # 只在当轮可见）。多轮里若重复下发，取最后一次（覆盖式）。
            safety_out = {k: v for k, v in item.items() if k != "type"}
        elif item["type"] == "done":
            answer_out = item["answer"]
        yield _sse(item)
    on_done(answer_out, sources_out, safety_out)


# ---------- 问答模式（SSE）----------
@app.post("/api/conversations/{conv_id}/chat")
def api_chat(conv_id: int, req: ChatReq):
    message = req.message.strip()
    if not message:
        raise HTTPException(400, "消息不能为空")
    _require_conv(conv_id)

    # 用户消息先落库（即使后续生成中断，历史也完整）
    storage.add_message(conv_id, "user", message)
    _auto_title(conv_id, message)

    # RAGSession 现在直接从 SQLite 读三层记忆，无需缓存会话对象
    sess = RAGSession(conv_id=conv_id)
    q: "queue.Queue" = queue.Queue()
    threading.Thread(target=_run_chat, args=(sess, message, q), daemon=True).start()

    return StreamingResponse(
        _sse_stream(q, lambda a, s, sf: storage.add_message(conv_id, "assistant", a, s, sf)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------- 问诊模式（Agent + SSE）----------
@app.get("/api/agent/{conv_id}/state")
def api_agent_state(conv_id: int):
    st = ConsultState.load(conv_id)
    return {"stage": st.stage, "stage_label": STAGE_LABEL[st.stage],
            "answered": st.answered, "total": st.total,
            "primary_key": st.primary_key, "report": st.last_result,
            "summary": st.summary_for_llm(),
            # 当前待答题：前端刷新/重开会话时据此恢复"点选作答条"
            "question": st.current_question()}


@app.post("/api/agent/{conv_id}/chat")
def api_agent_chat(conv_id: int, req: ChatReq):
    message = req.message.strip()
    if not message:
        raise HTTPException(400, "消息不能为空")
    _require_conv(conv_id)

    storage.add_message(conv_id, "user", message)
    _auto_title(conv_id, message)

    agent = ConsultAgent(conv_id, memory=_memory)
    # 点选快速作答：分值由按钮直接给出，走 ask_quick（无补充说明时零 LLM，秒回）
    if req.quick is not None:
        events = agent.ask_quick(req.quick.score, req.quick.note)
    else:
        events = agent.ask_stream(message)
    # 与问答端点同构：生产者线程跑完整个 Agent 循环（客户端断开不影响），
    # 消费者只管把事件转 SSE —— 举一反三：同步生成器直连模式在客户端中途
    # 断开时同样会挂起在 yield 上（问答端点的死锁教训，见文件头注释）。
    q: "queue.Queue" = queue.Queue()
    threading.Thread(target=_run_stream, args=(events, q),
                     daemon=True).start()

    return StreamingResponse(
        _sse_stream(q, lambda a, s, sf: storage.add_message(conv_id, "assistant", a, s, sf)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------- 调理方案（多智能体流水线 + SSE）----------
# 2026-09-15 测试缺陷：「经络 Agent 从未输出穴位」的根因不在提示词，
# 而是**流水线根本没接到 Web 上**——三个专家只有 CLI 入口能跑，
# 网页端自然"几乎只有主控在输出"。这个端点把它变成一等公民：
# 前端随时可以点「生成系统调理方案」，看到每个 Agent 分步产出。
def _run_pipeline(conv_id: int, message: str, q: "queue.Queue") -> None:
    """生产者：跑多智能体流水线，节点完成事件与最终答案写入队列。"""
    from app.multiagent.graph import run_pipeline
    from app.multiagent.state import render

    try:
        def on_step(node: str, delta: dict) -> None:
            # 节点级事件：前端可实时显示「食疗专家产出中…」
            brief = ""
            if node in ("diet_expert", "meridian_expert", "movement_expert"):
                n = sum(len(v) for k, v in (delta or {}).items()
                        if isinstance(v, list))
                brief = f"{n} 条建议"
            q.put({"type": "step", "node": node, "brief": brief})

        final = run_pipeline(conv_id, message, on_step=on_step)
        answer = final.get("final_answer") or render(final)
        # 安全判读卡：把接诊命中带给前端（与问答页同构）
        it = final.get("intake") or {}
        if it.get("hits") or it.get("conflicts") or it.get("questions"):
            q.put({"type": "safety",
                   "hits": it.get("hits") or [],
                   "tags": it.get("tags") or [],
                   "conflicts": it.get("conflicts") or [],
                   "stop": it.get("stop") or False,
                   "screening": it.get("screening") or [],
                   "gaps": it.get("gaps") or [],
                   "questions": it.get("questions") or []})
        q.put({"type": "trace", "nodes": [t.get("node") for t in final.get("trace") or []]})
        q.put({"type": "done", "answer": answer})
    except Exception as e:
        q.put({"type": "error", "text": str(e)})
    finally:
        q.put(_END)


@app.post("/api/plan/{conv_id}/chat")
def api_plan_chat(conv_id: int, req: ChatReq):
    message = req.message.strip()
    if not message:
        raise HTTPException(400, "消息不能为空")
    _require_conv(conv_id)

    storage.add_message(conv_id, "user", message)
    _auto_title(conv_id, message)

    q: "queue.Queue" = queue.Queue()
    threading.Thread(target=_run_pipeline, args=(conv_id, message, q),
                     daemon=True).start()

    return StreamingResponse(
        _sse_stream(q, lambda a, s, sf: storage.add_message(conv_id, "assistant", a, s, sf)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------- 会话记忆 / 档案 ----------
# 2026-09-16「记忆按会话隔离」：档案与记忆都按 conv_id 存，所以这几个
# 接口一律带 conv_id（不传 = 只看到"无会话"那份，不会误读别的对话）。
@app.get("/api/profile")
def api_profile(conv_id: int | None = None):
    return storage.get_profile(conv_id)


@app.get("/api/profile/panel")
def api_profile_panel(conv_id: int | None = None):
    """记忆面板：**本会话**结构化档案 + 一致性冲突 + 主诉时间线（P2-3）。

    与 /api/profile 的区别：这里是**给用户看和改的视图**，会额外带上
    「档案与自述对不上的地方」（如体质标签说阴虚、自述却畏寒），
    前端据此给用户一个自助复测入口——而不是让用户对着一堆键值对发呆。
    """
    from app import intake
    return intake.panel(conv_id)


class _ProfilePatch(BaseModel):
    value: str


@app.put("/api/profile/{key}")
def api_put_profile(key: str, req: _ProfilePatch, conv_id: int | None = None):
    """用户手动修正档案项（记忆面板的"可修正"能力）。"""
    storage.set_profile(key, req.value.strip(), conv_id)
    return {"ok": True, "key": key, "value": req.value.strip(),
            "conv_id": conv_id}


@app.delete("/api/profile/{key}")
def api_delete_profile(key: str, conv_id: int | None = None):
    storage.delete_profile(key, conv_id)
    return {"ok": True}


@app.get("/api/memories")
def api_memories(limit: int = 200, conv_id: int | None = None,
                 scope: str | None = None, include_archived: bool = False):
    """记忆条目。默认只给**本会话**的 + 用户明说「记住」的跨会话条目；
    传 include_archived=true 可看到 2026-09-16 之前遗留的旧全局条目。"""
    return storage.list_memories(limit=limit, conv_id=conv_id, scope=scope,
                                 include_archived=include_archived)


@app.delete("/api/memories/{mid}")
def api_delete_memory(mid: int):
    storage.delete_memory(mid)
    return {"ok": True}


# ---------- 静态前端 ----------
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/agent")
def agent_page():
    return FileResponse(WEB_DIR / "agent.html")


def _port_in_use(host: str, port: int) -> bool:
    """预检端口：有人监听则返回 True（用于给出比 uvicorn 10048 更可读的提示）。"""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def main() -> None:
    """启动 Web 服务（默认 http://127.0.0.1:7860）。

    端口被占时不直接抛 10048，而是打印可照抄的排查/换端口命令——
    本机常见原因是上一次的服务进程没退干净。
    """
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(
        prog="python -m app.server",
        description="中医养生 RAG Web 服务（问答 / + 问诊 /agent）")
    ap.add_argument("--host", default="127.0.0.1",
                    help="监听地址，默认 127.0.0.1（仅本机可访问）")
    ap.add_argument("--port", type=int, default=7860,
                    help="监听端口，默认 7860；被占时可换 7861 等")
    args = ap.parse_args()

    if _port_in_use(args.host, args.port):
        print(f"[error] {args.host}:{args.port} 已被占用，无法启动。")
        print("        1) 换端口启动： python -m app.server --port 7861")
        print("        2) 或先找出占用者并结束它：")
        print(f"           netstat -ano | findstr :{args.port}")
        print("           taskkill /F /PID <上面最后那列的数字>")
        raise SystemExit(1)

    print(f"[srv  ] 问答 http://{args.host}:{args.port}/"
          f" ；问诊 http://{args.host}:{args.port}/agent")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
