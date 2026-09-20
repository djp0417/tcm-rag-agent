# -*- coding: utf-8 -*-
"""会话持久化：SQLite 存对话（会话列表 + 消息），Web 服务重启不丢。

设计：
- 两张表：conversations（id/标题/置顶/时间戳）、messages（role/content/sources）；
- sources 存 JSON 字符串（引用来源列表），读取时反序列化；
- 每个操作开新连接（SQLite 文件锁足够，本地单用户场景无需连接池）；
- updated_at 在每次追加消息时刷新，会话列表按（置顶优先，其次最近活跃）排序。

用法：
    from app.storage import (
        list_conversations, create_conversation, delete_conversation,
        rename_conversation, set_pinned, add_message, get_messages,
    )
"""
import json
import sqlite3
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "store" / "chat.db"

# 记忆/档案的作用域取值（2026-09-16「记忆按会话隔离」）
SESSION_SCOPE = 0      # 无会话（CLI 单进程）自成作用域
LEGACY_SCOPE = -1      # 旧"全局档案/记忆"的归档归宿：不再被任何会话读到
MEM_SESSION = "session"    # 只属于写它的那个会话（默认）
MEM_GLOBAL = "global"      # 用户明说"记住"的事实，唯一允许跨会话
MEM_ARCHIVED = "archived"  # 历史遗留的全局条目（不再自动注入）

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    title      TEXT    NOT NULL DEFAULT '新对话',
    pinned     INTEGER NOT NULL DEFAULT 0,      -- 0/1
    created_at REAL    NOT NULL,
    updated_at REAL    NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id    INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role       TEXT    NOT NULL,                -- 'user' / 'assistant'
    content    TEXT    NOT NULL,
    sources    TEXT    NOT NULL DEFAULT '[]',   -- JSON 列表
    safety     TEXT    NOT NULL DEFAULT '{}',   -- 本轮安全判读（JSON，刷新后仍在）
    created_at REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conv_id);

-- 问诊状态机：一个会话一条（payload 为 ConsultState 的 JSON）
CREATE TABLE IF NOT EXISTS consult_state (
    conv_id    INTEGER PRIMARY KEY,
    stage      TEXT    NOT NULL,
    payload    TEXT    NOT NULL,
    updated_at REAL    NOT NULL
);

-- 会话档案（2026-09-16「记忆按会话隔离」改造的核心）
-- ------------------------------------------------------------------
-- 旧设计是**全局一份** profile（跨会话生效），实测后果：
--   · 换了个人设/换了个人来问，上一轮的信息被当成"这个人的档案"；
--   · 新开的对话第一轮就冒出好几张上一次的判读卡（"我的记忆有点乱"）。
-- 新设计：档案跟**会话**走——`conv_id` 是作用域主键的一部分。
--   · 新开的对话 = 空档案（不继承任何历史）；
--   · 同一会话内照旧逐轮累积（慢病/西药/年龄/在服中药/主诉时间线）；
--   · 跨会话引用必须由用户**主动问起**（"我上次说的阿胶还能吃吗"），
--     且只从 scope='global' 的记忆条目里取，不再读别的会话的档案。
CREATE TABLE IF NOT EXISTS session_profile (
    conv_id    INTEGER NOT NULL,   -- 0 = CLI/无会话；-1 = 历史遗留（归档，不再注入）
    key        TEXT    NOT NULL,
    value      TEXT    NOT NULL,
    updated_at REAL    NOT NULL,
    PRIMARY KEY (conv_id, key)
);

-- ⚠️ DEPRECATED（2026-09-16）：旧的全局档案表，已不再读写。
-- 保留建表语句只为两件事：① `_archive_legacy_profile` 能把老数据搬走；
-- ② 万一要回滚，老数据还在。搬迁完成后它恒为空表。
CREATE TABLE IF NOT EXISTS profile (
    key        TEXT    PRIMARY KEY,
    value      TEXT    NOT NULL,
    updated_at REAL    NOT NULL
);

-- 记忆条目：从对话里沉淀下来的"值得记住的事实"
-- subject（2026-09-15 上线清单②）：记录这条事实的**主体**——
--   self        = 用户本人的情况（可跨会话引用、参与档案一致性判断）
--   third_party = 用户替家人问出来的情况（女儿/父亲…）。
-- 没有这一列时，"女儿25岁湿热"和"我45岁高血压"会被当成**同一个人的
-- 档案变更**——上一轮还在追问"您之前说有糖尿病，现在还在吃吗"，
-- 正是把不同人误判成同一人的档案演化（上线清单②的根因判定）。
-- scope（2026-09-16「记忆按会话隔离」）：
--   session  = 只属于写它的那个会话（**默认**，新开对话看不到）；
--   global   = 用户**明说**"记住"的事实，是唯一允许跨会话的条目；
--   archived = 2026-09-16 之前遗留的"全局记忆"（历史测试数据），
--              永不自动注入，只能在记忆面板里看到并删除。
CREATE TABLE IF NOT EXISTS memories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id    INTEGER,                    -- 来源会话（session_id 语义）
    kind       TEXT    NOT NULL DEFAULT 'fact',   -- fact / preference / health
    text       TEXT    NOT NULL,
    subject    TEXT    NOT NULL DEFAULT 'self',   -- self / third_party
    user_id    TEXT    NOT NULL DEFAULT 'default',-- 用户标识（本地单用户= default）
    scope      TEXT    NOT NULL DEFAULT 'session',-- session / global / archived
    created_at REAL    NOT NULL
);

-- 会话摘要：早期历史被压缩后的「对话纪要」，解决"只有 6 轮记忆"的容量问题
CREATE TABLE IF NOT EXISTS summaries (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id    INTEGER NOT NULL,
    upto_id    INTEGER NOT NULL,           -- 已压缩到哪条消息（messages.id）
    text       TEXT    NOT NULL,
    created_at REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_summaries_conv ON summaries(conv_id);

-- 主诉时间线：**按会话隔离**（2026-09-15 实测串档事故的根治）。
-- 为什么不放 profile：慢病/西药/体质是「这个用户是谁」的事实，跨会话成立；
-- 但"舌苔白腻、大便黏"这类主诉叙事是「这次对话里说了什么」，
-- 换一个人设/换一个话题后，旧会话的症状被新会话当成当前事实引用，
-- 就是 P0 级的辨证污染。所以时间线跟会话走，会话删它也删。
CREATE TABLE IF NOT EXISTS timelines (
    conv_id    INTEGER PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
    items      TEXT    NOT NULL DEFAULT '',
    updated_at REAL    NOT NULL
);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """轻量迁移：老库补新列（CREATE TABLE IF NOT EXISTS 不会改已存在的表）。

    为什么必须补：安全判读一旦落库缺失，用户**刷新页面后安全提示就消失了**——
    对安全功能来说这是不可接受的降级（"这轮提示过，回头再看没了"）。
    """
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(messages)")}
    if "safety" not in cols:
        conn.execute("ALTER TABLE messages ADD COLUMN safety TEXT NOT NULL DEFAULT '{}'")
    # 2026-09-15 上线清单②：memories 补主体与用户标识列（老库平滑升级）
    mcols = {r["name"] for r in conn.execute("PRAGMA table_info(memories)")}
    if "subject" not in mcols:
        conn.execute("ALTER TABLE memories ADD COLUMN subject TEXT NOT NULL DEFAULT 'self'")
    if "user_id" not in mcols:
        conn.execute("ALTER TABLE memories ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default'")
    if "scope" not in mcols:
        # 老条目一律归档：它们都是"全局记忆"时代的产物（含大量测试人设），
        # 直接标成 session 会把它们塞进某个会话，标成 global 又会被跨会话召回。
        conn.execute("ALTER TABLE memories ADD COLUMN scope TEXT NOT NULL DEFAULT 'archived'")
        conn.execute("UPDATE memories SET scope = ? WHERE conv_id IS NOT NULL",
                     (MEM_SESSION,))          # 有来源会话的，本来就只属于那个会话
    _archive_legacy_profile(conn)


def _archive_legacy_profile(conn: sqlite3.Connection) -> None:
    """把旧的**全局** profile 归档到 LEGACY_SCOPE（只搬一次，幂等）。

    不删数据：搬完后档案面板仍能看到（标为"历史归档"），但**任何会话都读不到**——
    "新开的对话没有记忆"要成立，历史全局档案就必须退出注入链路。
    """
    rows = conn.execute("SELECT key, value, updated_at FROM profile").fetchall()
    if not rows:
        return
    conn.executemany(
        "INSERT INTO session_profile (conv_id, key, value, updated_at)"
        " VALUES (?, ?, ?, ?)"
        " ON CONFLICT(conv_id, key) DO UPDATE SET value=excluded.value,"
        " updated_at=excluded.updated_at",
        [(LEGACY_SCOPE, r["key"], r["value"], r["updated_at"]) for r in rows])
    conn.execute("DELETE FROM profile")


def _conn() -> sqlite3.Connection:
    """打开连接：首次自动建表；开外键约束以支持级联删除。"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA)
    _migrate(conn)
    return conn


# ---------- 会话 ----------
def list_conversations() -> list[dict]:
    """全部会话：置顶在前，组内按最近活跃倒序。"""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, title, pinned, created_at, updated_at,"
            " (SELECT count(*) FROM messages m WHERE m.conv_id = c.id) AS msg_count"
            " FROM conversations c"
            " ORDER BY pinned DESC, updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def create_conversation(title: str = "新对话") -> dict:
    now = time.time()
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO conversations (title, created_at, updated_at)"
            " VALUES (?, ?, ?)", (title, now, now))
        conv_id = cur.lastrowid
    return {"id": conv_id, "title": title, "pinned": 0,
            "created_at": now, "updated_at": now, "msg_count": 0}


def delete_conversation(conv_id: int) -> None:
    """删除会话及其全部消息（messages / timelines 外键级联删除）。

    2026-09-16 起档案与记忆也按会话隔离，所以这里要**一并清掉**——
    "记忆只属于这次对话"要成立，删掉对话就必须把它记住的东西也删掉，
    否则残留的档案会以"另一个会话的数据"的形态留在库里，越积越乱。
    """
    with _conn() as conn:
        conn.execute("DELETE FROM messages WHERE conv_id = ?", (conv_id,))
        conn.execute("DELETE FROM session_profile WHERE conv_id = ?", (conv_id,))
        conn.execute("DELETE FROM memories WHERE conv_id = ? AND scope = ?",
                     (conv_id, MEM_SESSION))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))


def rename_conversation(conv_id: int, title: str) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE conversations SET title = ? WHERE id = ?", (title, conv_id))


def set_pinned(conv_id: int, pinned: bool) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE conversations SET pinned = ? WHERE id = ?",
            (1 if pinned else 0, conv_id))


# ---------- 消息 ----------
def add_message(conv_id: int, role: str, content: str,
                sources: list | None = None,
                safety: dict | None = None) -> dict:
    """追加一条消息并刷新会话 updated_at（会话自动顶到列表上方）。"""
    now = time.time()
    src_json = json.dumps(sources or [], ensure_ascii=False)
    saf_json = json.dumps(safety or {}, ensure_ascii=False)
    with _conn() as conn:
        cur = conn.execute(
            "INSERT INTO messages (conv_id, role, content, sources, safety, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (conv_id, role, content, src_json, saf_json, now))
        msg_id = cur.lastrowid
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id))
    return {"id": msg_id, "conv_id": conv_id, "role": role,
            "content": content, "sources": sources or [],
            "safety": safety or {}, "created_at": now}


def get_messages(conv_id: int) -> list[dict]:
    """某会话的全部消息（按时间正序）。"""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, role, content, sources, safety, created_at"
            " FROM messages WHERE conv_id = ? ORDER BY id ASC", (conv_id,)
        ).fetchall()
    def _load(v, default):
        try:
            return json.loads(v) if v else default
        except (TypeError, ValueError):
            return default
    return [{"id": r["id"], "role": r["role"], "content": r["content"],
             "sources": _load(r["sources"], []),
             "safety": _load(r["safety"], {}),
             "created_at": r["created_at"]}
            for r in rows]


# ---------- 问诊状态机 ----------
def save_consult_state(conv_id: int, stage: str, payload: str) -> None:
    """写入/覆盖某会话的问诊状态（upsert，一个会话只保留最新一条）。"""
    with _conn() as conn:
        conn.execute(
            "INSERT INTO consult_state (conv_id, stage, payload, updated_at)"
            " VALUES (?, ?, ?, ?)"
            " ON CONFLICT(conv_id) DO UPDATE SET"
            " stage=excluded.stage, payload=excluded.payload,"
            " updated_at=excluded.updated_at",
            (conv_id, stage, payload, time.time()))


def load_consult_state(conv_id: int) -> dict | None:
    with _conn() as conn:
        r = conn.execute(
            "SELECT stage, payload, updated_at FROM consult_state WHERE conv_id = ?",
            (conv_id,)).fetchone()
    return dict(r) if r else None


# ---------- 会话档案（按 conv_id 隔离）----------
def scope_of(conv_id: int | None) -> int:
    """把"会话 id"归一成作用域 id：None（CLI / 无会话）→ SESSION_SCOPE。"""
    return SESSION_SCOPE if conv_id is None else int(conv_id)


def get_profile(conv_id: int | None = None) -> dict[str, str]:
    """**本会话**的结构化档案。

    语义变化（2026-09-16）：以前是"这个产品的唯一用户档案"，现在是
    "这次对话里了解到的情况"。新开的对话 → 空字典（不再继承任何历史）。
    """
    with _conn() as conn:
        rows = conn.execute(
            "SELECT key, value FROM session_profile WHERE conv_id = ?"
            " ORDER BY updated_at DESC", (scope_of(conv_id),)).fetchall()
    return {r["key"]: r["value"] for r in rows}


def set_profile(key: str, value: str, conv_id: int | None = None) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO session_profile (conv_id, key, value, updated_at)"
            " VALUES (?, ?, ?, ?)"
            " ON CONFLICT(conv_id, key) DO UPDATE SET value=excluded.value,"
            " updated_at=excluded.updated_at",
            (scope_of(conv_id), key, value, time.time()))


def delete_profile(key: str, conv_id: int | None = None) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM session_profile WHERE conv_id = ? AND key = ?",
                     (scope_of(conv_id), key))


def clear_profile(conv_id: int | None = None) -> None:
    """清空某会话的档案（测试与"重新开始"用）。"""
    with _conn() as conn:
        conn.execute("DELETE FROM session_profile WHERE conv_id = ?",
                     (scope_of(conv_id),))


def archived_profile() -> dict[str, str]:
    """历史遗留的全局档案（只读展示，**不参与任何会话的上下文**）。"""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT key, value FROM session_profile WHERE conv_id = ?"
            " ORDER BY updated_at DESC", (LEGACY_SCOPE,)).fetchall()
    return {r["key"]: r["value"] for r in rows}


# ---------- 记忆条目（scope 决定它能被谁看到）----------
def add_memory(text: str, kind: str = "fact", conv_id: int | None = None,
               subject: str = "self", user_id: str = "default",
               scope: str = MEM_SESSION) -> dict:
    """沉淀一条记忆。去重：文本完全相同则不重复插入。

    subject：事实主体——'self'=用户本人 / 'third_party'=替家人问的
    （2026-09-15 上线清单②：没有主体区分时，"女儿25岁湿热"会被当成
    用户本人的档案变更，跨会话追问时张冠李戴）。
    scope：能见范围——默认 'session'（只有本会话看得到）；
    只有用户**明说"记住"**时才写 'global'（见 memory.py::remember_turn）。
    """
    text = text.strip()
    now = time.time()
    with _conn() as conn:
        dup = conn.execute(
            "SELECT id FROM memories WHERE text = ? AND scope = ?",
            (text, scope)).fetchone()
        if dup:
            return {"id": dup["id"], "text": text, "kind": kind,
                    "conv_id": conv_id, "subject": subject, "scope": scope,
                    "user_id": user_id, "created_at": now, "duplicated": True}
        cur = conn.execute(
            "INSERT INTO memories (conv_id, kind, text, subject, user_id,"
            " scope, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (conv_id, kind, text, subject, user_id, scope, now))
        mid = cur.lastrowid
    return {"id": mid, "text": text, "kind": kind, "conv_id": conv_id,
            "subject": subject, "scope": scope, "user_id": user_id,
            "created_at": now, "duplicated": False}


def list_memories(limit: int = 200, conv_id: int | None = None,
                  scope: str | None = None,
                  include_archived: bool = False) -> list[dict]:
    """列记忆条目（作用域过滤是这里的核心语义）。

    - 不传 scope：**本会话的 session 条目** + 全部 global 条目
      （即"这次对话里我记住了什么" + "用户明确要求我一直记得的"）；
    - scope='session'：只取**本会话**的（记忆召回走这条）；
    - scope='global'：只要用户明说「记住」的跨会话条目；
    - include_archived=True：把 2026-09-16 之前的历史全局条目也带上
      （**只给记忆面板看**，不参与任何注入）。
    """
    cid = scope_of(conv_id)
    args: list = []
    if scope:
        cond = "scope = ?"
        args.append(scope)
        if scope == MEM_SESSION:
            cond += " AND conv_id = ?"
            args.append(cid)
    else:
        cond = "(conv_id = ? AND scope = ?) OR scope = ?"
        args.extend([cid, MEM_SESSION, MEM_GLOBAL])
        if include_archived:
            cond = f"({cond}) OR scope = ?"
            args.append(MEM_ARCHIVED)
        else:
            cond = f"({cond})"
    sql = ("SELECT id, conv_id, kind, text, subject, user_id, scope, created_at"
           " FROM memories WHERE " + cond + " ORDER BY id DESC LIMIT ?")
    args.append(limit)
    with _conn() as conn:
        rows = conn.execute(sql, args).fetchall()
    return [dict(r) for r in rows]


def count_archived_memories() -> int:
    with _conn() as conn:
        r = conn.execute("SELECT count(*) AS n FROM memories WHERE scope = ?",
                         (MEM_ARCHIVED,)).fetchone()
    return int(r["n"]) if r else 0


def delete_memory(mid: int) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM memories WHERE id = ?", (mid,))


# ---------- 主诉时间线（按会话隔离） ----------
def get_timeline(conv_id: int | None) -> str:
    """某会话的主诉时间线原文（分号分隔）；无记录返回空串。"""
    if conv_id is None:
        return ""
    with _conn() as conn:
        row = conn.execute(
            "SELECT items FROM timelines WHERE conv_id = ?", (conv_id,)).fetchone()
    return row["items"] if row else ""


def set_timeline(conv_id: int | None, items: str) -> None:
    """写某会话的主诉时间线（整串覆盖）。conv_id 为空时不写（无主可归）。"""
    if conv_id is None:
        return
    now = time.time()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO timelines (conv_id, items, updated_at) VALUES (?, ?, ?)"
            " ON CONFLICT(conv_id) DO UPDATE SET items = excluded.items,"
            " updated_at = excluded.updated_at",
            (conv_id, items, now))


# ---------- 会话摘要（历史压缩的产物） ----------
def add_summary(conv_id: int, upto_id: int, text: str) -> dict:
    """新增一段摘要（覆盖同一 upto_id 的旧摘要）。"""
    now = time.time()
    with _conn() as conn:
        old = conn.execute(
            "SELECT id FROM summaries WHERE conv_id = ? AND upto_id = ?",
            (conv_id, upto_id)).fetchone()
        if old:
            conn.execute("UPDATE summaries SET text = ?, created_at = ?"
                         " WHERE id = ?", (text, now, old["id"]))
            return {"id": old["id"], "upto_id": upto_id, "text": text}
        cur = conn.execute(
            "INSERT INTO summaries (conv_id, upto_id, text, created_at)"
            " VALUES (?, ?, ?, ?)", (conv_id, upto_id, text, now))
    return {"id": cur.lastrowid, "upto_id": upto_id, "text": text}


def get_summaries(conv_id: int) -> list[dict]:
    """按压缩范围正序返回该会话的全部摘要。"""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, upto_id, text, created_at FROM summaries"
            " WHERE conv_id = ? ORDER BY upto_id ASC", (conv_id,)).fetchall()
    return [dict(r) for r in rows]
