# -*- coding: utf-8 -*-
"""多 Agent 流水线命令行入口（Step 1 联调用）。

用法
----
    # 看有哪些会话已完成体质辨识（流水线需要这个前提）
    python -m app.multiagent --list

    # 打印流程图（mermaid，可贴进任何支持 mermaid 的地方）
    python -m app.multiagent --graph

    # 跑一次完整链路
    python -m app.multiagent --conv 12 "帮我出一份完整的调理方案"

    # 连跑多句（观察不同分流）
    python -m app.multiagent --conv 12 "你好" "我最近胸口疼" "帮我出一份调理方案"

    # 把最终状态导成 JSON（给评估脚本用）
    python -m app.multiagent --conv 12 "帮我出一份调理方案" --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys

from app.multiagent.graph import mermaid, run_pipeline
from app.multiagent.state import render


def _list_ready() -> None:
    """列出已完成体质辨识的会话（流水线的入口前提）。

    直接扫 `consult_state` 表而不是遍历 `conversations`：
    两者并非总是一一对应——早期调试留下的问诊状态，其会话行可能已经被删，
    但问诊结果本身完好（实测就踩到过：conv 13 是阳虚质 91.7 分，
    却因为 conversations 表里没有它而被漏掉）。**判断依据只看问诊结果本身。**
    """
    import sqlite3

    from app import storage
    from app.agent.state import ConsultState

    rows = []
    with sqlite3.connect(storage.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("""
            SELECT s.conv_id, s.payload, c.title
            FROM consult_state s
            LEFT JOIN conversations c ON c.id = s.conv_id
        """)
        for r in cur.fetchall():
            cs = ConsultState.from_json(r["payload"])
            if not (cs.total and cs.answered >= cs.total):
                continue
            rows.append((r["conv_id"], r["title"] or "（无标题）",
                         cs.primary_key, cs.answered, cs.total))
    rows.sort(key=lambda x: -x[0])
    if not rows:
        print("没有找到已完成体质辨识的会话。"
              "先在 Web 问诊页（/agent）把 27 题答完，再来跑流水线。")
        return
    print(f"{'会话':>6}  {'进度':>9}  主体质        标题")
    print("-" * 72)
    for cid, title, pk, ans, tot in rows:
        print(f"{cid:>6}  {ans:>4}/{tot:<4}  {pk or '（未判定）':<12}  {title[:34]}")


def _progress(node: str, delta: dict) -> None:
    """每个节点跑完打一行（等价于前端"思考过程"的骨架）。"""
    marks = {"router": "🧭", "collector": "📋", "diagnoser": "🔍", "planner": "📝",
             "safety": "🛡️", "editor": "✍️", "fast": "⚡", "urgent": "🚑",
             "need_consult": "📝"}
    detail = {k: v for k, v in (delta or {}).items()
              if k not in ("evidence", "trace", "final_answer")}
    ms = ""
    for t in (delta or {}).get("trace", []) or []:
        if t.get("node") == node:
            ms = f"{t.get('ms')} ms"
    brief = json.dumps(detail, ensure_ascii=False, default=str)
    print(f"  {marks.get(node, '·')} {node:<12} {ms:>8}  {brief[:110]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m app.multiagent",
        description="多 Agent 协作流水线（问诊 → 辨证 → 方案 → 安全审查 → 主编）")
    ap.add_argument("texts", nargs="*", help="用户输入（可给多句，依次跑）")
    ap.add_argument("--conv", type=int, help="会话 id（流水线需要已完成的体质辨识）")
    ap.add_argument("--list", action="store_true", help="列出可用的会话")
    ap.add_argument("--graph", action="store_true", help="打印流程图（mermaid）")
    ap.add_argument("--selftest", action="store_true",
                    help="控制流自检（用剧本替换 LLM，确定性验证回环与降级）")
    ap.add_argument("--json", metavar="FILE", help="把最终状态导出为 JSON")
    ap.add_argument("--quiet", action="store_true", help="不打印逐节点进度")
    args = ap.parse_args(argv)

    if args.graph:
        print(mermaid())
        return 0
    if args.selftest:
        from app.multiagent.selftest import run as run_selftest
        return run_selftest()
    if args.list:
        _list_ready()
        return 0
    if not args.conv:
        ap.error("需要 --conv <会话id>（用 --list 查看可用会话）")
    if not args.texts:
        ap.error("请给出至少一句用户输入")

    last_state: dict = {}
    for i, text in enumerate(args.texts, 1):
        print("=" * 72)
        print(f"[{i}/{len(args.texts)}] 用户：{text}")
        print("-" * 72)
        state = run_pipeline(args.conv, text,
                             on_step=None if args.quiet else _progress)
        last_state = state
        print(f"\n路由：{state.get('route')} —— {state.get('route_reason', '')}")
        if state.get("degraded"):
            print("⚠️ 已降级交付（安全审查未通过且重写次数用尽）")
        print("\n" + (state.get("final_answer") or "（无输出）"))
        print()

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(last_state, f, ensure_ascii=False, indent=2, default=str)
        print(f"[ok] 最终状态已写入 {args.json}")

    if not args.quiet:
        print(render(last_state))
    return 0


if __name__ == "__main__":
    sys.exit(main())
