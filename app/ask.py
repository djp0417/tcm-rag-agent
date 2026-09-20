# -*- coding: utf-8 -*-
"""RAG 问答命令行入口（底层逻辑在 app/rag.py 的 RAGSession）。

多轮说明：交互模式下整个会话共用一个 RAGSession，
追问（如"那饮食上呢"）会先被改写成独立问题再检索，可正确命中。

用法（须在项目根目录以模块方式运行）：
    单次提问：  python -m app.ask -q "阳虚体质冬天要注意什么"
    交互聊天：  python -m app.ask        （输入 quit 退出）
"""
import argparse

from app.rag import RAGSession


def print_result(r) -> None:
    """打印回答 + 改写后的问题 + 引用来源（多轮/排查时很有用）。"""
    if r.standalone != r.question:
        print(f"  [改写] {r.standalone}")
    print("\n回答:")
    print(r.answer)
    if r.sources:
        print("\n引用来源:")
        for s in r.sources:
            print(f"  - {s['source']} · {s['chapter']}（相关度 {s['score']}）")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--question", help="单次提问；不填则进入交互模式")
    args = parser.parse_args()

    if args.question:
        print_result(RAGSession().ask(args.question))
        return

    session = RAGSession()
    print("=== 中医养生科普 RAG 问答（输入 quit 退出）===")
    while True:
        try:
            q = input("\n你的问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit", "exit", "退出"):
            break
        if not q:
            continue
        try:
            print_result(session.ask(q))
        except SystemExit as e:
            print(f"[error] {e}")
        except Exception as e:                            # noqa: BLE001
            print(f"[error] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
