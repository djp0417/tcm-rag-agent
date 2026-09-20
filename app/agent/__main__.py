# -*- coding: utf-8 -*-
"""问诊 Agent 命令行入口：快速联调状态机与工具调用链路。

用法（项目根目录）：
    python -m app.agent                 # 新建会话，进入交互问诊
    python -m app.agent --conv 3        # 接着 3 号会话继续
    python -m app.agent --demo          # 自动跑一遍完整问诊（验收用）
    python -m app.agent --state 3       # 查看 3 号会话的问诊状态
"""
import argparse
import sys

from app import storage
from app.agent.agent import ConsultAgent
from app.agent.state import ConsultState, STAGE_LABEL

# --demo 用的模拟回答，**顺序必须与量表题目顺序严格一致**：
#   平和3 → 气虚3 → 阳虚3 → 阴虚3 → 痰湿3 → 湿热3 → 血瘀3 → 气郁3 → 特禀3
# 这套答案刻意塑造成「阳虚 + 气虚」的画像，用于验证判定逻辑是否落在预期上。
DEMO_ANSWERS = [
    "精力不太充沛，容易累", "睡眠还可以", "胃口一般，面色不太好",
    "很容易累，整天没什么精神", "爬两层楼就有点喘", "换季很容易感冒",
    "特别怕冷，手脚总是冰凉的", "很怕冷，冬天要穿很厚", "吃凉的容易拉肚子",
    "手脚心不热，反而偏凉", "不觉得口干", "很少失眠，也不盗汗",
    "偶尔觉得身体有点沉", "脸上出油不多，嘴里不发黏", "痰不多，也不太胸闷",
    "很少长痘，脸上不油", "口不苦，没有异味", "大便正常，小便不黄",
    "皮肤不太容易有瘀斑", "面色偏黄，黑眼圈有一点", "唇色正常，记性还行",
    "心情还不错", "不太紧张焦虑", "不是多愁善感的性格",
    "不过敏", "皮肤不起风团", "很少打喷嚏鼻塞",
]

C_DIM, C_CYAN, C_GREEN, C_RST = "\033[2m", "\033[36m", "\033[32m", "\033[0m"


def render(ev: dict, verbose: bool = True) -> None:
    """把非 delta 事件渲染成终端提示行。"""
    t = ev["type"]
    if not verbose:
        return
    if t == "stage":
        print(f"  {C_DIM}· {ev['label']}  [{ev['answered']}/{ev['total']}]{C_RST}")
    elif t == "tool":
        print(f"  {C_DIM}→ {ev['label']}{C_RST}")
    elif t == "reflection":
        for a in ev["attempts"]:
            mark = "命中" if a["accepted"] else "不足"
            print(f"  {C_DIM}  检索第{a['round']}轮 top1={a['top_score']} {mark}"
                  f" ←「{a['query']}」{C_RST}")
    elif t == "question":
        # 出题事件：题目由系统展示（Web 端是点选卡片），终端里直接打印
        print(f"{C_CYAN}第{ev['index']}/{ev['total']}题（{ev['type_name']}）{C_RST} {ev['question']}")
        print(f"  {C_DIM}1没有 · 2很少 · 3有时 · 4经常 · 5总是{C_RST}")
    elif t == "sources":
        for s in ev["sources"][:4]:
            print(f"  {C_DIM}  来源 {s['source']} · {s['chapter']} ({s['score']}){C_RST}")
    elif t == "constitution":
        line = f"  判定 主体质：{ev['primary']}"
        if ev["tendencies"]:
            line += f"  兼夹：{'、'.join(ev['tendencies'])}"
        print(f"{C_CYAN}{line}{C_RST}")
    elif t == "memory":
        s = ev["stats"]
        print(f"  {C_DIM}记忆 历史{s.get('total_messages', 0)}条 "
              f"注入{s.get('recent_kept', 0)}条 档案{s.get('profile_keys', 0)}项 "
              f"召回{s.get('memories_recalled', 0)}条{C_RST}")
    elif t == "error":
        print(f"\n\033[31m[错误] {ev['text']}{C_RST}")


def run_turn(agent: ConsultAgent, text: str, verbose: bool = True) -> None:
    """跑一轮并渲染事件流。

    两个标志位要分开记：
      · answered   —— 本轮是否已经输出过正文（**整轮持久**，跨过过程行也不重置）；
      · line_open  —— 当前是否停在正文行上（遇到过程行要先断行）。
    早期版本只用一个 started 标志，结果过程行（stage/tool）把它重置后，
    结尾的 done 事件又把同一段回答打印了一遍——实测踩到的重复输出 bug。
    """
    print(f"\n{C_CYAN}用户> {text}{C_RST}")
    answered = False
    line_open = False
    for ev in agent.ask_stream(text):
        t = ev["type"]
        if t == "delta":
            if not line_open:
                print(f"{C_GREEN}助手> {C_RST}", end="")
                line_open = True
            print(ev["text"], end="", flush=True)
            answered = True
        elif t == "done":
            if line_open:
                print()
                line_open = False
            if not answered:                     # 没有流式输出时兜底显示全文
                print(f"{C_GREEN}助手> {C_RST}{ev['answer']}")
                answered = True
        else:
            if line_open:
                print()
                line_open = False
            render(ev, verbose)
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="中医问诊 Agent（体质辨识状态机）")
    ap.add_argument("--conv", type=int, help="指定会话 id（默认新建）")
    ap.add_argument("--demo", action="store_true", help="自动跑完整问诊（验收用）")
    ap.add_argument("--state", type=int, help="查看指定会话的问诊状态后退出")
    ap.add_argument("--quiet", action="store_true", help="只输出回答，不显示过程")
    args = ap.parse_args()
    verbose = not args.quiet

    if args.state:
        st = ConsultState.load(args.state)
        print(f"会话 {args.state}：阶段 = {STAGE_LABEL[st.stage]}（{st.stage}）")
        print(f"进度：{st.answered}/{st.total}  主体质：{st.primary_key or '未判定'}")
        print(st.summary_for_llm())
        return

    conv_id = args.conv
    if conv_id is None:
        conv = storage.create_conversation("问诊（体质辨识）")
        conv_id = conv["id"]
        print(f"[新建会话] id={conv_id}")

    agent = ConsultAgent(conv_id)

    if args.demo:
        print("=== --demo：自动跑一遍完整体质辨识 ===")
        run_turn(agent, "我想测一下自己是什么体质", verbose)
        idx = 0
        for _turn in range(len(DEMO_ANSWERS) + 8):        # 留冗余，防止追问补答
            st = ConsultState.load(conv_id)
            if st.stage != "collecting" or idx >= len(DEMO_ANSWERS):
                break
            run_turn(agent, DEMO_ANSWERS[idx], verbose)
            idx += 1
        print("\n=== 问诊结束，最终状态 ===")
        st = ConsultState.load(conv_id)
        print(f"阶段：{STAGE_LABEL[st.stage]}  进度：{st.answered}/{st.total}")
        print(st.last_result or "（无判定结果）")
        print("\n=== 长期档案 ===")
        for k, v in storage.get_profile().items():
            print(f"  {k}：{v}")
        return

    print("=== 中医问诊 Agent（输入 quit 退出）===")
    print("提示：先说『我想测一下自己是什么体质』即可开始。\n")
    while True:
        try:
            text = input(f"\n{C_CYAN}用户> {C_RST}").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in ("quit", "exit", "退出"):
            break
        if not text:
            continue
        try:
            run_turn(agent, text, verbose)
        except Exception as e:                    # noqa: BLE001
            print(f"[error] {type(e).__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
