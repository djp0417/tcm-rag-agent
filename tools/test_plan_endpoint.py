# -*- coding: utf-8 -*-
"""验收用例 #6：多智能体可见性——打 /api/plan 端点，验证三专家产出。

用法：先起临时服务（--port 7861），再运行本脚本（E2E_PORT 可改端口）。
"""
from __future__ import annotations

import json
import os
import urllib.request

BASE = f"http://127.0.0.1:{os.environ.get('E2E_PORT', '7861')}"
os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")


def _req(path: str, method: str = "GET", payload: dict | None = None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(req, timeout=600) as r:
        body = r.read().decode()
    return json.loads(body) if body.strip().startswith(("{", "[")) else body


def _stream(path: str, payload: dict) -> list[dict]:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    events: list[dict] = []
    with urllib.request.urlopen(req, timeout=600) as r:
        buf = ""
        for raw in r:
            buf += raw.decode("utf-8", "replace")
            while "\n\n" in buf:
                frame, buf = buf.split("\n\n", 1)
                if frame.startswith("data: "):
                    events.append(json.loads(frame[6:]))
    return events


def main() -> int:
    # 快照 → 清档案 → 跑 → 还原（共享库，测试残留会漏进真实会话）
    snap = dict(_req("/api/profile"))
    for k in list(snap):
        _req(f"/api/profile/{urllib.request.quote(k, safe='')}", "DELETE")
    try:
        return _run()
    finally:
        for k in list(_req("/api/profile")):
            _req(f"/api/profile/{urllib.request.quote(k, safe='')}", "DELETE")
        for k, v in snap.items():
            _req(f"/api/profile/{urllib.request.quote(k, safe='')}", "PUT",
                 {"value": str(v)})
        print(f"（档案已还原：{len(snap)} 项）")


def _run() -> int:
    ok = True

    # 场景 0：没做体质辨识就要方案 → 不得硬跑流水线（引导或快答皆可）
    conv0 = _req("/api/conversations", "POST", {})
    evs0 = _stream(f"/api/plan/{conv0['id']}/chat",
                   {"message": "帮我出一份调理方案"})
    nodes0 = [e.get("node") for e in evs0 if e["type"] == "step"]
    errs0 = [e for e in evs0 if e["type"] == "error"]
    if errs0:
        print("  ⚠️ 端点返回 error 事件:", errs0)
    ans0 = next((e.get("answer", "") for e in evs0 if e["type"] == "done"), "")
    experts = {"diet_expert", "meridian_expert", "movement_expert"}
    guided = not errs0 and not (experts & set(nodes0)) and len(ans0) > 50
    print("【场景0】无问诊记录要方案 → 不硬跑流水线:", "通过" if guided else "不通过",
          f"（nodes={nodes0}，{len(ans0)} 字）")
    ok &= guided

    # 场景 1：先用零-LLM 的 quick 答完 27 题，再要方案 → 三专家产出
    conv = _req("/api/conversations", "POST", {})
    cid = conv["id"]
    # 开场：先发一句话让状态机进入 COLLECTING 并吐第一题
    _stream(f"/api/agent/{cid}/chat", {"message": "我想测一下自己是什么体质"})
    n = 0
    while n < 40:
        evs = _stream(f"/api/agent/{cid}/chat",
                      {"message": "（点选作答）", "quick": {"score": 2}})
        n += 1
        stage_ev = next((e for e in evs if e["type"] == "stage"), None)
        if stage_ev and stage_ev.get("stage") != "collecting":
            break
        if stage_ev and stage_ev.get("answered", 0) >= stage_ev.get("total", 27):
            break
        if not any(e["type"] == "question" for e in evs):
            break
    st = _req(f"/api/agent/{cid}/state")
    print(f"【准备】quick 答题 {st['answered']}/{st['total']}，阶段={st['stage']}")
    ok &= st["answered"] >= st["total"]

    # 自述关键信息（否则信息缺口触发 need_more 追问——那是 P1-7 的正确行为）
    _stream(f"/api/agent/{cid}/chat", {"message":
           "补充一下：我45岁女性，有高血压在吃氨氯地平，平时怕冷手脚凉，"
           "大便黏马桶不成形，舌苔白腻有齿痕，最近容易累"})
    print("【准备】已补充自述（年龄性别/慢病西药/舌象/二便/寒热）")

    try:
        print("【验收#6】多智能体可见性：/api/plan 端点")
        evs = _stream(f"/api/plan/{cid}/chat",
                      {"message": "我怕冷、大便黏、总觉得累，帮我出一份系统调理方案"})
        # 兜底异常会伪装成 error 事件且端点照回 200（真事故：改签名漏改 wrapper →
        # TypeError 被吞 → 节点/答案全空却"看日志一切正常"）。显式断言 + 打印全部类型。
        from collections import Counter
        print("  事件类型统计:", dict(Counter(e["type"] for e in evs)))
        errs = [e for e in evs if e["type"] == "error"]
        if errs:
            print("  ⚠️ 端点返回 error 事件:", errs)
        ok &= not errs
        nodes = [e.get("node") for e in evs if e["type"] == "step"]
        print("  节点序列:", nodes)
        experts = {"diet_expert", "meridian_expert", "movement_expert"}
        hit_experts = experts & set(nodes)
        print("  专家节点:", sorted(hit_experts))
        ans = next((e.get("answer", "") for e in evs if e["type"] == "done"), "")
        print(f"  最终答案 {len(ans)} 字；含穴位: {'穴' in ans or '三里' in ans}；"
              f"含运动: {'八段锦' in ans or '快走' in ans}")
        safety = next((e for e in evs if e["type"] == "safety"), None)
        print("  安全事件:", bool(safety), "| 命中", len((safety or {}).get("hits") or []))
        ok &= bool(hit_experts) and len(ans) > 200
        print("  结果:", "通过" if ok else "不通过")
    finally:
        for c in (cid, conv0["id"]):
            req = urllib.request.Request(f"{BASE}/api/conversations/{c}", method="DELETE")
            urllib.request.urlopen(req).read()
        print("（测试会话已删除）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
