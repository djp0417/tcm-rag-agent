# -*- coding: utf-8 -*-
"""验收「记忆按会话隔离」后，新会话第一轮不再被旧人设/档案糊脸。

背景（真实事故）
----------------
用户新开一个会话、只说「累、想补补」，安全卡立刻命中 6 项（附子/降压药/阿胶…），
回答还复述了旧测试人设的话（「胖了十来斤」「阴虚质」）。方向与直觉相反：
**不是没带上下文，而是把过期上下文当成了新事实**。

两个场景
--------
场景 A：新会话、档案为空 + 「累、想补补」→ 允许 0 命中（追问可以有）。
场景 B：**会话内**有高血压/氨氯地平档案 + 同一句话 →
        命中只允许以「来自你档案的提醒」形式出现（origin=profile），
        不得作为本轮事实展开（origin=message），也不得把旧档案内容
        当成新事实大段写进回答。

⚠️ 2026-09-16 重写：旧版把档案写到 /api/profile（无 conv_id → 无会话作用域），
然后在新会话里聊天——两者作用域不同，会话读不到那份档案，于是场景 B
的「✅ 通过」是**假通过**（它其实一直在验"空档案"）。现在档案一律写到
本次测试自己的会话里，删会话即完全还原，不再需要全局快照/还原那一套。

用法：
    先起临时实例  python -m app.server --port 7861
    再运行        E2E_PORT=7861 python -m tools.test_fresh_session
"""
from __future__ import annotations

import json
import os
import urllib.request

# 默认打 7861（临时实例）。7860 留给用户前台起的服务，测试不占。
BASE = f"http://127.0.0.1:{os.environ.get('E2E_PORT', '7861')}"

# 本机有系统代理时，urllib 会把 127.0.0.1 的请求也扔给代理 → 502。
urllib.request.install_opener(
    urllib.request.build_opener(urllib.request.ProxyHandler({})))

MSG = "我最近老是觉得累，睡够了也没精神，是不是虚啊？该吃点啥补补？"


def _req(path, method="GET", payload=None):
    data = json.dumps(payload or {}).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method=method)
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())


def _stream(path, message):
    req = urllib.request.Request(
        BASE + path, data=json.dumps({"message": message}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    events = []
    with urllib.request.urlopen(req, timeout=300) as r:
        buf = ""
        for raw in r:
            buf += raw.decode("utf-8", "replace")
            while "\n\n" in buf:
                frame, buf = buf.split("\n\n", 1)
                if frame.startswith("data: "):
                    events.append(json.loads(frame[6:]))
    return events


def run_case(tag: str, profile: dict) -> bool:
    """在**新建会话**里设置档案 → 问同一句话 → 断言 → 删会话。

    档案写在会话作用域上（`?conv_id=`），所以删掉会话就等于完全还原，
    既不碰用户真实数据，也不留任何跨会话残留。
    """
    conv = _req("/api/conversations", "POST", {"title": f"fresh-{tag}"})
    cid = conv["id"]
    for k, v in profile.items():
        _req(f"/api/profile/{urllib.request.quote(k, safe='')}?conv_id={cid}",
             "PUT", {"value": v})

    evs = _stream(f"/api/conversations/{cid}/chat", MSG)
    sev = next((e for e in evs if e["type"] == "safety"), None)
    ans = next((e["answer"] for e in evs if e["type"] == "done"), "")
    hits = (sev or {}).get("hits") or []
    # origin 缺省视为 message（本轮提到）——与本轮命中的判据保持一致
    msg_hits = [h for h in hits if (h.get("origin") or "message") == "message"]
    prof_hits = [h for h in hits if h.get("origin") == "profile"]

    print(f"\n【{tag}】本会话档案={profile or '空'}")
    print(f"  安全事件: {'无' if not sev else f'命中{len(hits)}项(本轮{len(msg_hits)}+档案{len(prof_hits)})'}"
          f" stop={(sev or {}).get('stop')}")
    for h in hits:
        print(f"    - {h['name']:<10} origin={h.get('origin'):<8} level={h['level']}")

    # 1) 本轮没提到的东西，不许作为"本轮事实"命中（否则就是有东西在糊脸）
    ok = not msg_hits
    # 2) 干净档案 → 不许有任何档案来源命中
    if not profile:
        ok &= not prof_hits
    # 3) 有档案 → 档案提醒应当出现（旧版假通过的位置：这里必须是 True）
    else:
        ok &= bool(prof_hits)
    # 4) 干净档案时回答开头不得凭空冒出附子（旧事故的直观表征）
    if not profile and "附子" in ans[:400]:
        ok = False
    print(f"  回答开头: {ans[:90]}…")
    print(f"  {'✅ 通过' if ok else '❌ 不通过'}"
          f"（本轮命中={'空' if not msg_hits else len(msg_hits)}，"
          f"档案提醒={'无' if not prof_hits else len(prof_hits)}）")

    _req(f"/api/conversations/{cid}", "DELETE")     # 档案/记忆随之清除
    # 复核：删会话后档案确实读不到了（隔离闭环）
    left = _req(f"/api/profile?conv_id={cid}")
    if left:
        print(f"  ⚠️ 删除会话后仍能读到档案：{left}")
        ok = False
    return ok


def main() -> int:
    ok1 = run_case("场景A：干净档案", {})
    ok2 = run_case("场景B：本会话慢病档案", {
        "慢病": "高血压", "西药": "氨氯地平（降压药）"})
    print("\n" + "=" * 60)
    print("结果：", "通过" if (ok1 and ok2) else "存在缺口")
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
