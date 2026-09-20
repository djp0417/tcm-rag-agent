# -*- coding: utf-8 -*-
"""MCP **协议层**回归（默认零 API）：真的起一个 Server 子进程，走 JSON-RPC 调它。

为什么必须有这一层（而不是只测纯函数）：
    `tools/test_mcp_tools.py` 验的是"逻辑对不对"，本文件验的是"协议通不通"。
    两者必须分开 —— 否则一个断言失败，你分不清是规则库错了还是握手错了。

    更重要的是：**泄漏守卫必须加在协议出口断言一次**。
    历史上"内部规则原文外泄"就是出在"守卫只挂在某一条出口"上。
    MCP 是新增入口，如果只在 Web 出口挂守卫，这里就会漏。

跑法：
    python -m tools.test_mcp_server              # 零 API（推荐，秒级）
    MCP_TEST_SEARCH=1 python -m tools.test_mcp_server   # 额外测 tcm_search（**会调 embedding API**）

判绿只认退出码。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OK = 0
BAD = 0

# 用户没提过、但内部规则里会出现的药名（逐名断言，防止"概化没兜住"）
_MUST_NOT_LEAK = ("阿司匹林", "华法林", "氯吡格雷", "水蛭", "麝香", "巴豆",
                  "甘遂", "朱砂", "雄黄", "丹参", "三七", "红参", "川芎",
                  "他汀", "红曲", "血脂康", "附子", "川乌", "益母草", "桃仁")

# 内部流程语言（第五轮退化：这些词被模型说给了用户）
_MUST_NOT_FLOW = ("再补充几点", "前面没有展开", "系统检查发现",
                  "挂起针对性结论", "只给不分型")


def check(label: str, ok: bool, detail: str = "") -> bool:
    global OK, BAD
    if ok:
        OK += 1
        print(f"  ✅ {label}")
    else:
        BAD += 1
        print(f"  ❌ {label}" + (f" —— {detail}" if detail else ""))
    return ok


def _text_of(res) -> str:
    """从 CallToolResult 里取回文本（结构化结果优先，退回 content[0].text）。"""
    sc = getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)
    if sc:
        return json.dumps(sc, ensure_ascii=False)
    parts = []
    for c in (getattr(res, "content", None) or []):
        t = getattr(c, "text", None)
        if t:
            parts.append(t)
    return "\n".join(parts)


def _as_dict(res) -> dict:
    sc = getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)
    if isinstance(sc, dict):
        return sc
    txt = _text_of(res)
    try:
        return json.loads(txt)
    except Exception:
        return {"_raw": txt}


def _is_loud_failure(res) -> bool:
    """这次调用是否**响亮失败**（协议层报错 / 结构化错误 / 参数校验被拒）。

    SDK 版本间错误标记可能叫 `isError` 或 `is_error`，而参数校验失败返回的是
    一段纯文本而非结构化 JSON —— 「失败是否响亮」这条断言必须三种形态都认。
    """
    for attr in ("isError", "is_error"):
        if getattr(res, attr, None):
            return True
    d = _as_dict(res)
    if d.get("ok") is False:
        return True
    txt = _text_of(res) + json.dumps(d, ensure_ascii=False)
    return any(k in txt for k in ("validation error", "Field required", "参数校验",
                                  "Error executing tool"))


def _is_err(res) -> bool:
    return bool(getattr(res, "is_error", None) or getattr(res, "isError", None))


# ---------------------------------------------------------------------------
def _server_params():
    from mcp.client.stdio import StdioServerParameters
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp"],
        cwd=str(ROOT),          # 自测时显式给；真实运行由 _boot 自切兜底
        env=dict(os.environ),
    )


async def _run() -> int:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    with_search = os.environ.get("MCP_TEST_SEARCH") == "1"

    async with stdio_client(_server_params()) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()

            # ============ 1. 工具清单 ============
            print("\n【工具清单 / 描述】")
            tools = {t.name: t for t in (await s.list_tools()).tools}
            for name in ("tcm_search", "tcm_safety_check",
                         "tcm_constitution", "tcm_intake_gaps"):
                check(f"声明了 {name}", name in tools, f"实际={sorted(tools)}")
            check("工具数为 4",
                  len(tools) == 4, f"实际={len(tools)}: {sorted(tools)}")
            empties = [n for n, t in tools.items()
                       if not (getattr(t, "description", "") or "").strip()]
            check("每个工具都有非空 description", not empties, f"空的={empties}")
            hard = ("何时用", "何时不用")
            missing = [n for n, t in tools.items()
                       if not all(k in (t.description or "") for k in hard)]
            check("每个工具的 description 都写了「何时用/何时不用」",
                  not missing, f"缺的={missing}")
            check("description 未泄露内部流程话术",
                  not [k for k in _MUST_NOT_FLOW
                       if any(k in (t.description or "") for t in tools.values())])

            # ============ 2. 资源清单 ============
            print("\n【资源清单】")
            try:
                rl = (await s.list_resources()).resources
                uris = [str(x.uri) for x in rl]
            except Exception as e:  # noqa: BLE001
                uris = []
                print(f"    （list_resources 失败：{type(e).__name__}: {e}）")
            check("tcm://kb/manifest 已注册",
                  any("kb/manifest" in u for u in uris), f"实际={uris}")
            check("tcm://guide/scope 已注册",
                  any("guide/scope" in u for u in uris), f"实际={uris}")

            # ============ 3. 安全判读：孕妇 + 当归 ============
            print("\n【安全判读 · 孕妇 + 当归 → 必须「明确禁止」】")
            res = await s.call_tool("tcm_safety_check", {
                "items": ["当归"],
                "states_text": "我怀孕5个月了，最近咳嗽，想喝当归鸡汤",
            })
            check("调用未报 is_error", not _is_err(res), _text_of(res)[:200])
            d = _as_dict(res)
            check("ok 字段为真", d.get("ok") is True, str(d)[:200])
            items = d.get("items") or []
            huoxue = [i for i in items
                      if i.get("matched") == "当归" or "活血" in str(i.get("item", ""))]
            check("命中「当归」对应项", bool(huoxue),
                  f"items={[i.get('item') for i in items]}")
            if huoxue:
                it = huoxue[0]
                check("当归档位 = forbid（明确禁止）",
                      it.get("tier") == "forbid", f"实际={it.get('tier')}")
                check("档位中文名 = 明确禁止",
                      it.get("tier_label") == "明确禁止", str(it.get("tier_label")))
                basis = str(it.get("basis", "")) + str(it.get("reading", ""))
                check("理由绑定妊娠/孕产（不得只说「需确认」）",
                      any(k in basis for k in ("孕产", "孕期", "妊娠")),
                      basis[:120])
            check("状态识别出妊娠",
                  "妊娠" in str(d.get("states")), str(d.get("states"))[:200])
            check("states 用「人话版」note（无内部药名）",
                  not [k for k in _MUST_NOT_LEAK
                       if k in str((d.get("states") or {}).get("notes"))])

            blob = json.dumps(d, ensure_ascii=False)
            leaks = [k for k in _MUST_NOT_LEAK if k in blob]
            check("零泄漏：返回里没有用户未提及的药名", not leaks, f"泄漏={leaks}")
            flows = [k for k in _MUST_NOT_FLOW if k in blob]
            check("零流程语言：返回里没有内部流程话术", not flows, f"命中={flows}")
            og = d.get("output_guard") or {}
            check("出口守卫未残留漏网名", int(og.get("remaining_count") or 0) == 0,
                  str(og))

            # ============ 4. 对照组：川贝（应可执行，与当归明显不同） ============
            print("\n【对照 · 川贝炖梨 → 应「可执行」，且与当归档位明显不同】")
            res2 = await s.call_tool("tcm_safety_check", {
                "items": ["川贝", "梨", "冰糖"],
                "states_text": "我怀孕5个月了，最近咳嗽",
            })
            d2 = _as_dict(res2)
            check("对照组调用成功", d2.get("ok") is True, str(d2)[:200])
            cb = [i for i in (d2.get("items") or [])
                  if i.get("matched") == "川贝" or "川贝" in str(i.get("item", ""))]
            check("命中「川贝」对应项", bool(cb),
                  f"items={[i.get('item') for i in (d2.get('items') or [])]}")
            if cb:
                check("川贝档位 = ok（可执行）",
                      cb[0].get("tier") == "ok", f"实际={cb[0].get('tier')}")
            if huoxue and cb:
                check("当归与川贝档位明显不同（区分度没丢）",
                      huoxue[0].get("tier") != cb[0].get("tier"),
                      f"{huoxue[0].get('tier')} vs {cb[0].get('tier')}")

            # ============ 4b. 无规则条目必须显式出现（不消失） ============
            print("\n【无规则条目 · 不得从结果里消失】")
            res2b = await s.call_tool("tcm_safety_check", {
                "items": ["川贝", "鸡肉"],
                "states_text": "我怀孕5个月了，最近咳嗽",
            })
            d2b = _as_dict(res2b)
            rows2b = d2b.get("items") or []
            by2b = {i.get("matched"): i for i in rows2b}
            gone = [w for w in (d2b.get("evaluated") or []) if w not in by2b]
            check("evaluated 里每个词都在 items 里有交代",
                  not gone,
                  f"消失={gone}；items={[i.get('item') for i in rows2b]}")
            ck = by2b.get("鸡肉")
            check("「鸡肉」以 has_rule=false / tier=null 出现（不是 ok）",
                  bool(ck) and ck.get("has_rule") is False
                  and ck.get("tier") is None, str(ck)[:200])
            check("带 no_rule_policy（客户端据此解读 tier=null）",
                  "已确认安全" in str(d2b.get("no_rule_policy") or ""),
                  str(d2b.get("no_rule_policy"))[:160])
            check("带 scope_discipline（自行食用 vs 医师处方）",
                  "自行食用" in str(d2b.get("scope_discipline") or ""),
                  str(d2b.get("scope_discipline"))[:160])

            # ============ 5. 资源可读 ============
            print("\n【资源可读】")
            try:
                mr = await s.read_resource("tcm://kb/manifest")
                txt = "\n".join(getattr(c, "text", "") for c in mr.contents)
                mj = json.loads(txt)
                check("语料清单可读且 ok", mj.get("ok") is True, txt[:160])
                check("语料清单有文件数与块数",
                      int(mj.get("file_count") or 0) > 0
                      and int(mj.get("total_chunks") or 0) > 0,
                      f"files={mj.get('file_count')} chunks={mj.get('total_chunks')}")
            except Exception as e:  # noqa: BLE001
                check("语料清单可读", False, f"{type(e).__name__}: {e}")

            try:
                sr = await s.read_resource("tcm://guide/scope")
                stxt = "\n".join(getattr(c, "text", "") for c in sr.contents)
                check("能力边界可读且写了「不能做什么」",
                      "不能做" in stxt, stxt[:160])
            except Exception as e:  # noqa: BLE001
                check("能力边界可读", False, f"{type(e).__name__}: {e}")

            # ============ 6. 会话档案资源（模板） ============
            print("\n【会话档案资源（只读）】")
            try:
                pr = await s.read_resource("tcm://session/1/profile")
                ptxt = "\n".join(getattr(c, "text", "") for c in pr.contents)
                pj = json.loads(ptxt)
                check("会话档案可读", pj.get("ok") is True, ptxt[:160])
                check("档案资源标明只读", pj.get("read_only") is True, ptxt[:160])
            except Exception as e:  # noqa: BLE001
                check("会话档案可读", False, f"{type(e).__name__}: {e}")

            # ============ 7. 错误处理（不吞异常，返回结构化错误） ============
            print("\n【参数校验】")
            # 显式传空串 = "我确认用户没提状态"；两个都空 → 我们自己的结构化错误
            res3 = await s.call_tool("tcm_safety_check",
                                     {"items": [], "states_text": ""})
            d3 = _as_dict(res3)
            check("空参数返回结构化错误而不是崩",
                  d3.get("ok") is False and "error" in d3, str(d3)[:200])

            # ★ 漏传 states_text 必须**响亮失败**，不能静默按空处理。
            #   2026-09-17 部署校验实测的坑：参数名写成 `states` 时，SDK 对多余字段
            #   是静默丢弃的，于是妊娠状态没进判定、当归返回"可执行"且毫无报错。
            #   安全判读最坏的失败模式就是漏报禁忌 —— 所以缺必填参数必须报错。
            res3b = await s.call_tool("tcm_safety_check", {"items": ["当归"]})
            check("漏传 states_text → 报错（不静默取默认值）",
                  _is_loud_failure(res3b), _text_of(res3b)[:200])

            # 参数名写错（states != states_text）同样必须失败 —— 这正是当初踩的坑
            res3c = await s.call_tool("tcm_safety_check",
                                      {"items": ["当归"], "states": "我怀孕5个月了"})
            check("参数名写错（states）→ 报错而不是静默丢参",
                  _is_loud_failure(res3c), _text_of(res3c)[:200])

            # ============ 8. 可选的 API 用例：检索 ============
            if with_search:
                print("\n【检索（会调 embedding API）】")
                res4 = await s.call_tool("tcm_search",
                                         {"query": "秋天干燥应该注意什么"})
                d4 = _as_dict(res4)
                check("检索返回结果", d4.get("ok") is True and d4.get("count", 0) > 0,
                      str(d4)[:200])
                if d4.get("results"):
                    r0 = d4["results"][0]
                    check("结果含 source/chapter/text",
                          all(k in r0 for k in ("source", "chapter", "text")),
                          str(r0)[:160])
            else:
                print("\n【检索】跳过（零 API 模式；要测请加 MCP_TEST_SEARCH=1）")

        # ============ 9. 模拟"不认 cwd 的宿主"——WorkBuddy 就是这样 ============
        # 这一节是**接真实宿主前必须过的关**：
        #   `python -m app.mcp` 要能 import 到 app 包，靠的是 sys.path；
        #   而宿主拉起子进程时的 cwd 是未知的（实测 WorkBuddy 的 mcpServers
        #   没有 cwd 字段）。所以配置里必须带 PYTHONPATH=<项目根>，
        #   让 import 与 chdir 两件事各有着落：
        #       PYTHONPATH → 找得到 app 包；_boot 自 chdir → chroma 相对路径能开库。
        print("\n【模拟不认 cwd 的宿主（WorkBuddy 实测如此）】")
        import tempfile

        from mcp.client.stdio import StdioServerParameters

        keep = ("PATH", "SYSTEMROOT", "SYSTEMDRIVE", "TEMP", "TMP", "WINDIR",
                "COMSPEC", "PATHEXT", "USERPROFILE", "APPDATA", "LOCALAPPDATA",
                "PROGRAMDATA", "NUMBER_OF_PROCESSORS", "PYTHONIOENCODING")
        env_min = {k: v for k, v in os.environ.items() if k in keep}
        check("模拟 env 里没有 API Key",
              not any("API_KEY" in k for k in env_min), str(sorted(env_min)))

        # ---- 负对照：cwd 错 + 没有 PYTHONPATH → 必须起不来 ----
        bad = StdioServerParameters(command=sys.executable, args=["-m", "app.mcp"],
                                    cwd=tempfile.gettempdir(), env=env_min)
        failed = False
        try:
            async with stdio_client(bad) as (rb, wb):
                async with ClientSession(rb, wb) as sb:
                    await sb.initialize()
        except Exception:  # noqa: BLE001 —— 这里"抛异常"才是期望结果
            failed = True
        check("负对照：cwd 错且无 PYTHONPATH 时确实起不来（所以配置必须带 PYTHONPATH）",
              failed, "居然起来了 —— 说明断言前提不成立，需重新确认")

        # ---- 正例：cwd 错 + 有 PYTHONPATH → 照常工作（这就是 mcp.json 的写法） ----
        env_min["PYTHONPATH"] = str(ROOT)
        good = StdioServerParameters(command=sys.executable, args=["-m", "app.mcp"],
                                     cwd=tempfile.gettempdir(), env=env_min)
        async with stdio_client(good) as (rg, wg):
            async with ClientSession(rg, wg) as sg:
                await sg.initialize()
                t2 = {x.name for x in (await sg.list_tools()).tools}
                check("cwd 错 + 有 PYTHONPATH → 工具清单正常",
                      {"tcm_search", "tcm_safety_check"} <= t2, str(sorted(t2)))
                r6 = await sg.call_tool("tcm_safety_check", {
                    "items": ["当归"],
                    "states_text": "我怀孕5个月了，最近咳嗽"})
                d6 = _as_dict(r6)
                hit6 = [i for i in (d6.get("items") or [])
                        if i.get("matched") == "当归"]
                check("cwd 错 + 有 PYTHONPATH → 档位仍为 forbid",
                      bool(hit6) and hit6[0].get("tier") == "forbid",
                      str(d6)[:200])
                check("cwd 错 + 有 PYTHONPATH → 零泄漏",
                      not [k for k in _MUST_NOT_LEAK
                           if k in json.dumps(d6, ensure_ascii=False)])

    print(f"\n{'=' * 56}")
    print(f"MCP 协议层回归：通过 {OK} 项，失败 {BAD} 项")
    return 1 if BAD else 0


def main() -> int:
    try:
        return asyncio.run(_run())
    except Exception:
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
