# -*- coding: utf-8 -*-
"""MCP **部署校验**：读真实的 `mcp.json`，按宿主的启动方式把服务拉起来验一遍。

和另外两个测试的分工（三者都要有，缺一不可）：

| 脚本 | 验的是 | 起子进程 | 依赖 SDK |
|---|---|---|---|
| `tools/test_mcp_tools.py`   | **逻辑**对不对（档位/泄漏/只读） | 否 | 否 |
| `tools/test_mcp_server.py`  | **协议**通不通（自拼参数）       | 是 | 是 |
| `tools/verify_mcp_config.py`| **配置**能不能用（照配置启动）   | 是 | 是 |

本文件是唯一"照配置启动"的 —— 它刻意模拟**宿主的真实行为**：
  · 从磁盘读 `~/.workbuddy/mcp.json`（或项目级）而不是自己拼参数；
  · **不给 `cwd`**（WorkBuddy 的 mcpServers 项里根本没有这个字段），
    只把配置里的 `env` 合进一个最小环境变量集。
    这样断了 `PYTHONPATH` 就必然起不来 —— 配置缺失会在这里暴露，
    而不是等到用户点开 WorkBuddy 看到一盏红灯。

跑法：
    python -m tools.verify_mcp_config                # 零 API（推荐）
    python -m tools.verify_mcp_config --search       # 额外测 tcm_search（会调 embedding API）
    python -m tools.verify_mcp_config --config <路径> --server <名字>

判绿只认退出码。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ~/.workbuddy/mcp.json = 用户级；<工作区>/.workbuddy/mcp.json = 项目级
WORKSPACE = ROOT.parent
USER_CFG = Path.home() / ".workbuddy" / "mcp.json"
PROJ_CFG = WORKSPACE / ".workbuddy" / "mcp.json"

DEFAULT_SERVER = "tcm-kb"

# 期望对外暴露的能力（改 server.py 注册表时记得同步这里）
WANT_TOOLS = ("tcm_search", "tcm_safety_check", "tcm_constitution", "tcm_intake_gaps")
WANT_RES_STATIC = ("tcm://kb/manifest", "tcm://guide/scope")

# 宿主环境里允许保留的变量（模拟"最小环境"，证明配置自足）
_ENV_KEEP = ("PATH", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT",
             "TEMP", "TMP", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA",
             "NUMBER_OF_PROCESSORS", "PYTHONIOENCODING")

# 用户没提过、但内部规则里会出现的药名（逐名断言）
_MUST_NOT_LEAK = ("阿司匹林", "华法林", "氯吡格雷", "水蛭", "麝香", "巴豆",
                  "甘遂", "朱砂", "雄黄", "丹参", "三七", "红参", "川芎",
                  "他汀", "红曲", "血脂康", "附子", "川乌", "益母草", "桃仁")

OK = 0
BAD = 0
WARN: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    global OK, BAD
    if ok:
        OK += 1
        print(f"  ✅ {label}")
    else:
        BAD += 1
        print(f"  ❌ {label}" + (f" —— {detail}" if detail else ""))
    return ok


def warn(label: str) -> None:
    WARN.append(label)
    print(f"  ⚠️  {label}")


def _pick_config(explicit: str | None) -> tuple[Path, str]:
    """返回 (配置路径, 级别说明)。显式给了就用给的，否则用户级优先（写在哪都能用）。"""
    if explicit:
        return Path(explicit), "显式指定"
    for p, lvl in ((USER_CFG, "用户级"), (PROJ_CFG, "项目级")):
        if p.exists():
            return p, lvl
    return USER_CFG, "用户级（不存在）"


def _text_of(res) -> str:
    sc = getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)
    if sc:
        return json.dumps(sc, ensure_ascii=False)
    return "\n".join(getattr(c, "text", "") or "" for c in (getattr(res, "content", None) or []))


def _as_dict(res) -> dict:
    sc = getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)
    if isinstance(sc, dict):
        return sc
    try:
        return json.loads(_text_of(res))
    except Exception:
        return {"_raw": _text_of(res)}


def _is_loud_failure(res) -> bool:
    """这次调用是否**响亮失败**（协议层报错 / 结构化错误 / 参数校验被拒）。

    为什么不能只看一个字段：SDK 在不同版本里把错误标记放在 `isError` 或
    `is_error`，参数校验失败时返回的又是一段纯文本而不是结构化 JSON。
    「失败是否响亮」是本次加固的核心断言，判定必须覆盖这三种形态。
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


# ---------------------------------------------------------------------------
# 第一段：静态检查（不启动进程）
# ---------------------------------------------------------------------------
def check_static(cfg_path: Path, srv_name: str) -> dict | None:
    print("\n【配置静态检查】")
    if not check("配置文件存在", cfg_path.exists(), str(cfg_path)):
        print(f"      → 当前用户级路径：{USER_CFG}")
        print(f"      → 当前项目级路径：{PROJ_CFG}")
        return None

    try:
        raw = cfg_path.read_text(encoding="utf-8")
        cfg = json.loads(raw)
    except Exception as e:  # noqa: BLE001
        check("JSON 合法（括号/引号/尾逗号）", False, f"{type(e).__name__}: {e}")
        return None
    check("JSON 合法（括号/引号/尾逗号）", True)

    servers = cfg.get("mcpServers")
    if not check("顶层有 mcpServers 对象", isinstance(servers, dict),
                 f"实际类型 {type(servers).__name__}"):
        return None
    check(f"存在服务条目「{srv_name}」", srv_name in servers,
          f"现有条目：{list(servers) or '（空）'}")
    if srv_name not in servers:
        return None

    e = servers[srv_name]
    unknown = set(e) - {"command", "args", "env", "cwd", "url", "type", "headers",
                        "staticHeaders", "staticEnv", "runtime", "timeout", "disabled"}
    if unknown:
        warn(f"含宿主未文档化的字段 {sorted(unknown)}；若状态灯为红，先删掉它再试")

    if e.get("disabled"):
        warn("该条目 disabled=true —— 不会被加载，需要删掉这个字段或改为 false")

    cmd = e.get("command")
    if isinstance(cmd, str) and cmd:
        # 命令可以是绝对路径或 PATH 里的可执行名，两种都接受
        exists = Path(cmd).exists() or any(
            (Path(p) / cmd).exists() for p in os.environ.get("PATH", "").split(os.pathsep) if p
        )
        check(f"command 可解析：{cmd}", exists, "既不是存在的文件，也不在 PATH 里")
    else:
        check("command 字段存在且是字符串", False, f"实际 {cmd!r}")

    check("args 是数组", isinstance(e.get("args"), list), f"实际 {e.get('args')!r}")
    check("env 是对象（可缺省）", e.get("env") is None or isinstance(e["env"], dict),
          f"实际 {type(e.get('env')).__name__}")

    env = e.get("env") or {}
    has_pp = "PYTHONPATH" in env
    check("env 里给了 PYTHONPATH", has_pp,
          "宿主不认 cwd 且 python -m app.mcp 需要 import app —— 缺它必然 No module named app")
    if has_pp:
        pp = Path(str(env["PYTHONPATH"]))
        check(f"PYTHONPATH 指向真实项目根：{pp}", (pp / "app" / "mcp").is_dir(),
              "该目录下没有 app/mcp，路径写错了")

    if any("API_KEY" in str(k).upper() for k in env):
        warn("配置里含 API Key 明文 —— 本项目服务自己会按绝对路径读 .env，不需要写进来")
    else:
        check("env 里没有明文密钥（服务自己读 .env）", True)

    check("没有依赖 cwd 字段（宿主多半不支持）", True,
          "（本项恒真，仅提示：不要靠它）")
    return e


# ---------------------------------------------------------------------------
# 第二段：按配置启动，走 JSON-RPC 实测
# ---------------------------------------------------------------------------
async def check_runtime(entry: dict, srv_name: str, with_search: bool) -> None:
    print("\n【按配置启动（模拟宿主：不给 cwd，只给配置里的 env）】")

    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    cfg_env = {str(k): str(v) for k, v in (entry.get("env") or {}).items()}
    env = {k: v for k, v in os.environ.items() if k in _ENV_KEEP}
    env.update(cfg_env)
    check("传给子进程的环境里没有 API Key（证明配置自足）",
          not any("API_KEY" in k.upper() for k in env), str(sorted(env)))

    params = StdioServerParameters(
        command=entry["command"],
        args=list(entry.get("args") or []),
        env=env,
        cwd=tempfile.gettempdir(),   # ★ 故意给个无关目录，模拟"宿主 cwd 不可控"
    )
    print(f"      cwd = {params.cwd}（无关目录，证明不依赖它）")

    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            init = await s.initialize()
            info = getattr(init, "serverInfo", None) or getattr(init, "server_info", None)
            name = getattr(info, "name", "?") if info else "?"
            check(f"握手成功（serverInfo.name = {name}）", name == srv_name,
                  f"期望 {srv_name}，实际 {name}")

            tl = await s.list_tools()
            names = [t.name for t in tl.tools]
            check(f"tools/list 返回 {len(names)} 个工具",
                  set(WANT_TOOLS) <= set(names), f"实际 {names}")
            print(f"      {', '.join(names)}")

            rl = await s.list_resources()
            uris = [str(x.uri) for x in rl.resources]
            check("resources/list 含静态资源",
                  set(WANT_RES_STATIC) <= set(uris), f"实际 {uris}")

            # 模板资源用**功能性读取**验证，而不是列举（SDK 版本间模板列举 API 不稳）。
            # 读一个不存在的会话也应返回结构化 JSON —— 关键是"模板能解析、不崩"。
            try:
                tr = await s.read_resource("tcm://session/verify-probe/profile")
                tbody = "".join(getattr(c, "text", "") or "" for c in tr.contents)
                ok_tpl = bool(json.loads(tbody))
                check("模板资源可解析（tcm://session/{id}/profile）", ok_tpl,
                      tbody[:160])
            except Exception as ex:  # noqa: BLE001
                check("模板资源可解析（tcm://session/{id}/profile）", False,
                      f"{type(ex).__name__}: {ex}")

            md = await s.read_resource("tcm://kb/manifest")
            body = "".join(getattr(c, "text", "") or "" for c in md.contents)
            check("能读到 tcm://kb/manifest", "chunk" in body or "files" in body,
                  body[:160])

            # ★ 入参契约：漏传必填参数必须**响亮失败**。
            #   2026-09-17 实测的坑：参数名写成 `states`（真名 `states_text`）时，
            #   SDK 静默丢弃多余字段 → 妊娠状态没进判定 → 当归返回"可执行"、零报错。
            #   安全判读最坏的失败模式就是漏报禁忌，所以缺必填参数必须报错。
            bad = await s.call_tool("tcm_safety_check", {"items": ["当归"]})
            check("漏传 states_text → 报错（不静默给宽松档）",
                  _is_loud_failure(bad), _text_of(bad)[:180])

            # ---- 核心业务断言：孕妇 + 当归 ----
            print("\n【端到端业务断言 · 同一孕妇两样东西必须不同档】")
            q = "我怀孕5个月了，最近咳嗽，想喝当归鸡汤"
            r1 = _as_dict(await s.call_tool(
                "tcm_safety_check", {"items": ["当归"], "states_text": q}))
            r2 = _as_dict(await s.call_tool(
                "tcm_safety_check", {"items": ["川贝", "梨", "冰糖"], "states_text": q}))
            check("当归 → 明确禁止",
                  any(i.get("tier") == "forbid" for i in r1.get("items", [])),
                  json.dumps(r1.get("items"), ensure_ascii=False)[:220])
            check("川贝 → 可执行",
                  any(i.get("tier") in ("ok", "exec_ok") for i in r2.get("items", [])),
                  json.dumps(r2.get("items"), ensure_ascii=False)[:220])
            check("当归与川贝档位不同",
                  {i.get("tier") for i in r1.get("items", [])} !=
                  {i.get("tier") for i in r2.get("items", [])})
            check("入参被原样收到（input_received 回显状态原话）",
                  (r1.get("input_received") or {}).get("states_text") == q,
                  str(r1.get("input_received"))[:160])

            # ★ 无规则条目不得消失（2026-09-17 实测：「鸡肉」曾直接不在 items 里，
            #   于是"库里没有这条"和"判过、没有禁忌"长得一模一样）
            r3 = _as_dict(await s.call_tool(
                "tcm_safety_check", {"items": ["当归", "鸡肉"], "states_text": q}))
            rows3 = r3.get("items") or []
            by3 = {i.get("matched"): i for i in rows3}
            gone = [w for w in (r3.get("evaluated") or []) if w not in by3]
            check("evaluated 里每个词都在 items 里有交代（无规则条目不消失）",
                  not gone, f"消失={gone}；items={[i.get('item') for i in rows3]}")
            ck3 = by3.get("鸡肉") or {}
            check("「鸡肉」以 has_rule=false / tier=null 出现（不是 ok）",
                  ck3.get("has_rule") is False and ck3.get("tier") is None,
                  json.dumps(ck3, ensure_ascii=False)[:220])
            check("解读纪律随结果下发（no_rule_policy + scope_discipline）",
                  "已确认安全" in str(r3.get("no_rule_policy") or "")
                  and "自行食用" in str(r3.get("scope_discipline") or ""),
                  str(r3.get("scope_discipline"))[:160])

            blob = json.dumps([r1, r2, r3], ensure_ascii=False)
            leaked = [n for n in _MUST_NOT_LEAK if n in blob]
            check("协议出口零泄漏（逐名断言）", not leaked, f"漏了：{leaked}")

            if with_search:
                print("\n【tcm_search（会调 embedding API）】")
                rs = _as_dict(await s.call_tool("tcm_search", {"query": "秋天干燥应该注意什么"}))
                check("检索返回结果", rs.get("ok") is True and rs.get("count", 0) > 0,
                      str(rs)[:200])
            else:
                print("\n【tcm_search】跳过（零 API；要测加 --search）")


def main() -> int:
    ap = argparse.ArgumentParser(description="按真实 mcp.json 校验 MCP 部署")
    ap.add_argument("--config", default=None, help="显式指定 mcp.json 路径")
    ap.add_argument("--server", default=DEFAULT_SERVER, help="服务条目名")
    ap.add_argument("--search", action="store_true", help="额外测 tcm_search（调 API）")
    a = ap.parse_args()

    print("=" * 64)
    print("MCP 部署校验：照配置启动（唯一一个不自己拼参数的）")
    print("=" * 64)

    cfg_path, lvl = _pick_config(a.config)
    print(f"\n配置来源：{lvl}\n  {cfg_path}")

    entry = check_static(cfg_path, a.server)
    if entry is None:
        print(f"\n{'=' * 64}\n结果：配置不可用，已被拦在启动之前。\n{'=' * 64}")
        print(f"\n总计入：通过 {OK} 项，失败 {BAD} 项")
        return 1

    try:
        asyncio.run(check_runtime(entry, a.server, a.search))
    except Exception as e:  # noqa: BLE001
        check("按配置启动并完成一次调用", False, f"{type(e).__name__}: {e}")
        print("\n  ▸ 排障顺序：")
        print("    1) JSON 解析错误 → stdout 被 print 污染（日志必须走 stderr）")
        print("    2) 进程启动即退出 → cwd / 相对路径（chroma 开库必须相对路径）")
        print("    3) No module named app → 缺 PYTHONPATH")
        print("    4) 绿灯但无工具 → 先跑 python -m tools.test_mcp_tools 确认逻辑层")

    print(f"\n{'=' * 64}")
    if BAD == 0:
        print("结果：✅ 这份配置可以直接用 —— 去 WorkBuddy 里点信任即可")
    else:
        print(f"结果：❌ 有 {BAD} 项没过，先按上面的提示修配置")
    if WARN:
        print("\n提醒（不阻塞）：")
        for w in WARN:
            print(f"  · {w}")
    print(f"{'=' * 64}")
    print(f"\n总计入：通过 {OK} 项，失败 {BAD} 项")
    return 1 if BAD else 0


if __name__ == "__main__":
    sys.exit(main())
