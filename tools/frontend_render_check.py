# -*- coding: utf-8 -*-
"""前端回归：语法 + 渲染。零 API、秒级，改 `web/*.html` 后必须跑。

为什么前端需要自己的回归
------------------------
`web/index.html` 与 `web/agent.html` 的界面全是模板字符串拼出来的 HTML，
Python 侧的一切回归（安全层、架构层、端到端）都碰不到它们。于是有两类
**后端全绿、界面全错**的问题只能在这里发现：

  ① 语法类：模板串少一个 `}` / `)` —— 浏览器直接白屏，Python 侧毫无察觉；
  ② 契约类：字段名写错或用了废弃字段。最典型的一次是本轮改造：
     后端已经把判读文本按「档位 × 用户条件」条件化（`Scan.SafetyHit.reading`），
     前端却还在读 `verdict`（rules.py 里按最坏情况写死的通用话术）。
     结果是"血虚无湿的人问阿胶，界面仍说'不适合'"——**不报错、只是变差**。

做法：把内联 `<script>` 抽出来，用极小 DOM 桩载进 Node，再喂一份**真实事件
载荷**（字段名照抄 app/safety/scan.py::SafetyHit.to_dict），断言渲染结果里
五档档位、档位依据、状态约束、挂起说明、换人设确认、引用可采信性都在，
且**没有**退回旧 verdict。

用法：python tools/frontend_render_check.py [--page web/index.html]
"""
from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
# 优先用 PATH 里的 node；找不到再回退到常见安装位置（不绑定任何个人路径）
NODE = shutil.which("node") or shutil.which("node.exe") or "node"
PAGES = ("web/index.html", "web/agent.html")

# ---------------------------------------------------------------------------
# 真实事件载荷（与后端 app/safety/scan.py::SafetyHit.to_dict 字段一一对应）
# ---------------------------------------------------------------------------
EVENT = {
    "type": "safety", "stop": True,
    "hits": [
        # forbid：毒性药材 + 与档案交叉
        {"kind": "herb", "key": "fuzi", "name": "附子", "level": "high",
         "sub": "toxic", "why": "附子有毒，需炮制与控量。",
         "not_for": "孕妇、心律失常者", "verdict": "（旧版通用话术）",
         "matched": "附子", "tags": ["hypertension", "renal"],
         "tag_labels": ["高血压", "肾功能不全"], "taking": True, "escalated": True,
         "origin": "message", "tier": "forbid", "tier_label": "明确禁止",
         "cond_labels": ["高血压", "肾功能不全"], "unassessed_gaps": [],
         "reading": "**不建议自行服用**。附子有毒、需炮制与控量；你**正在服用**降压药，"
                    "叠加起来风险更高。\n· 立即停用\n· 与中医师确认"},
        # conditional：命中条件 → 有条件
        {"kind": "herb", "key": "ejiao", "name": "阿胶", "level": "medium",
         "sub": "food", "why": "阿胶滋腻碍胃。", "not_for": "", "verdict": "",
         "matched": "阿胶", "tags": ["damp", "damp_heat"],
         "tag_labels": ["痰湿/湿困", "湿热"], "taking": False, "escalated": False,
         "origin": "message", "tier": "conditional", "tier_label": "对证但有条件",
         "cond_labels": ["痰湿/湿困"], "unassessed_gaps": [],
         "reading": "**现在不适合，先化湿再补**。你{[痰湿/湿困]}的表现正是禁忌面。"},
        # confirm：档位依据来自后端 tier_basis（不是前端拼 cond_labels）
        {"kind": "herb", "key": "mahuang", "name": "麻黄", "level": "medium",
         "sub": "herb", "why": "麻黄会升压。", "not_for": "", "verdict": "",
         "matched": "麻黄", "tags": ["hypertension"], "tag_labels": ["高血压"],
         "taking": False, "escalated": False, "origin": "message",
         "tier": "confirm", "tier_label": "需专业确认",
         "tier_basis": "这一味必须先分品种（红参温 / 生晒参平 / 西洋参凉），须中医师定",
         "cond_labels": [], "unassessed_gaps": ["tongue"],
         "reading": "**先别急着用，这一条我还判断不了**。**结论先挂起**：把下面几个问题答一下。"},
        # confirm + 条件未评估 → 结论挂起
        {"kind": "herb", "key": "renshen", "name": "人参", "level": "medium",
         "sub": "herb", "why": "参类分温平凉三性。", "not_for": "", "verdict": "",
         "matched": "人参", "tags": [], "tag_labels": [],
         "taking": False, "escalated": False, "origin": "message",
         "tier": "confirm", "tier_label": "需专业确认",
         "tier_basis": "条件未评估，结论先挂起：还缺舌象",
         "cond_labels": [], "unassessed_gaps": ["tongue"],
         "reading": "**先别急着用，这一条我还判断不了**。**结论先挂起**：把下面几个问题答一下。"},
        # not_needed：第三轮新增第五档——"对你没用"≠"对你有害"，必须与 forbid 分开配色
        {"kind": "herb", "key": "lurong", "name": "鹿茸", "level": "low",
         "sub": "herb", "why": "鹿茸温补壮阳。", "not_for": "", "verdict": "",
         "matched": "鹿茸", "tags": ["damp_heat"], "tag_labels": ["湿热"],
         "taking": False, "escalated": False, "origin": "message",
         "tier": "not_needed", "tier_label": "不建议（无适应症）",
         "tier_basis": "未见需要它的依据（它针对的是「阳虚」）",
         "cond_labels": [], "unassessed_gaps": [],
         "reading": "**不必吃**。你现在的表现里没有它对应的证，吃它不会解决问题。"},
        # origin=profile → 折叠区
        {"kind": "drug", "key": "antihypertensive", "name": "降压药", "level": "high",
         "sub": "drug", "why": "不可自行停减。", "not_for": "", "verdict": "",
         "matched": "氨氯地平", "tags": ["hypertension"], "tag_labels": ["高血压"],
         "taking": True, "escalated": False, "origin": "profile",
         "tier": "confirm", "tier_label": "需专业确认", "cond_labels": [],
         "unassessed_gaps": [], "reading": "**不能自行停药**。"},
    ],
    "tags": ["hypertension"], "conflicts": [], "screening": ["hypertension"],
    "gaps": ["tongue"],
    # 状态类约束：优先级高于一切药名规则，前端必须单独成块、排在命中之前
    "states": [{"key": "pregnancy", "label": "妊娠期", "matched": "怀孕",
                "note": "孕产期用药必须由产科/中医师确认，活血与毒性药一律先停"}],
    "questions": [{"key": "tongue", "label": "舌象", "ask": "舌头什么样？"}],
    "subject_switch": True,
    "subject_switch_detail": {"prev_label": "寒（畏寒怕冷、喜热）",
                              "cur_label": "热（口苦苔黄腻 / 潮热盗汗）",
                              "axis": "寒↔热", "where": "本会话此前的描述"},
    "plan": None,
}

# 引用来源载荷：传说/巫术性记载在取用侧已被剔除或降级（app/credibility.py），
# 前端必须把"仅作文化背景 / 不可采信"显式标出来，不能让用户以为那是正经依据。
SOURCES = [
    {"source": "《神农本草经》.pdf", "chapter": "上品", "score": 0.82,
     "credibility": "ok", "credibility_label": "可采信"},
    {"source": "抱朴子.txt", "chapter": "仙药", "score": 0.71,
     "credibility": "background", "credibility_label": "仅作文化背景"},
]

DWARF = r"""
// ---- 极小 DOM 桩：内联脚本加载期会绑定大量控件，一律吞掉 ----
const stub = new Proxy(function () {}, {
  get: (t, k) => {
    if (k === 'style' || k === 'dataset') return {};
    if (k === 'classList') return { add(){}, remove(){}, toggle(){}, contains(){ return false; } };
    if (k === 'value' || k === 'innerHTML' || k === 'textContent') return '';
    if (k === Symbol.toPrimitive) return () => '';
    return stub;
  },
  set: () => true, apply: () => stub, construct: () => stub,
});
global.window = stub;
global.document = stub;
global.localStorage = stub;
global.location = stub;
global.EventSource = stub;
global.fetch = async () => new Promise(() => {});  // 挂起即可：桩里不发网络请求，
// 也**不能 throw**——页面脚本加载期会调 loadConvs()，抛错会变成 unhandled rejection
global.requestAnimationFrame = () => 0;
"""

ASSERT_JS = r"""
const ev = %s;
const srcs = %s;
let out = '';
try { out = safetyHtml(ev); }
catch (e) { console.log('RENDER_FAIL ' + e.message); process.exit(2); }
let srcOut = '';
try { srcOut = sourcesHtml(srcs); }
catch (e) { console.log('RENDER_FAIL sources: ' + e.message); process.exit(2); }
console.log(JSON.stringify({safety: out, sources: srcOut}));
"""


def _blocks(page: str) -> list[str]:
    html = (ROOT / page).read_text(encoding="utf-8")
    return re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, re.S)


def syntax(page: str) -> bool:
    """node --check：模板字符串少个括号会白屏，这一步专抓它。"""
    tmp = ROOT / ("_syntax_%s.js" % pathlib.Path(page).stem)
    tmp.write_text("\n;\n".join(_blocks(page)), encoding="utf-8")
    r = subprocess.run([NODE, "--check", str(tmp)],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    tmp.unlink()
    ok = r.returncode == 0
    print(f"  语法 node --check ：{'✅ OK' if ok else '❌ ' + (r.stderr or '')[:300]}")
    return ok


def render(page: str) -> bool:
    """用真实事件载荷跑一遍 safetyHtml()/sourcesHtml()，断言五档/依据/挂起/换人设/引用都渲染出来。"""
    js = (DWARF + "\n;\n".join(_blocks(page)) + "\n"
          + ASSERT_JS % (json.dumps(EVENT, ensure_ascii=False),
                         json.dumps(SOURCES, ensure_ascii=False)))
    tmp = ROOT / ("_render_%s.js" % pathlib.Path(page).stem)
    tmp.write_text(js, encoding="utf-8")
    r = subprocess.run([NODE, str(tmp)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    tmp.unlink()
    line = (r.stdout or "").strip().splitlines()
    if not line:
        print(f"  渲染 safetyHtml  ：❌ 无输出 {(r.stderr or '')[:300]}")
        return False
    if line[-1].startswith("RENDER_FAIL"):
        print(f"  渲染 safetyHtml  ：❌ {line[-1]}")
        return False
    d = json.loads(line[-1])
    out, src = d.get("safety", ""), d.get("sources", "")
    need = ['明确禁止', '不建议（无适应症）', '对证但有条件', '需专业确认',
            '档位依据', '未见需要它的依据', '你的状态约束', '妊娠期',
            '结论已挂起', '先确认一件事', '本会话此前的描述', '所以对你而言']
    miss = [s for s in need if s not in out]
    stale = '旧版通用话术' in out          # 未退回 rules.py 写死的 verdict
    stars = bool(re.search(r"\*\*", re.sub(r"<[^>]+>", "", out)))  # ** 未转成 <b>
    src_miss = [s for s in ['引用来源', '仅作文化背景'] if s not in src]
    ok = not (miss or stale or stars or src_miss)
    print(f"  渲染 safetyHtml  ：{'✅' if ok else '❌'} 长度 {len(out)}"
          f"{'｜缺 ' + '、'.join(miss) if miss else ''}"
          f"{'｜退回旧 verdict' if stale else ''}"
          f"{'｜** 未渲染' if stars else ''}"
          f"{'｜引用徽章缺 ' + '、'.join(src_miss) if src_miss else ''}")
    return ok


def main() -> int:
    pages = PAGES
    if "--page" in sys.argv:
        pages = (sys.argv[sys.argv.index("--page") + 1],)
    all_ok = True
    for p in pages:
        print(f"\n【{p}】")
        all_ok &= syntax(p)
        all_ok &= render(p)
    print("\n" + "=" * 56)
    print("前端回归：", "全部通过" if all_ok else "存在问题")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
