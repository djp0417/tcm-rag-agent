# -*- coding: utf-8 -*-
"""Agent 包：体质辨识状态机 + function calling 主循环。

模块划分（每个文件职责单一）：
    constitution.py  九种体质量表 + 转化分 + 判定规则（**确定性内核，可单测**）
    state.py         问诊状态机（阶段推进 + SQLite 持久化）
    tools.py         工具定义与执行器（按阶段控制工具可见性）
    agent.py         DeepSeek function calling 主循环（组装事件流）
    __main__.py      命令行入口，用于快速联调

注意：本文件**刻意保持为空**，不要在这里 import 子模块——
state.py / tools.py / agent.py 之间存在相互引用，在此处提前导入会触发循环导入。
"""
