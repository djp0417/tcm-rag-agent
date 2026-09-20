# -*- coding: utf-8 -*-
"""MCP Server：把本项目的领域能力以**标准协议**对外暴露（阶段 1：只读）。

    python -m app.mcp              # stdio（本地客户端，如 WorkBuddy / Claude Desktop）
    python -m app.mcp --http --port 7863   # Streamable HTTP（第二阶段）

为什么这个项目适合被封装成 MCP Server（三个已有地基）：
    1. **能力已工具化**：检索 / 安全判读 / 体质辨识都是独立函数，不依赖 Web 层；
    2. **安全结论是确定性的**：档位由规则库判出，不经 LLM —— 可以作为**结构化字段**
       返回，客户端模型无法改写（若只给自然段，它的 prompt 会把"明确禁止"柔和化成
       "建议谨慎"，我们整套确定性就白做了）；
    3. **已有输出契约层**：`contract.enforce` / `constraints.scrub` 可直接复用在工具出口。

分层（**不要打乱**）：
    _boot.py      启动引导：把工作目录切到项目根（宿主不认 cwd 字段，见文件内说明）
    schemas.py    对外契约文案（description / instructions）—— descript 决定模型会不会调
    tools.py      4 个工具的**纯函数**实现（不 import mcp，可单独测试）
    resources.py  只读资源（语料清单 / 能力边界 / 会话档案）
    server.py     组装 MCPServer（本文件）
"""
