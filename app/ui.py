# -*- coding: utf-8 -*-
"""Gradio Web 界面：把 RAG 问答开放给普通用户（无需命令行）。

功能：
- 多轮对话聊天窗口（底层共用一个 RAGSession，追问可正确检索）；
- 回答末尾自动附【引用来源】（文件 · 篇章 · 相关度），可核查、可溯源；
- "开启新话题"按钮一键清空对话记忆；
- 生成式流式输出：先回显问题，再等完整回答（避免界面卡死观感）。

用法（须在项目根目录以模块方式运行）：
    python -m app.ui
    → 自动打开浏览器 http://127.0.0.1:7860（局域网访问加 share=False, server_name="0.0.0.0"）
"""
import gradio as gr

from app.rag import RAGSession

session = RAGSession()


def respond(message: str, history: list[dict]):
    """Gradio 回调：收到新消息 → 调 RAGSession → 更新聊天记录。

    history 由 Gradio 维护（type="messages" 格式），
    检索用的长期记忆由 session 自己管理，两边各司其职。
    """
    message = message.strip()
    if not message:
        yield history, ""
        return
    history = history + [{"role": "user", "content": message}]
    yield history, ""                                   # 先把用户消息刷上屏

    result = session.ask(message)
    sources = "\n".join(
        f"- 📄 {s['source']} · {s['chapter']}（相关度 {s['score']}）"
        for s in result.sources
    )
    reply = f"{result.answer}\n\n---\n📚 **引用来源**\n{sources}"
    history.append({"role": "assistant", "content": reply})
    yield history, ""


def new_topic():
    """清空对话记忆，开启新话题（向量库与已入库语料不受影响）。"""
    session.reset()
    return []


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="中医养生科普助手") as demo:
        gr.Markdown(
            "# 🌿 中医养生科普助手\n"
            "基于本地知识库（《黄帝内经·素问》全本 + 中医体质/四季养生科普）的 RAG 问答，"
            "回答仅依据资料库内容并附引用来源。"
            "**本助手只做养生科普，不构成医疗建议。**"
        )
        # Gradio 6：Chatbot 只用 messages 格式（role/content 字典），
        # 不再有 type 参数（那是 4.x/5.x 时代的兼容开关）
        chatbot = gr.Chatbot(height=480, label="对话")
        msg = gr.Textbox(
            placeholder="试试：春天应该怎么养生？／阳虚体质有什么表现？／晚上几点睡比较好？",
            label="你的问题",
        )
        with gr.Row():
            send = gr.Button("发送", variant="primary")
            clear = gr.Button("🧹 开启新话题（清空记忆）")

        send.click(respond, [msg, chatbot], [chatbot, msg])
        msg.submit(respond, [msg, chatbot], [chatbot, msg])
        clear.click(new_topic, None, chatbot)
    return demo


if __name__ == "__main__":
    build_ui().launch(inbrowser=True)
