import gradio as gr
import os
from openai import OpenAI
from prompt import PROMPT_TEMPLATE

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def analyze(text):
    if not text:
        return "请输入财报内容"

    prompt = PROMPT_TEMPLATE.format(text=text)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "你是一个专业的股票财报分析助手"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


with gr.Blocks(title="AI投研实验室") as demo:

    gr.Markdown("# 🧪 AI投研实验室")
    gr.Markdown("上传或粘贴财报内容，AI帮你做结构化分析")

    input_box = gr.Textbox(
        label="输入财报文本",
        placeholder="粘贴财报、公告或公司介绍..."
    )

    btn = gr.Button("开始分析")

    output = gr.Textbox(
        label="AI分析结果",
        lines=20
    )

    btn.click(fn=analyze, inputs=input_box, outputs=output)

demo.launch(
    server_name="0.0.0.0",
    server_port=10000
)
