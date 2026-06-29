import gradio as gr
import os
from openai import OpenAI
from pypdf import PdfReader

# DeepSeek client
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def analyze(file):
    if file is None:
        return "请上传PDF文件"

    text = read_pdf(file)

    prompt = f"""
请对以下财报进行结构化分析：

1. 收入变化
2. 利润变化
3. 现金流情况
4. 增长质量判断
5. 风险点（最多3条）
6. 一句话总结

财报内容：
{text}
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是专业股票投研分析师"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


with gr.Blocks(title="AI投研分析工具") as demo:

    gr.Markdown("# 🧠 AI投研分析工具（DeepSeek版）")

    file_input = gr.File(label="上传财报PDF")

    btn = gr.Button("开始分析")

    output = gr.Textbox(lines=20)

    btn.click(fn=analyze, inputs=file_input, outputs=output)

demo.launch(server_name="0.0.0.0", server_port=10000)
