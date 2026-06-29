import gradio as gr
import os
from openai import OpenAI
from pypdf import PdfReader

# DeepSeek API
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 读取PDF
def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


# 核心分析函数
def analyze(file):
    if file is None:
        return "请上传财报PDF"

    text = read_pdf(file)

    prompt = f"""
你是一名专业股票分析师，请对以下财报进行分析，并严格按格式输出：

【要求输出结构】

📊 公司健康评分（0-100）：
- 请给出一个总分，并简要解释

📈 收入质量（A/B/C）：
💰 利润质量（A/B/C）：
💵 现金流质量（A/B/C）：

⚠️ 风险提示（最多3条）：
1.
2.
3.

🧠 投资结论（一句话）：

【评分规则】
- 收入增长稳定 + 高分
- 利润质量好 + 高分
- 现金流健康 + 高分
- 风险越多分越低

【财报内容】
{text}
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是资深港美股投研分析师"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


# UI界面升级
with gr.Blocks(title="AI投研评分系统") as demo:

    gr.Markdown("# 🧠 AI投研评分系统（升级版）")
    gr.Markdown("上传财报PDF，自动生成公司健康评分 + 投资分析报告")

    with gr.Row():
        file_input = gr.File(label="上传财报PDF")
        btn = gr.Button("🚀 开始分析")

    output = gr.Textbox(
        label="AI分析报告",
        lines=25
    )

    btn.click(fn=analyze, inputs=file_input, outputs=output)

demo.launch(server_name="0.0.0.0", server_port=10000)
