from flask import Flask, request, jsonify
from transformers import LlamaTokenizer, LlamaForMultimodal, CLIPProcessor
from PIL import Image
import torch

app = Flask(__name__)

# 初始化模型和分词器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = LlamaTokenizer.from_pretrained("path_to_llama_multimodal_model")
model = LlamaForMultimodal.from_pretrained("path_to_llama_multimodal_model").to(device)
processor = CLIPProcessor.from_pretrained("path_to_llama_multimodal_model")

@app.route("/generate", methods=["POST"])
def generate():
    # 获取请求中的图像和文本
    text = request.form.get("text")
    image = Image.open(request.files["image"])

    # 处理输入
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True).to(device)

    # 生成响应
    with torch.no_grad():
        outputs = model.generate(**inputs)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 返回生成的文本
    return jsonify({"generated_text": generated_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
