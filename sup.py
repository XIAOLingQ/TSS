import os
import shutil
from io import BytesIO
import gradio as gr
Frist_chat = True

import requests
from PIL import Image, ImageDraw
from ultralytics import YOLO
from llava.conversation import (default_conversation, conv_templates,
                                   SeparatorStyle)
import hashlib
reqhost = 'http://127.0.0.1:8001'

# 示例模型列表和其他文本
models = ["TSS"]
title_markdown = "# Welcome to TiAmo"
tos_markdown = "## Terms of Service"
learn_more_markdown = "## Learn More"

block_css = """
#buttons button {
    min-width: min(120px,100%);
}
"""
no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."


def add_text(state, text, image, image_process_mode):
    if state is None:
        state = default_conversation.copy()

    if len(text) <= 0 and image is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 6

    text = text[:1536]  # Hard cut-off

    if image is not None:
        text = text[:1200]  # Hard cut-off for images
        if '<image>' not in text:
            text = text + '\n<image>'
        text = (text, image, image_process_mode)
    else:
        # No image provided, just use the text
        pass

    state.append_message(state.roles[0], text)
    state.skip_next = False
    print(state)
    return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 6


def delete_all_files_in_folder(folder_path):
    # 删除指定目录下的所有文件和子目录。
    if os.path.exists(folder_path):
        # 遍历目录中的所有文件和子目录
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                # 检查路径是否是文件或目录，并删除
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # 删除文件或符号链接
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # 删除目录及其所有内容
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print(f'The directory {folder_path} does not exist.')

class Args:
    def __init__(self):
        self.model_path = "./llava-v1.5-7b/"
        self.model_base = None
        self.device = "cuda"
        self.conv_mode = None
        self.temperature = 0.9
        self.max_new_tokens = 800
        self.load_8bit = False
        self.load_4bit = True
        self.debug = False

def crop_image(original_image, box):
    x1, y1, x2, y2 = map(int, box)
    return original_image.crop((x1, y1, x2, y2))

def run_yolo(original_image):
    cropped_img_path = os.path.join('yolo_pimages', f'yolo.jpg')
    original_image.save(cropped_img_path)
    yolo_model = YOLO('./yolov8n.pt')

    results = yolo_model(original_image)
    draw_image = original_image.copy()
    draw = ImageDraw.Draw(draw_image)

    class_names = {
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
        5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
        10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
        14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
        20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
        25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
        30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
        35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
        39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
        44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
        49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
        54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
        59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
        64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
        69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
        74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
        79: "toothbrush"
    }

    # 处理结果列表
    for result in results:
        boxes = result.boxes  # bbox输出的Boxes对象
        cls = boxes.cls
        conf = boxes.conf
        xyxyn = boxes.xyxyn

        # 将tensor转换为numpy数组并将数据移至CPU
        cls_numpy = cls.cpu().numpy()
        conf_numpy = conf.cpu().numpy()
        xyxyn_numpy = xyxyn.cpu().numpy()

        # 遍历每个检测
        for i in range(len(cls_numpy)):
            # 将归一化坐标转换为图像坐标
            box = xyxyn_numpy[i]
            xmin, ymin, xmax, ymax = box[0] * original_image.width, box[1] * original_image.height, box[
                2] * original_image.width, box[3] * original_image.height

            # 绘制边界框
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=2)

            # 使用映射从cls编号获取类别标签
            class_label = class_names.get(int(cls_numpy[i]), 'Unknown')

            # 准备带有类别和置信度的文本
            label = f'{class_label}: {conf_numpy[i]:.2f}'

            # 绘制类别和置信度文本
            draw.text((xmin, ymin), label, fill="white")

    # 创建一个目录来保存裁剪的图像，如果该目录不存在的话
    os.makedirs('yolo_pimages', exist_ok=True)

    # 再次处理结果列表
    for result in results:
        boxes = result.boxes  # bbox输出的Boxes对象
        cls = boxes.cls
        conf = boxes.conf
        xyxyn = boxes.xyxyn

        # 将tensor转换为numpy数组并将数据移至CPU
        cls_numpy = cls.cpu().numpy()
        conf_numpy = conf.cpu().numpy()
        xyxyn_numpy = xyxyn.cpu().numpy()

        # 遍历每个检测
        for i in range(len(cls_numpy)):
            # 将归一化坐标转换为图像坐标
            box = xyxyn_numpy[i]
            xmin, ymin, xmax, ymax = box[0] * original_image.width, box[1] * original_image.height, box[
                2] * original_image.width, box[3] * original_image.height

            # 裁剪图像并保存
            cropped_img = crop_image(original_image, (xmin, ymin, xmax, ymax))
            cropped_img_path = os.path.join('yolo_pimages', f'{class_label}_{i}_{conf_numpy[i]:.2f}.jpg')
            cropped_img.save(cropped_img_path)
    return


def generate_image(prompt, img_path):
    # 设置请求URL和请求体
    url = "https://stablediffusionapi.com/api/v3/img2img"
    payload = {
        "key": "EMrkPCZe1iNcqOJmkUXnmWZGEaQ1G2WjUlfq98GRjEloSWepM5xhKcajP0Ih",
        "prompt": prompt,
        "negative_prompt": None,
        "init_image": generate_image_url("E:\\SHENNET\\TSS\\TSS\\1.jpeg"),
        "width": "512",
        "height": "512",
        "samples": "1",
        "num_inference_steps": "30",
        "safety_checker": "no",
        "enhance_prompt": "yes",
        "guidance_scale": 7.5,
        "strength": 0.7,
        "seed": None,
        "webhook": None,
        "track_id": None
    }

    # 发送POST请求
    response = requests.post(url, json=payload)

    # 检查请求是否成功
    if response.status_code == 200:
        result = response.json()
        if result.get("status") == "success":
            # 获取生成的图像URL
            output_url = result["output"][0]
            print(f"Image generated successfully: {output_url}")

            # 请求生成的图像
            image_response = requests.get(output_url)
            if image_response.status_code == 200:
                img = Image.open(BytesIO(image_response.content))
                img.show()
                # 保存图像
                img.save("generated_image.png")
                return img
            else:
                print("Failed to retrieve the generated image.")
        else:
            print("Image generation failed:", result.get("tip"))
    else:
        print("Request failed with status code:", response.status_code)

def generate_image_from_last_response(chat_history):
    return generate_image("A detailed illustration of traditional Tibetan ethnic clothing on a white background,featuring intricate embroidery,vibrant colors,and ornate patterns. The outfit should include a long,flowing robe (chuba) with fur trimming around the collar and sleeves,a wide belt,and decorative jewelry such as turquoise and coral beads. The design should incorporate traditional Tibetan motifs,including geometric patterns and auspicious symbols. The image should capture the rich texture and detailed craftsmanship of the fabric,with no background distractions.,a \(phrase\),","https://picabstract-preview-ftn.weiyun.com/ftn_pic_abs_v3/a95a162eac5e3e8f91278d3760297b269f19928f7cf9313767ab42a9714b01629c9102df2ff76fa6c9ca59ce6c901ddc?pictype=scale&from=30113&version=3.3.3.3&fname=2.jpg&size=750")

def resllava(state, model_selector, temperature, top_p, max_new_tokens, request: gr.Request):
    prompt = state.get_prompt()
    print(prompt)
    folder_path = 'yolo_pimages'
    delete_all_files_in_folder(folder_path)
    images = state.get_images(return_pil=True)

    if images:
        for image in images:
            run_yolo(image)
        images_path = folder_path
    else:
        images_path = None

    pload = {
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2,
        "images_path": images_path,
    }
    try:
        response = requests.post(reqhost + "/llava", json=pload, timeout=100)
        res = response.json()
        if not res or res == 'None':
            res = "I'm sorry, but I couldn't generate a response. Could you please try rephrasing your question or providing more context?"
        print(f"Response: {res}")
        print(f"Response type: {type(res)}")
        state.append_message(state.roles[1], res)
        print(state)
        return (state, state.to_gradio_chatbot()) + (no_change_btn,) * 6
    except Exception as e:
        print(f"Error occurred: {e}")
        error_message = "An error occurred while processing your request. Please try again."
        state.append_message(state.roles[1], error_message)
        return (state, state.to_gradio_chatbot()) + (no_change_btn,) * 6

def clear_chat(state, text, image, image_process_mode):
    return [], None, None,None,""

def generate_image_url(image):
    # Gradio会自动处理图像文件并存储在本地临时目录中，我们只需获取图像的路径。
    if image:
        file_path = image
        # 构造本地链接
        file_url = f"http://127.0.0.1:7860/gradio_api/file={file_path}"
        return file_url
    else:
        return None, ""