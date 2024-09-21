import ast
import os
import shutil
Frist_chat = True
import gradio as gr
import requests
from PIL import Image, ImageDraw
from ultralytics import YOLO
from llava.conversation import (default_conversation, conv_templates,
                                   SeparatorStyle)
import hashlib
reqhost = 'http://127.0.0.1:8001'

# 示例模型列表和其他文本
models = ["model1", "model2"]
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
    """
    删除指定目录下的所有文件和子目录。

    Args:
        folder_path (str): 目标目录路径。
    """
    # 检查目录是否存在
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
        self.model_path = "E:\\SHENNET\\TSS\\LLaVA\\llava-v1.5-7b"
        self.model_base = None
        self.device = "cuda"
        self.conv_mode = None
        self.temperature = 0.2
        self.max_new_tokens = 512
        self.load_8bit = False
        self.load_4bit = True
        self.debug = False

def crop_image(original_image, box):
    x1, y1, x2, y2 = map(int, box)
    return original_image.crop((x1, y1, x2, y2))

def run_yolo(original_image):
    cropped_img_path = os.path.join('yolo_pimages', f'yolo.jpg')
    original_image.save(cropped_img_path)
    yolo_model = YOLO('E:\\SHENNET\\TSS\\LLaVA\\yolov8n.pt')
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

def generate_image(prompt):
    # 这是一个示例函数，实际的Stable Diffusion生成代码可能会有所不同
    # url = "https://via.placeholder.com/512?text=Generated+Image"  # 占位符图像
    # response = requests.get(url)
    img = Image.open("OIP-C.jpg")
    return img

def generate_image_from_last_response(chat_history):
    if len(chat_history) > 0:
        last_response = chat_history[-1][1]  # 获取最后一次回答
        return generate_image(last_response)
    return None


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



def build_demo(embed_mode, cur_dir=None, concurrency_count=10):

    textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)
    with gr.Blocks(title="TiAmo", theme=gr.themes.Default(), css=block_css) as demo:
        state = gr.State()

        if not embed_mode:
            gr.Markdown("# TiAmo Chatbot")

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row(elem_id="model_selector_row"):
                    model_selector = gr.Dropdown(
                        choices=models,
                        value=models[0] if len(models) > 0 else "",
                        interactive=True,
                        show_label=False,
                        container=False
                    )

                imagebox = gr.Image(type="pil")
                image_process_mode = gr.Radio(
                    ["Crop", "Resize", "Pad", "Default"],
                    value="Default",
                    label="Preprocess for non-square image",
                    visible=False
                )

                if cur_dir is None:
                    cur_dir = os.path.dirname(os.path.abspath(__file__))

                gr.Examples(examples=[
                    [f"1.jpeg", "How about this Tibetan costume."],
                    [f"2.jpg", "Describe the Miao costumes in the picture."]
                ], inputs=[imagebox, textbox])

                with gr.Accordion("Parameters", open=False) as parameter_row:
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True,
                                            label="Temperature")
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P")
                    max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True,
                                                  label="Max output tokens")

            with gr.Column(scale=8):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="TiAmo Chatbot",
                    height=650,
                    layout="panel",
                )
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(value="Send", variant="primary")

                with gr.Row(elem_id="buttons") as button_row:
                    upvote_btn = gr.Button(value="  Upvote", interactive=False)
                    downvote_btn = gr.Button(value="  Downvote", interactive=False)
                    flag_btn = gr.Button(value="  Flag", interactive=False)
                    regenerate_btn = gr.Button(value="  Regenerate", interactive=False)
                    clear_btn = gr.Button(value="  Clear", interactive=False)
                    generate_img_btn = gr.Button(value="Generate Image", variant="secondary")

        if not embed_mode:
            gr.Markdown("## Terms of Service")
            gr.Markdown("## Learn More")

        url_params = gr.JSON(visible=False)

        # Register listeners (移除涉及后端接口的监听器)
        btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn, generate_img_btn]

        clear_btn.click(
            lambda: (None, None, "", None) + (False,) * len(btn_list),
            None,
            [state, chatbot, textbox, imagebox] + btn_list,
            queue=False
        )

        submit_btn.click(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox] + btn_list
        ).then(
            resllava,
            [state, model_selector, temperature, top_p, max_output_tokens],
            [state, chatbot] + btn_list,
        )

        generate_img_btn.click(
            generate_image_from_last_response,
            [chatbot],
            [imagebox]
        )

    return demo

demo = build_demo(embed_mode=False)
demo.launch()

