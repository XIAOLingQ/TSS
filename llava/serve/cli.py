import argparse
import os
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
import time
def crop_image(original_image, box):
    x1, y1, x2, y2 = map(int, box)
    return original_image.crop((x1, y1, x2, y2))
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
# 文件夹路径
folder_path = 'yolo_pimages'

#处理多张图片
def load_images(image_files):
    images = [load_image(image_file) for image_file in image_files]
    return images


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def mine(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    # 加载预训练的 YOLOv8n 模型
    yolo_model = YOLO('E:\SHENNET\TSS\LLaVA\yolov8n.pt')
    # 使用 PIL 加载一张图片
    original_image = Image.open(args.image_file)
    print(args.image_file)
    # 在图片上运行推理
    results = yolo_model(original_image)
    # 创建原始图片的副本以绘制边界框
    draw_image = original_image.copy()
    draw = ImageDraw.Draw(draw_image)

    # 类别映射
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

    # 获取文件夹中的所有图片文件
    image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
                   file.endswith(('jpg', 'jpeg', 'png'))]

    # 加载所有图片
    images = load_images(image_files)
    print(images)

    # 获取每个图片的尺寸
    image_sizes = [image.size for image in images]
    for image in images:
        image.show()
        time.sleep(5)
    # # 加载图像文件列表
    # image_files = ['R-C.jpg', 'R-C-1.jpg']  # 替换为实际的图像路径
    # images = load_images(image_files)
    # image_sizes = [image.size for image in images]
    # 图像预处理
    image_tensor = process_images(images, image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if images is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            images = None

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(
            model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True)

        outputs = tokenizer.decode(output_ids[0]).strip()
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

class Args:
    def __init__(self):
        self.model_path = "E:\\SHENNET\\TSS\\LLaVA\\llava-v1.5-7b"
        self.model_base = None
        self.image_file = "E:\\SHENNET\\TSS\\LLaVA\\bus.jpg"
        self.device = "cuda"
        self.conv_mode = None
        self.temperature = 0.2
        self.max_new_tokens = 512
        self.load_8bit = False
        self.load_4bit = True
        self.debug = False

if __name__ == "__main__":
    args = Args()
    mine(args)
