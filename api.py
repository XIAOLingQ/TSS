import os
import re
from transformers import TextStreamer
import torch
from PIL import Image, ImageDraw
import requests
from fastapi import FastAPI, Request, BackgroundTasks
import uvicorn
from io import BytesIO

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

app = FastAPI()
def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def load_images(image_files):
    images = [load_image(image_file) for image_file in image_files]
    return images

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


def load_models(args):
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor,context_len= load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    return tokenizer, model, image_processor,context_len

class Modelmine:
    def __init__(self):
        tokenizer, model, image_processor, context_len = load_models(Args())
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len



    def response(self, user_prompt, image_path):
        if image_path and os.path.exists(image_path):
            image_files = [os.path.join(image_path, file) for file in os.listdir(image_path) if
                           file.endswith(('jpg', 'jpeg', 'png'))]
        else:
            image_files = []

        print(f"Image files: {image_files}")

        if not image_files:
            # No images found, process text-only input
            input_ids = self.tokenizer(user_prompt, return_tensors='pt').input_ids.to(self.model.device)
            image_tensor = None
            image_sizes = None
        else:
            # Process images
            images = load_images(image_files)
            print(f"Loaded images: {images}")
            image_sizes = [image.size for image in images]
            image_tensor = process_images(images, self.image_processor, self.model.config)

            if image_tensor is not None:
                if isinstance(image_tensor, list):
                    image_tensor = [img.to(self.model.device, dtype=torch.float16) for img in image_tensor]
                else:
                    image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

            input_ids = tokenizer_image_token(user_prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                                              return_tensors='pt').unsqueeze(0).to(self.model.device)

        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        args = Args()

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True)

        outputs = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return outputs


@app.post("/llava")
async def llava(request: Request):
    try:
        params = await request.json()
        print(params)

        image_path = params.get('images_path')
        prompt = params.get('prompt', '')

        pattern = re.compile(r"USER:.*", re.DOTALL)
        match = pattern.search(prompt)

        user_prompt = match.group() if match else prompt  # Use the whole prompt if no USER: is found

        print("Extracted USER part:")
        print(user_prompt)
        print(f"Image path: {image_path}")

        response = model.response(user_prompt, image_path)
        return response
    except Exception as e:
        print(f"Error in /llava endpoint: {e}")
        return {"error": str(e)}

model = Modelmine()
uvicorn.run(app, host='127.0.0.1', port=8001)