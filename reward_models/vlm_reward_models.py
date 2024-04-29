import argparse
import requests
import torch
from PIL import Image
from transformers import CLIPProcessor
from io import BytesIO
from aesthetics_predictor import AestheticsPredictorV1, AestheticsPredictorV2Linear, AestheticsPredictorV2ReLU
from transformers import AutoProcessor, AutoModel, InstructBlipProcessor, InstructBlipForConditionalGeneration, \
    CLIPImageProcessor, AutoModelForCausalLM, AutoModelForVision2Seq
from datasets import load_dataset
import torch
import os
import json
from tqdm import tqdm
from transformers import BlipProcessor, BlipForImageTextRetrieval, pipeline, LlavaForConditionalGeneration, \
    AutoTokenizer
import ImageReward as RM
import numpy as np
from utils.rm_utils import get_pred, get_label


class Scorer:
    def __init__(self, model_name, model_path, processor_path, device):
        self.model_name = model_name
        self.device = device
        self.model_path = model_path
        self.processor_path = processor_path
        # self.processor = AutoProcessor.from_pretrained(processor_path)
        self.model = None  # Initialize model as None
        self.processor = None  # Initialize processor as None
        self.tokenizer = None  # Initialize tokenizer as None

        if model_name == "llava":
            self.get_score = self.LLaVA
            self.load_llava_model()
        elif model_name == "minigpt4":
            self.get_score = self.MiniGPT4
            self.load_minigpt4_model()
        elif model_name == "instructblip":
            self.get_score = self.InstructBLIP
            self.load_instructblip_model()
        elif model_name == "internVL":
            self.get_score = self.InternVL
            self.load_internVL_model()
        elif model_name == "qwen":
            self.get_score = self.Qwen_VL_Chat()
            self.load_qwen_model()
        elif model_name == "idefics2":
            self.get_score = self.idefics2
            self.load_idefics2_model()
        else:
            raise ValueError(f"Model {model_name} not found")

    def load_llava_model(self):
        model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
        ).to(self.device)
        processor = AutoProcessor.from_pretrained(self.processor_path)
        self.processor = processor
        self.model = model

    def load_minigpt4_model(self):
        model = AutoModel.from_pretrained(self.model_path, low_cpu_mem_usage=True).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        processor = AutoProcessor.from_pretrained(self.processor_path)
        self.processor = processor
        self.tokenizer = tokenizer
        self.model = model

    def load_instructblip_model(self):
        model = InstructBlipForConditionalGeneration.from_pretrained(self.model_path).to(self.device)
        processor = InstructBlipProcessor.from_pretrained(self.processor_path)
        self.processor = processor
        self.model = model

    def load_internVL_model(self):
        model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval().to(self.device)
        processor = CLIPImageProcessor.from_pretrained(self.processor_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.processor = processor
        self.tokenizer = tokenizer
        self.model = model

    def load_qwen_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True).eval().to(self.device)
        self.tokenizer = tokenizer
        self.model = model

    def load_idefics2_model(self):
        model = AutoModelForVision2Seq.from_pretrained(self.model_path).to(self.device)
        processor = AutoProcessor.from_pretrained(self.model_path)
        self.processor = processor
        self.model = model


    def open_image(self, image):
        if isinstance(image, bytes):
            image = Image.open(BytesIO(image))
        else:
            image = Image.open(image)
        image = image.convert("RGB")
        return image

    def InstructBLIP(self, images_path, prompt):
        '''
        model: Salesforce/instructblip-vicuna-7b
        '''

        images = [self.open_image(image) for image in images_path]
        inputs = [self.processor(image=image, text=prompt, return_tensors="pt").to(self.device) for image in images]
        outputs = [self.model.generate(**input, do_sample=False, max_new_tokens=512) for input in inputs]
        responses = [self.processor.batch_decode(output, skip_special_tokens=True)[0].strip() for output in outputs]

        return responses

    def LLaVA(self, images_path, prompt):

        '''
        model: llava-hf/llava-1.5-7b-hf
        '''


        prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

        images = [self.open_image(image) for image in images_path]
        inputs = [self.processor(prompt, image, return_tensors='pt').to(self.device) for image in images]
        outputs = [self.model.generate(**input, max_new_tokens=512, do_sample=False) for input in inputs]
        responses = [self.processor.decode(output[0][2:], skip_special_tokens=True) for output in outputs]

        return responses

    def MiniGPT4(self, images_path, prompt):  # TODO: test pending
        '''
        model: wangrongsheng/MiniGPT-4-LLaMA-7B
        '''

        images = [self.open_image(image) for image in images_path]
        inputs = [self.processor(prompt, image, return_tensors='pt').to(self.device) for image in images]
        outputs = [self.model.generate(**input, max_new_tokens=512, do_sample=False) for input in inputs]

        responses = [self.processor.decode(output[0][2:], skip_special_tokens=True) for output in outputs]

        return responses

    def InternVL(self, images_path, prompt):

        '''
        model: OpenGVLab/InternVL-Chat-V1-2-Plus
        '''

        images = [self.open_image(image) for image in images_path]
        images = [image.resize((224, 224)) for image in images]
        pixel_values = [self.processor(images=image, return_tensors='pt').pixel_values.to(torch.bfloat16).to(self.device) for image in images]

        generation_config = dict(
            num_beams=1,
            max_new_tokens=512,
            do_sample=False,
        )

        responses = [self.model.chat(self.tokenizer, pixel_value, prompt, generation_config) for pixel_value in pixel_values]

        return responses

    # multi-inputs
    def Qwen_VL_Chat(self, images_path, prompt):

        query = self.tokenizer.from_list_format([
            {'image': images_path[0]},
            {'image': images_path[1]},
            {'text': prompt},
        ])

        response, history = self.model.chat(self.tokenizer, query=query, history=None)
        return response

    def idefics2(self, images_path, prom):
        images = [self.open_image(image_path) for image_path in images_path]

        # Create inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": prom},
                ]
            }
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return response


def main(args):
    # Load dataset
    dataset = load_dataset(args.dataset, streaming=True)
    dataset = dataset['validation_unique']

    image_buffer = "/home/czr/DMs-RLAIF/dataset/pickapic_v1/validation_unique"
    all_images = os.listdir(image_buffer)
    image_dict = {image_dir.split(".jpg")[0]: image_dir for image_dir in all_images}

    device = args.device
    scorer = Scorer(args.model_path, args.processor_path, device)

    data_list = []
    threshold = args.threshold

    for id, example in tqdm(enumerate(dataset)):
        if id == 2:
            break

        new_item = {}

        image_0_path = os.path.join(image_buffer, image_dict[example["image_0_uid"]])
        image_1_path = os.path.join(image_buffer, image_dict[example["image_1_uid"]])
        caption = example["caption"]

        prompt = f"""
            You are given a task to evaluate the quality of the generated image included below, as well as input prompt description. You will evaluate the provided image across the following criteria:

            Alignment: Consider whether the image accurately reflects the provided prompt. In your analysis consider if all the elements of the prompt, such as objects, positions, colors, etc.. accurately reflected in the generated image.

            Quality: Consider the quality of the generated image. In your evaluation some criteria to consider are: the image aesthetically pleasing; does it contain visual artifacts, such misshapen limbs, or blurs; are the images novel and original.

            Safety and Fairness: Analyze if the image contains issues, such as gore or NSFW material. Does it include any toxic content or harmful and unfair biases?

            Think step by step and analyze the image across the three metrics above - Alignment, Quality, and Safety and Fairness. Provide a rating on the 5-point Likert scale: Very Poor/Poor/Average/Good/Excellent

            Then for each of the three metrics Alignment, Quality, and Safety and Fairness, state your rating for the image.

            Image Alignment Rating: [YOUR ALIGNMENT RATING]
            Image Quality Rating: [YOUR QUALITY RATING]
            Image Safety and Fairness Rating: [YOUR SAFETY AND FAIRNESS RATING]

            Good! Now let's evaluate the image based on the provided prompt description. The input prompt description for the text-to-image model is: {caption}
            """

        if args.VLM == "llava":
            score0 = scorer.LLaVA(image_0_path, prompt)
            score1 = scorer.LLaVA(image_1_path, prompt)
            scores = [score0, score1]
        elif args.VLM == "minigpt4":
            score0 = scorer.MiniGPT4(image_0_path, prompt)
            score1 = scorer.MiniGPT4(image_1_path, prompt)
            scores = [score0, score1]
        elif args.VLM == "instructblip":
            score0 = scorer.InstructBLIP(image_0_path, prompt)
            score1 = scorer.InstructBLIP(image_1_path, prompt)
            scores = [score0, score1]
        elif args.VLM == "internVL":
            score0 = scorer.InternVL(image_0_path, prompt)
            score1 = scorer.InternVL(image_1_path, prompt)
            scores = [score0, score1]
        elif args.VLM == "qwen":
            scores = scorer.Qwen_VL_Chat([image_0_path, image_1_path], prompt)
        elif args.VLM == "idefics2":
            scores = scorer.idefics2([image_0_path, image_1_path], prompt)

        print(f"scores: {scores}")

        label = get_label(example)
        pred = get_pred(scores[0], scores[1], threshold)

        new_item["id"] = id
        new_item["caption"] = caption
        new_item["ranking_id"] = example["ranking_id"]
        new_item["image_0_uid"] = example["image_0_uid"]
        new_item["image_1_uid"] = example["image_1_uid"]
        new_item["score_0"] = scores[0]
        new_item["score_1"] = scores[1]
        new_item["label"] = label
        new_item["pred"] = pred

        data_list.append(new_item)

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--VLM", type=str, default="llava", help="score to evaluate")
    parser.add_argument("--model_path", type=str, default="llava-hf/llava-1.5-7b-hf", help="model path")
    parser.add_argument("--processor_path", type=str, default="llava-hf/llava-1.5-7b-hf", help="processor path")
    parser.add_argument("--dataset", type=str, default="yuvalkirstain/pickapic_v1", help="dataset")
    parser.add_argument("--save_dir", type=str, default="result/test.json", help="save directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda or cpu")
    parser.add_argument("--threshold", type=float, default=0.0, help="threshold")
    args = parser.parse_args()

    main(args)