import argparse
import requests
import torch
from PIL import Image
from transformers import CLIPProcessor
from io import BytesIO
from aesthetics_predictor import AestheticsPredictorV1, AestheticsPredictorV2Linear, AestheticsPredictorV2ReLU
from transformers import AutoProcessor, AutoModel, InstructBlipProcessor, InstructBlipForConditionalGeneration, CLIPImageProcessor, AutoModelForCausalLM, AutoModelForVision2Seq
from datasets import load_dataset
import torch
import os
import json
from tqdm import tqdm
from transformers import BlipProcessor, BlipForImageTextRetrieval, pipeline, LlavaForConditionalGeneration, AutoTokenizer
import hpsv2
import ImageReward as RM
import numpy as np


def get_pred(prob_0, prob_1, threshold):
    if abs(prob_1 - prob_0) <= threshold:
        pred = "tie"
    elif prob_0 > prob_1:
        pred = "0"
    else:
        pred = "1"
    return pred


def get_label(example):
    if example["label_0"] == 0.5:
        label = "tie"
    elif example["label_0"] == 1:
        label = "0"
    else:
        label = "1"
    return label


class VLM_scorer:
    def __init__(self, model_path, processor_path, device):
        self.device = device
        self.model_path = model_path
        self.processor_path = processor_path
        self.processor = AutoProcessor.from_pretrained(processor_path)
        self.model = AutoModel.from_pretrained(model_path).eval().to(device)

    def open_image(self, image):
        if isinstance(image, bytes):
            image = Image.open(BytesIO(image))
        else:
            image = Image.open(image)
        image = image.convert("RGB")
        return image

    def InstructBLIP(self, image_path, prompt):
        '''
        model: Salesforce/instructblip-vicuna-7b
        '''
        model = InstructBlipForConditionalGeneration.from_pretrained(self.model_path).to(self.device)
        processor = InstructBlipProcessor.from_pretrained(self.processor_path)

        image = self.open_image(image_path)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=512,
            # min_length=1,
            # top_p=0.9,
            # repetition_penalty=1.5,
            # length_penalty=1.0,
            # temperature=1,
            # num_beams=5,
        )
        response = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return response

    def LLaVA(self, image_path, prompt):

        '''
        model: llava-hf/llava-1.5-7b-hf
        '''

        model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path,
            # torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(self.device)

        processor = AutoProcessor.from_pretrained(self.processor_path)

        image = self.open_image(image_path)
        inputs = processor(prompt, image, return_tensors='pt').to(self.device, torch.float16)

        output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        response = processor.decode(output[0][2:], skip_special_tokens=True)
        return response

    def MiniGPT4(self, image_path, prompt):
        '''
        model: wangrongsheng/MiniGPT-4-LLaMA-7B
        '''
        model = AutoModel.from_pretrained(self.model_path, low_cpu_mem_usage=True).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        processor = AutoProcessor.from_pretrained(self.processor_path)

        image = self.open_image(image_path)
        inputs = processor(prompt, image, return_tensors='pt').to(self.device)

        output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        response = processor.decode(output[0][2:], skip_special_tokens=True)

        return response


    def InternVL(self, image_path, prompt):

        '''
        model: OpenGVLab/InternVL-Chat-V1-2-Plus
        '''

        model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval().to(self.device)

        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        image = self.open_image(image_path)
        image = image.resize((448, 448))
        image_processor = CLIPImageProcessor.from_pretrained(self.processor_path)

        pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(torch.bfloat16).to(self.device)

        generation_config = dict(
            num_beams=1,
            max_new_tokens=512,
            do_sample=False,
        )

        response = model.chat(tokenizer, pixel_values, prompt, generation_config)
        return response


    def Qwen_VL_Chat(self, images_path, prompt):

        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True).eval().to(self.device)

        images = [self.open_image(image_path) for image_path in images_path]

        query = tokenizer.from_list_format([
            {'image': images},
            {'text': prompt},
        ])

        response, history = model.chat(tokenizer, query=query, history=None)
        return response

    def idefics2(self, images_path, prom):
        images = [self.open_image(image_path) for image_path in images_path]

        processor = AutoProcessor.from_pretrained(self.model_path)
        model = AutoModelForVision2Seq.from_pretrained(self.model_path).to(self.device)

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
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)

        return response


def main(args):
    # Load dataset
    dataset = load_dataset(args.dataset, streaming=True)
    dataset = dataset['validation_unique']

    image_buffer = "/home/czr/DMs-RLAIF/dataset/pickapic_v1/validation_unique"
    all_images = os.listdir(image_buffer)
    image_dict = {image_dir.split(".jpg")[0]: image_dir for image_dir in all_images}

    device = args.device
    scorer = VLM_scorer(args.model_path, args.processor_path, device)

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
