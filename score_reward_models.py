import argparse
import requests
import torch
from PIL import Image
from transformers import CLIPProcessor
from io import BytesIO
from aesthetics_predictor import AestheticsPredictorV1, AestheticsPredictorV2Linear, AestheticsPredictorV2ReLU
from transformers import AutoProcessor, AutoModel, InstructBlipProcessor, InstructBlipForConditionalGeneration
from datasets import load_dataset
import torch
import os
import json
from tqdm import tqdm
from transformers import BlipProcessor, BlipForImageTextRetrieval, pipeline, LlavaForConditionalGeneration
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

class Scorer:
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

    def get_aesthetics_score(self, images_path, caption):
        images = [self.open_image(image) for image in images_path]

        predictor = AestheticsPredictorV2Linear.from_pretrained(self.model_path)
        processor = CLIPProcessor.from_pretrained(self.processor_path)
        inputs = processor(images=images, return_tensors="pt")

        predictor = predictor.to(self.device)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = predictor(**inputs)
        aesthetics_score = outputs.logits
        aesthetics_score = aesthetics_score.cpu().tolist()
        scores = [score[0] for score in aesthetics_score]

        return scores

    def get_clipscore(self, images_path, caption):
        images = [self.open_image(image) for image in images_path]

        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        text_inputs = self.processor(
            text=caption,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            image_embs = self.model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

            text_embs = self.model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            scores = (text_embs @ image_embs.T)[0]

        return scores.cpu().tolist()

    def get_blipscore(self, images_path, caption):
        images = [self.open_image(image) for image in images_path]

        processor = BlipProcessor.from_pretrained(self.processor_path)
        model = BlipForImageTextRetrieval.from_pretrained(self.model_path).eval().to(self.device)

        with torch.no_grad():
            inputs = processor(images, caption, return_tensors="pt").to(self.device)
            cosine_score = model(**inputs, use_itm_head=False)[0]
            cosine_score = cosine_score.cpu().tolist()
            scores = [score[0] for score in cosine_score]

        return scores

    def get_pickscore(self, images_path, caption):
        images = [self.open_image(image) for image in images_path]

        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        text_inputs = self.processor(
            text=caption,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            image_embs = self.model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

            text_embs = self.model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T)[0]

            probs = torch.softmax(scores, dim=-1)

        return probs.cpu().tolist()

    def ImageReward(self, images_path, caption):
        model = RM.load("ImageReward-v1.0").to(self.device)
        scores = model.score(caption, images_path)

        return scores



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

        if args.score == "clipscore":
            scores = scorer.get_clipscore([image_0_path, image_1_path], caption)
        elif args.score == "pickscore":
            scores = scorer.get_pickscore([image_0_path, image_1_path], caption)
        elif args.score == "blipscore":
            scores = scorer.get_blipscore([image_0_path, image_1_path], caption)
        elif args.score == "AestheticScore":
            scores = scorer.get_aesthetics_score([image_0_path, image_1_path], caption)
        elif args.score == "HPS_v2.1":
            scores = hpsv2.score([image_0_path, image_1_path], caption, hps_version="v2.1")
            scores = np.array(scores).tolist()
        elif args.score == "ImageReward":
            scores = scorer.ImageReward([image_0_path, image_1_path], caption)


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
    parser.add_argument("--score", type=str, default="HPS_v2.1", help="score to evaluate")
    parser.add_argument("--model_path", type=str, default="yuvalkirstain/PickScore_v1", help="model path")
    parser.add_argument("--processor_path", type=str, default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K", help="processor path")
    parser.add_argument("--dataset", type=str, default="yuvalkirstain/pickapic_v1", help="dataset")
    parser.add_argument("--save_dir", type=str, default="result/test.json", help="save directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda or cpu")
    parser.add_argument("--threshold", type=float, default=0.0, help="threshold")
    args = parser.parse_args()

    main(args)
