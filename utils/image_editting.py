import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
from functools import partial
import cv2
import requests
import io
from io import BytesIO
from PIL import Image
import numpy as np
from pathlib import Path
import sys
sys.path.append("./")
sys.path.append("./utils/GroundingDINO/")
import warnings
warnings.filterwarnings("ignore")
import torch
from torchvision.ops import box_convert

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import annotate, load_image, predict
import groundingdino.datasets.transforms as T

from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionInpaintPipeline
import supervision as sv



def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()

    return model


def download_image(url, image_file_path):
    r = requests.get(url, timeout=4.0)
    if r.status_code != requests.codes.ok:
        assert False, 'Status code error: {}.'.format(r.status_code)

    with Image.open(io.BytesIO(r.content)) as im:
        im.save(image_file_path)

    print('Image downloaded from url: {} and saved to: {}.'.format(url, image_file_path))

def generate_masks_with_grounding(image_source, boxes):
    h, w, _ = image_source.shape
    boxes_unnorm = boxes * torch.Tensor([w, h, w, h])
    boxes_xyxy = box_convert(boxes=boxes_unnorm, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    mask = np.zeros_like(image_source)
    for box in boxes_xyxy:
        x0, y0, x1, y1 = box
        mask[int(y0):int(y1), int(x0):int(x1), :] = 255
    return mask


# Use this command for evaluate the Grounding DINO model
# Or you can download the model by yourself
# ckpt_repo_id = "ShilongLiu/GroundingDINO"
# ckpt_filenmae = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
# ckpt_config_filename = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swint_ogc.pth"
ckpt_config_filename = "GroundingDINO_SwinT_OGC.cfg.py"

    
model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)

pipe = pipe.to("cuda")

image_url = 'https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/cats.png'
local_image_path = 'cats.png'


# download image
# download_image(image_url, local_image_path)

TEXT_PROMPT = "bus"
BOX_TRESHOLD = 0.1
TEXT_TRESHOLD = 0.1

# local_image_path = "/home/czr/object-bias/benchmark_dataset/image_to_text/high_cooc/object_existence/normal/person_surfboard/COCO_val2014_000000340658.jpg"
local_image_path = "co-occurrence/dataset/image_to_text/low_cooc/object_existence/normal/bus_snowboard/COCO_val2014_000000170629.jpg"

image_source, image = load_image(local_image_path)

boxes, logits, phrases = predict(
    model=model, 
    image=image, 
    caption=TEXT_PROMPT, 
    box_threshold=BOX_TRESHOLD, 
    text_threshold=TEXT_TRESHOLD
)

print("boxes", boxes)
# move the box down a little bit


annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
annotated_frame = annotated_frame[...,::-1] # BGR to RGB

# image_source: np.ndarray
# annotated_frame: np.ndarray


image_mask = generate_masks_with_grounding(image_source, boxes)

# image source
image_source = Image.fromarray(image_source)
image_source.save("co-occurrence/examples/image-to-text/low_occurrence/object existence/image_source.jpg")

# annotated image
annotated_image = Image.fromarray(annotated_frame)
annotated_image.save("co-occurrence/examples/image-to-text/low_occurrence/object existence/annotated_image.jpg")

# image mask
image_mask = Image.fromarray(image_mask)
image_mask.save("co-occurrence/examples/image-to-text/low_occurrence/object existence/masked_image.jpg")

image_source_for_inpaint = image_source.resize((512, 512))
image_mask_for_inpaint = image_mask.resize((512, 512))

# prompt = "person sleeping under the chair"
prompt = "snowboard"
#image and mask_image should be PIL images.
#The mask structure is white for inpainting and black for keeping as is
image_inpainting = pipe(prompt=prompt, image=image_source_for_inpaint, mask_image=image_mask_for_inpaint).images[0]

image_inpainting = image_inpainting.resize((image_source.size[0], image_source.size[1]))

image_inpainting.save("co-occurrence/examples/image-to-text/low_occurrence/object existence/inpainting_image.jpg")



