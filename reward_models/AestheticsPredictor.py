import requests
import torch
from PIL import Image
from transformers import CLIPProcessor

from aesthetics_predictor import AestheticsPredictorV1, AestheticsPredictorV2Linear, AestheticsPredictorV2ReLU


def get_aesthetics_score(image_path, model_id, device):
    # Load the aesthetics predictor
    predictor = AestheticsPredictorV2Linear.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)

    image = Image.open(image_path)

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")

    # Move to GPU if available
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"

    predictor = predictor.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Perform inference
    with torch.no_grad():
        outputs = predictor(**inputs)
    prediction = outputs.logits.item()

    return prediction


# TODO: Test
# shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE
model_id = "shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

image_path = "good.png"
aesthetics_score = get_aesthetics_score(image_path, model_id, device)
print("Aesthetics score:", aesthetics_score)


image_path = "bad.png"
aesthetics_score = get_aesthetics_score(image_path, model_id, device)
print("Aesthetics score:", aesthetics_score)
