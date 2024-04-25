from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests

def InstructBLIP(url, prompt, model_name="Salesforce/instructblip-vicuna-7b"):
    model = InstructBlipForConditionalGeneration.from_pretrained(model_name)
    processor = InstructBlipProcessor.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        do_sample=False,
        max_length=256,
        # min_length=1,
        # top_p=0.9,
        # repetition_penalty=1.5,
        # length_penalty=1.0,
        # temperature=1,
        # num_beams=5,
    )
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    return generated_text

# Example usage
url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
prompt = "What is unusual about this image?"
generated_text = InstructBLIP(url, prompt)
print(generated_text)
