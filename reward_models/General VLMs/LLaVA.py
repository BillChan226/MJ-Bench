import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

def LLaVA(image_url, prompt, model_id="llava-hf/llava-1.5-7b-hf"):
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(0)

    processor = AutoProcessor.from_pretrained(model_id)

    raw_image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    generated_text = processor.decode(output[0][2:], skip_special_tokens=True)
    return generated_text

# Example usage
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
prompt = "USER: <image>\nWhat are these?\nASSISTANT:"
generated_text = LLaVA(image_url, prompt)
print(generated_text)
