import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor, AutoTokenizer

def InternVL(model_path, image_path, question, device="cuda"):
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    image = Image.open(image_path).convert('RGB')
    image = image.resize((448, 448))
    image_processor = CLIPImageProcessor.from_pretrained(model_path)

    pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
    pixel_values = pixel_values.to(torch.bfloat16).to(device)

    generation_config = dict(
        num_beams=1,
        max_new_tokens=512,
        do_sample=False,
    )

    response = model.chat(tokenizer, pixel_values, question, generation_config)
    return response

# Example usage
model_path = "OpenGVLab/InternVL-Chat-V1-2-Plus"
image_path = './examples/image2.jpg'
question = "请详细描述图片"
response = InternVL(model_path, image_path, question)
print(question, response)


