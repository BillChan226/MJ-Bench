from transformers import AutoModelForCausalLM, AutoTokenizer

def Qwen_VL_Chat(image_url, text_query, model_name="Qwen/Qwen-VL-Chat"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", trust_remote_code=True).eval()

    query = tokenizer.from_list_format([
        {'image': image_url},
        {'text': text_query},
    ])

    response, history = model.chat(tokenizer, query=query, history=None)
    return response


# Example usage
image_url = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'
text_query = '这是什么'
response = Qwen_VL_Chat(image_url, text_query)
print(response)
