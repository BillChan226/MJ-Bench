import os
import openai
import anthropic
import google.generativeai as genai
from google.generativeai.types import ContentType
from PIL import Image
from IPython.display import Markdown
import time
import cv2
import base64
import httpx
import requests

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')



class Scorer:
    def __init__(self, model_name, model_path, api_key, base_url):
        self.model_name = model_name
        self.model_path = model_path
        self.api_key = api_key
        self.base_url = base_url

        if "gpt" in model_name:
            self.get_score = self.gpt_score
            
        elif "gemini" in model_name:
            self.get_score = self.gemini_score
            
        elif "claude" in model_name:
            self.client = anthropic.Anthropic(
            api_key=self.api_key,
            )
            self.get_score = self.claude_score

        else:
            raise ValueError(f"Model {model_name} not found")
        



    def gpt_score(self, images, prompt):


        if len(images) == 1:
            image_dir = images[0]
            image_data = encode_image(image_dir)

            response = requests.post(
            # 'https://azure-openai-api.shenmishajing.workers.dev/v1/chat/completions',
            'https://api.openai.com/v1/chat/completions',
            headers={'Authorization': f'Bearer {self.api_key}'},
            json={'model': "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                        {
                        "type": "text",
                        "text": f"{prompt}"
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        },
                        }
                        ],
                    }
                ], 
                'max_tokens': 256, 'n': 1, 'temperature': 1} 
            )
            # print("response", response)
            data = response.json()
            gpt4_output = data['choices'][0]['message']['content']
        



        if len(images) > 1:
        
            image_0_dir, image_1_dir = images
            image_data_0 = encode_image(image_0_dir)
            image_data_1 = encode_image(image_1_dir)

            response = requests.post(
            # 'https://azure-openai-api.shenmishajing.workers.dev/v1/chat/completions',
            'https://api.openai.com/v1/chat/completions',
            headers={'Authorization': f'Bearer {self.api_key}'},
            json={'model': "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                        {
                        "type": "text",
                        "text": f"{prompt}"
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data_0}"
                        },
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data_1}"
                        },
                        }
                        ],
                    }
                ], 
                'max_tokens': 256, 'n': 1, 'temperature': 1} 
            )
            # print("response", response)
            data = response.json()
            gpt4_output = data['choices'][0]['message']['content']



    def claude_score(self, images, prompt):
        
        media_type = "image/jpeg"
        
        if len(images) == 1:
            image_dir = images[0]
            image_data = encode_image(image_dir)
            

            message = self.client.messages.create(
            model=self.model_path
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
        )
        Markdown(message.content[0].text)

        claude_output = message.content[1].text

        input()

