# from diffusers import StableDiffusionPipeline
# import torch

# model_id = "runwayml/stable-diffusion-v1-5"
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")

# prompt = "A woman is laying on her back with her legs in the air and her breasts exposed."
# image = pipe(prompt).images[0]  
    
# image.save("test.png")



import numpy as np
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
import cv2 as cv
def dummy(images, **kwargs): 
 return images, [False] * len(images)

if __name__ == '__main__':
 pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
 pipe.safety_checker = dummy
 pipe = pipe.to("cuda")

#  prompt = "A woman is laying on her back with her legs in the air and her breasts exposed."


 prompt = "A beautiful woman masturbating."

#  new_image = pipe(prompt, num_inference_steps=50).images[0]
 image = pipe(prompt).images[0] 
 image.save("test.png")

#  plt.save('test.png',image)