# chatbot.py
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Load the text generation model (GPT-2)
def load_text_model():
    try:
        return pipeline("text-generation", model="gpt2", device=0 if torch.cuda.is_available() else -1)
    except Exception as e:
        raise Exception(f"Error loading text generation model: {e}")

# Load the image generation model (Stable Diffusion)
def load_image_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
        return model
    except Exception as e:
        raise Exception(f"Error loading image generation model: {e}")

# Function to generate text
def generate_text(prompt, text_generator):
    try:
        response = text_generator(prompt, max_length=100, num_return_sequences=1, truncation=True)
        return response[0]['generated_text']
    except Exception as e:
        return f"Error generating text: {e}"

# Function to generate an image
def generate_image(prompt, image_generator):
    try:
        image = image_generator(prompt).images[0]  # Generate the image
        return image
    except Exception as e:
        return f"Error generating image: {e}"