import streamlit as st
from PIL import Image
import google.generativeai as genai
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from huggingface_hub import login
import torch
import os
from io import BytesIO
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "your_google_api_key_here")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "your_huggingface_token_here")

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    login(token=HUGGINGFACE_TOKEN)
except Exception as e:
    logging.error(f"API configuration error: {e}")

logging.getLogger('nltk').setLevel(logging.ERROR)
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

MODEL_ID = "runwayml/stable-diffusion-v1-5"
MODEL_PATH = "./models/stable-diffusion-v1-5"

@st.cache_resource  
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            logging.info("Downloading Stable Diffusion model (first-time setup)...")
            model = DiffusionPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float32,
                use_safetensors=True
            )
            model.scheduler = DPMSolverMultistepScheduler.from_config(model.scheduler.config)  # Fix scheduler issue
            model.save_pretrained(MODEL_PATH)
        else:
            logging.info("Loading model from local storage...")
            model = DiffusionPipeline.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.float32,
                use_safetensors=True
            )
            model.scheduler = DPMSolverMultistepScheduler.from_config(model.scheduler.config)  # Fix scheduler issue
        return model.to("cpu") 
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None

pipe = load_model()

def generate_text(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text if response else "No response from model."
    except Exception as e:
        logging.error(f"Text generation error: {e}")
        return f"Error generating text: {str(e)}"

def generate_image(prompt):
    try:
        if not pipe:
            logging.error("Stable Diffusion model is not loaded.")
            return "Error: Model not loaded."
        
        logging.info(f"Generating image for prompt: {prompt}")
        image = pipe(prompt, num_inference_steps=50).images[0]  
        image_path = "./models/generated_image.png"
        os.makedirs("./models", exist_ok=True) 
        image.save(image_path)
        logging.info(f"Image saved at: {image_path}")
        return image_path
    except Exception as e:
        logging.error(f"Image generation error: {e}")
        return f"Error generating image: {str(e)}"

def handle_combined_query(prompt, uploaded_file=None):
    logging.info(f"Processing user request with prompt: {prompt}")

    image_keywords = ["generate", "image", "show", "create", "draw", "picture", "visualize"]
    text_keywords = ["what", "who", "explain", "describe", "tell me about", "meaning of", "define"]

    words = prompt.lower().split()

    contains_image_request = any(word in words for word in image_keywords)
    contains_text_request = any(word in words for word in text_keywords)

    image_description = None
    if contains_image_request:
        match = re.search(r"(?:generate|image|show|create|draw|picture)\s+(.+)", prompt, re.IGNORECASE)
        if match:
            image_description = match.group(1).strip()
            logging.info(f"Detected image description: {image_description}")
        else:
            logging.warning("Failed to extract image description from prompt.")

    text_query = prompt
    for word in image_keywords:
        text_query = text_query.replace(word, "").strip()

    if contains_image_request and image_description:
        logging.info("Detected image generation request.")
        image_path = generate_image(image_description)
        if os.path.exists(image_path):
            st.image(image_path, caption=f"Generated Image: {image_description}", use_column_width=True)
        else:
            st.error(f"Image generation failed: {image_path}")

    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            image_bytes = buffered.getvalue()
            analysis_result = analyze_image(image_bytes)
            st.write(f"🖼 **Image Analysis Result:**\n{analysis_result}")
        except Exception as e:
            logging.error(f"Error analyzing image: {e}")
            st.error(f"Error analyzing image: {e}")

    if contains_text_request:
        logging.info("Detected text generation request.")
        text_response = generate_text(text_query)
        st.write(f"📝 **Response:**\n{text_response}")

st.title("🤖 Multi-Modal Chatbot")

text_input = st.text_input("Enter your query:", placeholder="Ask me anything...")

uploaded_file = st.file_uploader("Upload an image for analysis", type=["png", "jpg", "jpeg"])

if st.button("Submit"):
    if text_input.strip() or uploaded_file:
        handle_combined_query(text_input.strip(), uploaded_file)
    else:
        st.warning("Enter a valid prompt or upload an image.")

st.subheader("🎨 Generate an Image")
image_prompt = st.text_input("Enter image prompt:", placeholder="Describe the image you want...")

if st.button("Generate Image"):
    if image_prompt.strip():
        image_path = generate_image(image_prompt)
        if os.path.exists(image_path):
            st.image(image_path, caption="Generated Image", use_column_width=True)
        else:
            st.error(image_path)
    else:
        st.warning("Enter a valid image generation prompt.")