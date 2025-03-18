pip install google-generativeai ipywidgets pillow

import google.generativeai as genai
from diffusers import StableDiffusionPipeline
from huggingface_hub import login
from PIL import Image
import torch
import os 
import io
import ipywidgets as widgets
from IPython.display import display, clear_output, Image as DisplayImage
import os
from transformers import logging as transformers_logging

genai.configure(api_key="AIzaSyDSmcKB_A5N8c74AQKNXVJ03-YHfyTzF2A")
login(token='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')

nltk_logger = logging.getLogger('nltk')
nltk_logger.setLevel(logging.ERROR)

transformers_logging.set_verbosity_error()

def generate_text(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error while generating text: {str(e)}"

prompt = "Explain what makes a red car fast."
response = generate_text(prompt)
print(f"Generated Text: {response}")

def generate_image(prompt):
    try:
        print("\nGenerating an image...")
        model_id = "runwayml/stable-diffusion-v1-5"  
        model_path = "./models/stable-diffusion-v1-5"

        if not os.path.exists(model_path):
            print("Model not found locally. Downloading...")
            pipe = StableDiffusionPipeline.from_pretrained(model_id)
            pipe.save_pretrained(model_path)
        else:
            print("Loading model from local storage...")
            pipe = StableDiffusionPipeline.from_pretrained(model_path)

        pipe = pipe.to("cpu")

        image = pipe(prompt).images[0]
        image_path = "./models/generated_image.png" 
        image.save(image_path)

        return image_path
    except Exception as e:
        return f"Error while generating image: {str(e)}"  

prompt = "A red sports car racing through a city."
image_path = generate_image(prompt)
if isinstance(image_path, str) and "❌" not in image_path:
    print(f"Generated Image: {image_path}")
    display(Image.open(image_path))  
else:
    print(image_path)

def analyze_image(image_bytes):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([image_bytes])
        return response.text
    except Exception as e:
        return f"Error while analyzing image: {str(e)}"

output = widgets.Output()

upload_button = widgets.FileUpload(accept='image/*', multiple=False)
submit_button = widgets.Button(description="Submit")

output = widgets.Output()

def on_submit(button):
    with output:
        clear_output()

        if not upload_button.value:
            print("Please upload an image first.")
            return

        if isinstance(upload_button.value, tuple):
            uploaded_file = upload_button.value[0]
        else:
            uploaded_file = next(iter(upload_button.value.values()))

        image_bytes = uploaded_file['content']  

        try:
            os.makedirs("./models", exist_ok=True)
            image_path = "./models/uploaded_image.png"
            with open(image_path, "wb") as f:
                f.write(image_bytes)

            image = Image.open(io.BytesIO(image_bytes))

            print("Analyzing uploaded image")
            analysis_result = analyze_image(image)
            print(f"Image Analysis Result:\n{analysis_result}")
        except Exception as e:
            print(f"Error while processing image: {e}")

submit_button.on_click(on_submit)
display(widgets.VBox([
    widgets.Label("Upload an image to analyze it:"),
    upload_button,
    submit_button,
    output
]))

text_input = widgets.Text(
    placeholder="Enter your text prompt here...",
    description="Text Input:",
    disabled=False
)

upload_button = widgets.FileUpload(
    accept='image/*', 
    multiple=False
)

submit_button = widgets.Button(description="Submit")

output = widgets.Output()

def handle_combined_query(prompt, uploaded_file=None):
    with output:
        clear_output()
        parts = prompt.lower().split("and")
        image_part = None
        text_part = None
        for part in parts:
            if any(keyword in part for keyword in ["generate", "image", "show"]):
                image_part = part.strip()
            else:
                text_part = part.strip()
        if image_part:
            print("Processing image request")
            image_prompt = image_part.replace("generate image of", "").replace("show me", "").strip()
            image_bytes = generate_image(image_prompt)
            if isinstance(image_bytes, bytes):
                print(f"Generated Image:")
                display(DisplayImage(data=image_bytes))
            else:
                print(image_bytes)
        if uploaded_file:
            print("Analyzing uploaded image")
            try:
                image_bytes = bytes(uploaded_file['content'])
                image = Image.open(BytesIO(image_bytes))
                analysis_result = analyze_image(image)
                print(f"Image Analysis Result:\n{analysis_result}")
            except Exception as e:
                print(f"Error while analyzing image: {e}")
        if text_part:
            print("\nProcessing text request")
            subject = image_prompt if image_part else "the topic"
            refined_text_prompt = text_part.replace("it", subject)
            text_response = generate_text(refined_text_prompt)
            print(f"Text Response:\n{text_response}")

def on_submit(button):
    with output:
        #clear_output()
        prompt = text_input.value.strip()
        uploaded_file = upload_button.value[0] if upload_button.value else None
        if prompt or uploaded_file:
            handle_combined_query(prompt, uploaded_file)
        else:
            print("Please enter a valid prompt or upload an image.")

submit_button.on_click(on_submit)

display(widgets.VBox([
    widgets.Label("Multi-Modal Chatbot"),
    text_input,
    upload_button,
    submit_button,
    output
]))

#Results are saved in Task2/models folder 



