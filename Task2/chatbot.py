# import google.generativeai as palm
# import openai
# import requests
# from PIL import Image
# import io
# from Task1.Untitled import generate_summary

# import google.generativeai as genai

# genai.configure(api_key="AIzaSyBkBRgVm5sOp-HkmvQRPjc0ZPAAJdMSf3I")

# model = genai.GenerativeModel('gemini-1.5-pro-001')


# def generate_text(prompt):
#     try:
#         response = model.generate_content(prompt)
#         return response.text  # Extract and return response text
#     except Exception as e:
#         return f"Error: {e}"


# # Function to generate an image using DALL·E
# import openai
# def generate_image(prompt):
#     try:
#         response = openai.Image.create(
#             prompt=prompt,
#             n=1,
#             size="1024x1024"
#         )
#         print("API Response:", response)  # Debugging line
#         image_url = response["data"][0]["url"]
#         return image_url
#     except openai.error.OpenAIError as e:
#         print(f"OpenAI API error: {e}")
#     except Exception as e:
#         print(f"Unexpected error: {e}")
#     return None



# # Function to analyze an image (placeholder for Gemini AI or CLIP)
# def analyze_image(image_url):
#     # Placeholder for image analysis (e.g., object detection, scene understanding)
#     # You can integrate Gemini AI or CLIP here
#     return "This is a placeholder for image analysis."

# # Multi-modal chatbot
# def multimodal_chatbot():
#     print("Welcome to the Multi-Modal Chatbot! Type 'exit' to quit.")
#     while True:
#         user_input = input("You: ")

#         if user_input.lower() == "exit":
#             print("Chatbot: Goodbye!")
#             break

#         # Check if the user wants to generate an image
#         if "generate image" in user_input.lower():
#             prompt = user_input.replace("generate image", "").strip()
#             if prompt:
#                 print("Chatbot: Generating an image...")
#                 image_url = generate_image(prompt)
#                 print(f"Chatbot: Here's your image: {image_url}")
#             else:
#                 print("Chatbot: Please provide a description for the image.")

#         # Check if the user provides an image URL for analysis
#         elif user_input.startswith("http") and any(ext in user_input for ext in [".jpg", ".png", ".jpeg"]):
#             print("Chatbot: Analyzing the image...")
#             analysis_result = analyze_image(user_input)
#             print(f"Chatbot: {analysis_result}")

#         # Check if the user wants to summarize a long text
#         elif "summarize" in user_input.lower():
#             text_to_summarize = user_input.replace("summarize", "").strip()
#             if text_to_summarize:
#                 print("Chatbot: Summarizing the text...")
#                 summary = generate_summary(text_to_summarize, summary_length=3)  # Use the summarization tool
#                 print(f"Chatbot: Here's the summary:\n{summary}")
#             else:
#                 print("Chatbot: Please provide the text to summarize.")

#         # Handle text-based queries
#         else:
#             print("Chatbot: Processing your query...")
#             response = generate_text(user_input)
#             print(f"Chatbot: {response}")

# # Run the chatbot
# if __name__ == "__main__":
#     multimodal_chatbot()

from transformers import pipeline
#from diffusers import StableDiffusionPipeline
from PIL import Image
from diffusers import DiffusionPipeline
import torch

from huggingface_hub import login
login(token="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")

# Load text generation model from Hugging Face
text_generator = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", max_length=200)

# Set device (Prefer GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
image_generator = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder").to(device)

def generate_image(prompt):
    try:
        image = image_generator(prompt).images[0]
        image_path = "kandinsky_image.png"
        image.save(image_path)
        return f"Chatbot: Image saved as {image_path}"
    except Exception as e:
        return f"Error generating image: {e}"

# Function to generate text
def generate_text(prompt):
    try:
        response = text_generator(prompt, max_length=200, num_return_sequences=1)
        return response[0]['generated_text']
    except Exception as e:
        return f"Error generating text: {e}"


# Multi-modal chatbot
def multimodal_chatbot():
    print("Welcome to the Multi-Modal Chatbot! Type 'exit' to quit.")
    
    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        # Image generation
        if "generate image" in user_input.lower():
            prompt = user_input.replace("generate image", "").strip()
            if prompt:
                print("Chatbot: Generating an image, please wait...")
                image_path = generate_image(prompt)
                print(f"Chatbot: Here's your image saved as {image_path}" if "Error" not in image_path else image_path)
            else:
                print("Chatbot: Please provide a description for the image.")

        # Image analysis (Placeholder)
        elif user_input.startswith("http") and any(ext in user_input for ext in [".jpg", ".png", ".jpeg"]):
            print("Chatbot: Analyzing the image...")
            analysis_result = analyze_image(user_input)
            print(f"Chatbot: {analysis_result}")

        # General text processing
        else:
            print("Chatbot: Processing your query...")
            response = generate_text(user_input)
            print(f"Chatbot: {response}")

# Run the chatbot
if __name__ == "__main__":
    multimodal_chatbot()
