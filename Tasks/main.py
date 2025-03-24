import os
import streamlit as st
from PIL import Image
import google.generativeai as genai
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, PNDMScheduler
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import io
import traceback
import torch
import re
import logging

os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

nltk.download("punkt")

sentiment_pipeline = pipeline("sentiment-analysis")

st.set_page_config(page_title="Multi-Modal Chatbot", layout="centered")

st.title("🤖 Multi-Modal Chatbot with Summarization, Sentiment, and Image Analysis")
st.write("AI chatbot that can handle text, images, sentiment analysis, and summarization.")

MODEL_ID = "runwayml/stable-diffusion-v1-5"
MODEL_PATH = "./models/stable-diffusion-v1-5"

@st.cache_resource(ttl=3600, show_spinner=False)
def load_stable_diffusion():
    """
    Load the Stable Diffusion model.
    """
    try:
        if not os.path.exists(MODEL_PATH):
            logging.info("Downloading Stable Diffusion model (first-time setup)...")
            model = DiffusionPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float32,
                use_safetensors=True
            )
            model.scheduler = DPMSolverMultistepScheduler.from_config(model.config)
            model.save_pretrained(MODEL_PATH)
        else:
            logging.info("Loading model from local storage...")
            model = DiffusionPipeline.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.float32,
                use_safetensors=True
            )
            model.scheduler = DPMSolverMultistepScheduler.from_config(model.scheduler.config)

        return model.to("cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.write(traceback.format_exc())  
        return None

pipe = load_stable_diffusion()

def extractive_summary(text, num_sentences=3):
    """
    Generate an extractive summary of the given text.
    """
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text

    vectorizer = CountVectorizer().fit_transform(sentences)
    vectors = vectorizer.toarray()
    cosine_matrix = cosine_similarity(vectors)

    scores = np.sum(cosine_matrix, axis=1)
    ranked_sentences = [sentences[i] for i in np.argsort(scores, axis=0)[::-1]]
    summary = " ".join(ranked_sentences[:num_sentences])

    return summary

def generate_text_response(prompt):
    """
    Generate a text response using Google Gemini.
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error while generating text: {str(e)}"

def generate_image(prompt):
    """
    Generate an image based on the given prompt using Stable Diffusion.
    """
    try:
        if not pipe:
            st.error("❌ Stable Diffusion model is not loaded.")
            return None, "❌ Model not loaded properly."

        logging.info(f"Generating image for prompt: {prompt}")
        image = pipe(prompt, num_inference_steps=25).images[0]
        image_path = "./models/generated_image.png"
        os.makedirs("./models", exist_ok=True)  
        image.save(image_path)
        logging.info(f"Image saved at: {image_path}")
        return image_path, None
    except Exception as e:
        logging.error(f"❌ Error while generating the image: {str(e)}")
        st.write(traceback.format_exc())  
        return None, f"❌ Error while generating the image: {str(e)}"
    
def analyze_sentiment(text):
    """
    Analyze the sentiment of the given text.
    """
    try:
        result = sentiment_pipeline(text)[0]
        label = result["label"]
        score = result["score"]

        if label == "POSITIVE":
            response = "😊 Glad to hear that! How can I help you further?"
        elif label == "NEGATIVE":
            response = "😔 I'm sorry to hear that. Let me try to make things better."
        else:
            response = "😐 Okay, I'll do my best to help you with that."

        return label, score, response
    except Exception as e:
        return f"❌ Error in sentiment analysis: {str(e)}"

def analyze_uploaded_image(uploaded_file):
    """
    Analyze the uploaded image using Google Gemini.
    """
    try:
        img_bytes = uploaded_file.read()
        img = Image.open(io.BytesIO(img_bytes))

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([img])

        description = response.text if response.text else "No description generated."
        return description
    except Exception as e:
        return f"❌ Error in image analysis: {str(e)}"

def handle_combined_query(text, uploaded_file=None):
    """
    Handle combined queries involving both text and image.
    """
    try:
        text_response, image_response, image_description = None, None, None

        image_keywords = ["generate", "image", "show", "create", "draw", "picture", "visualize"]
        match = re.search(r"(?:generate|image|show|create|draw|picture)\s+(.+)", text, re.IGNORECASE)
        image_prompt = match.group(1).strip() if match else None

        text_query = text
        for word in image_keywords:
            text_query = text_query.replace(word, "").strip()

        if text_query:
            text_response = generate_text_response(text_query)

        if image_prompt:
            image_path, error = generate_image(image_prompt)
            if error:
                image_response = error
            else:
                image_response = image_path

        if uploaded_file:
            image_description = analyze_uploaded_image(uploaded_file)

        return text_response, image_response, image_description
    except Exception as e:
        return f"❌ Error in combined query handling: {str(e)}", None, None
    
menu = st.sidebar.radio(
    "Choose an option:",
    (
        "💬 Chat with AI",
        "📄 Summarize Text",
        "😎 Sentiment Analysis",
        "🎨 Generate Image",
        "📸 Upload and Analyze Image",
        "🤖 Combined Query"
    )
)

if menu == "💬 Chat with AI":
    st.subheader("💬 Chat with AI")
    prompt = st.text_input("Enter your prompt:")

    if st.button("Generate Response"):
        if prompt:
            with st.spinner("Generating response..."):
                response = generate_text_response(prompt)
            st.write("**Response:**")
            st.write(response)
        else:
            st.warning("⚠️ Please enter a prompt.")

elif menu == "📄 Summarize Text":
    st.subheader("📄 Summarize Text")
    text = st.text_area("Enter the text to summarize:")
    num_sentences = st.slider("Number of sentences in summary:", 1, 10, 3)

    if st.button("Generate Summary"):
        if text:
            with st.spinner("Summarizing text..."):
                summary = extractive_summary(text, num_sentences)
            st.write("**Summary:**")
            st.write(summary)
        else:
            st.warning("⚠️ Please enter some text to summarize.")

elif menu == "😎 Sentiment Analysis":
    st.subheader("😎 Sentiment Analysis")
    text = st.text_area("Enter text for sentiment analysis:")

    if st.button("Analyze Sentiment"):
        if text:
            with st.spinner("Analyzing sentiment..."):
                label, score, response = analyze_sentiment(text)
            st.write(f"**Sentiment:** {label} (Score: {score:.2f})")
            st.write(f"**Response:** {response}")
        else:
            st.warning("⚠️ Please enter some text to analyze.")

elif menu == "🎨 Generate Image":
    st.subheader("🎨 Generate Image")
    prompt = st.text_input("Enter prompt for image generation:")

    if st.button("Generate Image"):
        if prompt:
            with st.spinner("Generating image..."):
                image_path, error = generate_image(prompt)
            if error:
                st.error(error)
            else:
                st.image(image_path, caption="Generated Image", use_column_width=True)
        else:
            st.warning("⚠️ Please enter a prompt.")

elif menu == "📸 Upload and Analyze Image":
    st.subheader("📸 Upload and Analyze Image")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if st.button("Analyze Image"):
        if uploaded_file is not None:
            with st.spinner("Analyzing image..."):
                result = analyze_uploaded_image(uploaded_file)
            st.write("**Analysis Result:**")
            st.write(result)
        else:
            st.warning("⚠️ Please upload an image.")

elif menu == "🤖 Combined Query":
    st.subheader("🤖 Combined Query")
    prompt = st.text_input("Enter combined query:")
    uploaded_file = st.file_uploader("Optional: Upload an image...", type=["jpg", "jpeg", "png"])

    if st.button("Submit Query"):
        if prompt or uploaded_file:
            with st.spinner("Processing query..."):
                text_response, image_response, image_description = handle_combined_query(prompt, uploaded_file)

            if text_response:
                st.write("**Text Response:**")
                st.write(text_response)

            if image_response:
                if isinstance(image_response, str) and image_response.startswith("❌"):
                    st.error(image_response)  
                else:
                    st.image(image_response, caption="Generated Image", use_column_width=True)

            if image_description:
                st.write("**Image Description:**")
                st.write(image_description)
                
        else:
            st.warning("⚠️ Please enter a query or upload an image.")

