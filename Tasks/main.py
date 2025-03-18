import os
import streamlit as st
from diffusers import StableDiffusionPipeline
from PIL import Image
import google.generativeai as genai
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import io
import traceback
import torch  # Ensure torch is imported

# ✅ Secure API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyDSmcKB_A5N8c74AQKNXVJ03-YHfyTzF2A"

# ✅ Configure Google Gemini API
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# ✅ Download NLTK resources
nltk.download("punkt")

# ✅ Load sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# ✅ Initialize Streamlit app
st.set_page_config(page_title="Multi-Modal Chatbot", layout="centered")

# ✅ Title and Description
st.title("🤖 Multi-Modal Chatbot with Summarization, Sentiment, and Image Analysis")
st.write("AI chatbot that can handle text, images, sentiment analysis, and summarization.")

@st.cache_resource(ttl=3600, show_spinner=False)
def load_stable_diffusion():
    """
    Load the Stable Diffusion model.
    """
    try:
        model_id = "runwayml/stable-diffusion-v1-5"  # Use a public and stable model
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        return pipe
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.write(traceback.format_exc())  # Print full error details
        return None

# ✅ Extractive Summarization Section
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

# ✅ Text Generation Section
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

# ✅ Image Generation Section (Stable Diffusion)
def generate_image(prompt):
    """
    Generate an image based on the given prompt using Stable Diffusion.
    """
    pipe = load_stable_diffusion()
    if pipe is None:
        return None, "❌ Model not loaded properly."

    try:
        image = pipe(prompt).images[0]
        return image, None
    except Exception as e:
        st.error(f"❌ Error while generating the image: {str(e)}")
        st.write(traceback.format_exc())  # Print full error details
        return None, f"❌ Error while generating the image: {str(e)}"

# ✅ Sentiment Analysis Section
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

# ✅ Analyze Uploaded Image (Vision Model)
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

# ✅ Combined Query Handling
def handle_combined_query(text, uploaded_file=None):
    """
    Handle combined queries involving both text and image.
    """
    try:
        text_response, image_response, image_description = None, None, None

        # Better combined query splitting
        parts = text.split(" and ", 1)
        text_prompt = parts[0].strip() if len(parts) > 0 else ""
        image_prompt = parts[1].strip() if len(parts) > 1 else ""

        if text_prompt:
            text_response = generate_text_response(text_prompt)

        if image_prompt:
            image_response, image_error = generate_image(image_prompt)
            if image_error:
                image_response = image_error

        if uploaded_file:
            image_description = analyze_uploaded_image(uploaded_file)

        return text_response, image_response, image_description
    except Exception as e:
        return f"❌ Error in combined query handling: {str(e)}", None, None

# ✅ Streamlit Sidebar Menu
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

# ✅ Option 1: Chat with AI
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

# ✅ Option 2: Summarize Text
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

# ✅ Option 3: Sentiment Analysis
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

# ✅ Option 4: Generate Image
elif menu == "🎨 Generate Image":
    st.subheader("🎨 Generate Image")
    prompt = st.text_input("Enter prompt for image generation:")

    if st.button("Generate Image"):
        if prompt:
            with st.spinner("Generating image..."):
                image, error = generate_image(prompt)
            if error:
                st.error(error)
            else:
                st.image(image, caption="Generated Image", use_column_width=True)
        else:
            st.warning("⚠️ Please enter a prompt.")

# ✅ Option 5: Upload and Analyze Image
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

# ✅ Option 6: Combined Query
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
                st.image(image_response, caption="Generated Image", use_column_width=True)

            if image_description:
                st.write("**Image Description:**")
                st.write(image_description)
        else:
            st.warning("⚠️ Please enter a query or upload an image.")