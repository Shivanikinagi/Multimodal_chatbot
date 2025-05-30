import os
import streamlit as st
from PIL import Image
import google.generativeai as genai
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from transformers import pipeline
import nltk
from transformers import AutoImageProcessor
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import io
import traceback
import torch
import re
import logging
from transformers import MarianMTModel, MarianTokenizer
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Download NLTK data
nltk.download("punkt")

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize Streamlit app
st.set_page_config(page_title="NextGen AI Companion", layout="centered")

# Initialize session state for gamification and language
if "points" not in st.session_state:
    st.session_state.points = 0
if "language" not in st.session_state:
    st.session_state.language = "en"

# Language mapping for translation
LANGUAGE_MAP = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
}

# Load translation models for multilingual support
@st.cache_resource
def load_translation_model(src_lang, tgt_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"❌ Failed to load translation model: {str(e)}")
        return None, None

# Load emotion detection model for granularity
@st.cache_resource
def load_emotion_model():
    try:
        model = SentenceTransformer('distilbert-base-uncased')
        return model
    except Exception as e:
        st.error(f"❌ Failed to load emotion model: {str(e)}")
        return None

emotion_model = load_emotion_model()

# Define emotion labels and corresponding visuals
EMOTIONS = ["excited", "frustrated", "confused", "happy", "neutral"]
EMOTION_VISUALS = {
    "excited": {"emoji": "🎉", "color": "background-color: #f7c948;"},
    "frustrated": {"emoji": "🤔", "color": "background-color: #ff6b6b;"},
    "confused": {"emoji": "❓", "color": "background-color: #8e44ad;"},
    "happy": {"emoji": "😊", "color": "background-color: #2ecc71;"},
    "neutral": {"emoji": "😐", "color": "background-color: #7f8c8d;"},
}

# Load Stable Diffusion model
MODEL_ID = "runwayml/stable-diffusion-v1-5"
MODEL_PATH = "./models/stable-diffusion-v1-5"

@st.cache_resource(ttl=3600, show_spinner=False)
def load_stable_diffusion():
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

# Translation function
def translate_text(text, src_lang, tgt_lang):
    if src_lang == tgt_lang:
        return text
    tokenizer, model = load_translation_model(src_lang, tgt_lang)
    if not tokenizer or not model:
        return text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

# Emotion detection with granularity
def analyze_emotion(text):
    try:
        if not emotion_model:
            return "neutral", 0.0, "😐 Neutral response. How can I assist you?"
        emotion_embeddings = {
            "excited": emotion_model.encode("I am so thrilled and full of energy!"),
            "frustrated": emotion_model.encode("This is so annoying, I can’t figure it out!"),
            "confused": emotion_model.encode("I don’t understand what’s happening here."),
            "happy": emotion_model.encode("I feel great, everything is going well!"),
            "neutral": emotion_model.encode("I have no strong feelings about this."),
        }
        text_embedding = emotion_model.encode(text)
        similarities = {emotion: F.cosine_similarity(
            torch.tensor(text_embedding), torch.tensor(emb), dim=0
        ).item() for emotion, emb in emotion_embeddings.items()}
        detected_emotion = max(similarities, key=similarities.get)
        score = similarities[detected_emotion]
        responses = {
            "excited": "🎉 Wow, you’re excited! Let’s keep the energy up!",
            "frustrated": "🤔 I’m sorry you’re feeling frustrated. Let’s work through this together.",
            "confused": "❓ I can help clarify things for you! What’s confusing?",
            "happy": "😊 I’m glad you’re happy! How can I make your day even better?",
            "neutral": "😐 Okay, let’s see how I can assist you."
        }
        st.session_state.points += 10  # Award points for interaction
        return detected_emotion, score, responses[detected_emotion]
    except Exception as e:
        return "neutral", 0.0, f"❌ Error in emotion analysis: {str(e)}"

# Extractive summarization
def extractive_summary(text, num_sentences=3):
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

# Generate text response using Google Gemini
def generate_text_response(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error while generating text: {str(e)}"

# Generate image using Stable Diffusion
def generate_image(prompt):
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
        st.session_state.points += 20  # Award points for image generation
        return image_path, None
    except Exception as e:
        logging.error(f"❌ Error while generating the image: {str(e)}")
        st.write(traceback.format_exc())
        return None, f"❌ Error while generating the image: {str(e)}"

# Analyze uploaded image using Google Gemini
def analyze_uploaded_image(uploaded_file):
    try:
        img_bytes = uploaded_file.read()
        img = Image.open(io.BytesIO(img_bytes))
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([img])
        description = response.text if response.text else "No description generated."
        st.session_state.points += 15  # Award points for image analysis
        return description
    except Exception as e:
        return f"❌ Error in image analysis: {str(e)}"

# Handle combined queries
def handle_combined_query(text, uploaded_file=None):
    try:
        text_response, image_response, image_description = None, None, None
        image_keywords = ["generate", "image", "show", "create", "draw", "picture", "visualize"]
        match = re.search(r"(?:generate|image|show|create|draw|picture)\s+(.+)", text, re.IGNORECASE)
        image_prompt = match.group(1).strip() if match else None
        text_query = text
        for word in image_keywords:
            text_query = text_query.replace(word, "").strip()

        # Analyze emotion
        if text_query:
            text_en = translate_text(text_query, st.session_state.language, "en")
            emotion_label, _, emotion_response = analyze_emotion(text_en)
            emotion_response = translate_text(emotion_response, "en", st.session_state.language)
            text_response = f"**Emotion:** {emotion_label}\n{emotion_response}"
            ai_response = generate_text_response(text_en)
            ai_response = translate_text(ai_response, "en", st.session_state.language)
            text_response += f"\n**AI Response:** {ai_response}"

        if image_prompt:
            image_path, error = generate_image(image_prompt)
            if error:
                image_response = error
            else:
                image_response = image_path

        if uploaded_file:
            image_description = analyze_uploaded_image(uploaded_file)
            image_description = translate_text(image_description, "en", st.session_state.language)

        return text_response, image_response, image_description
    except Exception as e:
        return f"⚠️ Error in combined query handling: {str(e)}", None, None

# Trivia game for gamification
def trivia_game():
    questions = [
        {
            "question": "What is the capital city of Australia?",
            "options": ["Sydney", "Canberra", "Melbourne", "Brisbane"],
            "correct": "Canberra"
        },
        {
            "question": "Who wrote the play \"Romeo and Juliet\"?",
            "options": ["Charles Dickens", "William Shakespeare", "Mark Twain", "Jane Austen"],
            "correct": "William Shakespeare"
        },
        {
            "question": "What is the chemical symbol for gold?",
            "options": ["Ag", "Au", "Gd", "Ga"],
            "correct": "Au"
        },
        {
            "question": "What planet is known as the Red Planet?",
            "options": ["Venus", "Mars", "Jupiter", "Saturn"],
            "correct": "Mars"
        },
        {
            "question": "Who was the first President of the United States?",
            "options": ["Thomas Jefferson", "Abraham Lincoln", "George Washington", "John Adams"],
            "correct": "George Washington"
        },
        {
            "question": "In which year did the Titanic sink?",
            "options": ["1905", "1912", "1920", "1898"],
            "correct": "1912"
        },
        {
            "question": "Which river is the longest in the world?",
            "options": ["Amazon River", "Mississippi River", "Nile River", "Yangtze River"],
            "correct": "Nile River"
        },
        {
            "question": "What is the largest desert in the world?",
            "options": ["Arabian Desert", "Gobi Desert", "Sahara Desert", "Kalahari Desert"],
            "correct": "Sahara Desert"
        },
        {
            "question": "Who played Jack Dawson in the movie \"Titanic\"?",
            "options": ["Brad Pitt", "Tom Cruise", "Leonardo DiCaprio", "Johnny Depp"],
            "correct": "Leonardo DiCaprio"
        },
        {
            "question": "What is the name of the fictional African country in \"Black Panther\"?",
            "options": ["Zamunda", "Wakanda", "Genosha", "Narnia"],
            "correct": "Wakanda"
        },
        {
            "question": "How many players are there in a soccer team?",
            "options": ["10", "11", "12", "9"],
            "correct": "11"
        },
        {
            "question": "Which country won the FIFA World Cup in 2018?",
            "options": ["Brazil", "Germany", "France", "Argentina"],
            "correct": "France"
        },
        {
            "question": "Who is the author of the \"Harry Potter\" series?",
            "options": ["Suzanne Collins", "J.K. Rowling", "Stephen King", "George R.R. Martin"],
            "correct": "J.K. Rowling"
        },
        {
            "question": "What is the title of the first book in the \"Lord of the Rings\" series?",
            "options": ["The Two Towers", "The Return of the King", "The Fellowship of the Ring", "The Hobbit"],
            "correct": "The Fellowship of the Ring"
        },
        {
            "question": "Which band released the album \"Abbey Road\"?",
            "options": ["The Rolling Stones", "The Beatles", "Queen", "Pink Floyd"],
            "correct": "The Beatles"
        },
        {
            "question": "Who is known as the \"Queen of Pop\"?",
            "options": ["Beyoncé", "Madonna", "Lady Gaga", "Taylor Swift"],
            "correct": "Madonna"
        },
        {
            "question": "Which animal is known as the 'King of the Jungle'?",
            "options": ["Tiger", "Lion", "Leopard", "Cheetah"],
            "correct": "Lion"
        }
    ]
    question = random.choice(questions)
    st.write(f"**Trivia Question:** {question['question']}")
    option = st.radio("Select an answer:", question["options"])
    if st.button("Submit Answer"):
        if option == question["correct"]:
            st.session_state.points += 50
            st.success("Correct! +50 points")
        else:
            st.error("Incorrect. Try again!")
        return True
    return False

# Streamlit UI
st.title("🤖 NextGen AI Companion")
st.write("AI chatbot that handles text, images, emotion analysis, summarization and gamification.")

# Sidebar for navigation and gamification
st.sidebar.write(f"**Points:** {st.session_state.points}")
progress = min(st.session_state.points, 100)
st.sidebar.progress(progress)
st.sidebar.write("Earn points by interacting with the chatbot!")

# Language selection
language_choice = st.sidebar.selectbox("Select Language:", list(LANGUAGE_MAP.values()))
st.session_state.language = [k for k, v in LANGUAGE_MAP.items() if v == language_choice][0]

# Menu options
menu = st.sidebar.radio(
    "Choose an option:",
    (
        "💬 Chat with AI",
        "📄 Summarize Text",
        "😎 Emotion Analysis",
        "🎨 Generate Image",
        "📸 Upload and Analyze Image",
        "🤖 Combined Query",
        "🎮 Play Trivia",
    )
)

# Emotion-driven visuals container
def display_emotion_visual(emotion):
    if emotion in EMOTION_VISUALS:
        visual = EMOTION_VISUALS[emotion]
        st.markdown(
            f"<div style='{visual['color']} padding: 10px; border-radius: 5px;'>{visual['emoji']} {emotion.capitalize()}</div>",
            unsafe_allow_html=True
        )

if menu == "💬 Chat with AI":
    st.subheader("💬 Chat with AI")
    prompt = st.text_input("Enter your prompt:")
    if st.button("Generate Response"):
        if prompt:
            with st.spinner("Analyzing emotion..."):
                prompt_en = translate_text(prompt, st.session_state.language, "en")
                emotion_label, _, emotion_response = analyze_emotion(prompt_en)
                emotion_response = translate_text(emotion_response, "en", st.session_state.language)
            display_emotion_visual(emotion_label)
            st.write(f"**Response:** {emotion_response}")
            with st.spinner("Generating response..."):
                ai_response = generate_text_response(prompt_en)
                ai_response = translate_text(ai_response, "en", st.session_state.language)
            st.write("**AI Response:**")
            st.write(ai_response)
        else:
            st.warning("⚠️ Please enter a prompt.")

elif menu == "📄 Summarize Text":
    st.subheader("📄 Summarize Text")
    text = st.text_area("Enter the text to summarize:")
    num_sentences = st.slider("Number of sentences in summary:", 1, 10, 3)
    if st.button("Generate Summary"):
        if text:
            with st.spinner("Summarizing text..."):
                text_en = translate_text(text, st.session_state.language, "en")
                summary = extractive_summary(text_en, num_sentences)
                summary = translate_text(summary, "en", st.session_state.language)
            st.write("**Summary:**")
            st.write(summary)
        else:
            st.warning("⚠️ Please enter some text to summarize.")

elif menu == "😎 Emotion Analysis":
    st.subheader("😎 Emotion Analysis")
    text = st.text_area("Enter text for emotion analysis:")
    if st.button("Analyze Emotion"):
        if text:
            with st.spinner("Analyzing emotion..."):
                text_en = translate_text(text, st.session_state.language, "en")
                emotion_label, score, response = analyze_emotion(text_en)
                response = translate_text(response, "en", st.session_state.language)
            display_emotion_visual(emotion_label)
            st.write(f"**Emotion:** {emotion_label} (Score: {score:.2f})")
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
                result = translate_text(result, "en", st.session_state.language)
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
                emotion_match = re.search(r"\*\*Emotion:\*\* (\w+)", text_response)
                if emotion_match:
                    emotion_label = emotion_match.group(1).lower()
                    display_emotion_visual(emotion_label)
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

elif menu == "🎮 Play Trivia":
    st.subheader("🎮 Play Trivia")
    st.write("Answer trivia questions to earn points!")
    trivia_game()