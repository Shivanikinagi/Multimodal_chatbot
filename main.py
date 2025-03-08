import streamlit as st
from Task2.chatbot import load_text_model, load_image_model, generate_text, generate_image
from PIL import Image
from Task1.task1 import generate_summary

# Set up the Streamlit app title
st.title("Multi-Modal App: Summarize Text, Generate Text, and Generate Image")

# Load models
@st.cache_resource
def load_models():
    try:
        text_generator = load_text_model()
        image_generator = load_image_model()
        return text_generator, image_generator
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Load models
text_generator, image_generator = load_models()

# Streamlit app layout
def main():
    st.write("Welcome to the Multi-Modal App! Choose an option below.")

    # Option selection
    option = st.radio("Select an option:", ["Summarize Text", "Generate Text", "Generate Image"])

    if option == "Summarize Text":
        st.subheader("Text Summarization")
        text = st.text_area("Enter the text to summarize:")
        if st.button("Summarize"):
            if text.strip():
                # Summarize text
                summary = generate_summary(text, summary_length=3)  # Use your trained summarization model
                st.success("Summary:")
                st.write(summary)
            else:
                st.warning("Please enter some text to summarize.")

    elif option == "Generate Text":
        st.subheader("Text Generation")
        prompt = st.text_input("Enter your text prompt:")
        if st.button("Generate Text"):
            if prompt:
                if text_generator is None:
                    st.error("Text generation model failed to load. Please check your setup.")
                else:
                    # Generate text
                    generated_text = generate_text(prompt, text_generator)
                    st.success("Generated Text:")
                    st.write(generated_text)
            else:
                st.warning("Please enter a text prompt.")

    elif option == "Generate Image":
        st.subheader("Image Generation")
        prompt = st.text_input("Enter your image prompt:")
        if st.button("Generate Image"):
            if prompt:
                if image_generator is None:
                    st.error("Image generation model failed to load. Please check your setup.")
                else:
                    # Generate image
                    with st.spinner("Generating image, please wait..."):
                        image = generate_image(prompt, image_generator)
                    if isinstance(image, Image.Image):
                        st.success("Generated Image:")
                        st.image(image, caption=f"Generated Image: {prompt}", use_column_width=True)
                    else:
                        st.error(image)  # Display error message
            else:
                st.warning("Please enter an image prompt.")

# Run the app
if __name__ == "__main__":
    main()