Multi-Modal AI Chatbot  

Overview  
This project implements a Multi-Modal AI Chatbot integrating Google Gemini AI and Stable Diffusion v1.5. It provides text generation, AI-powered image generation, and image analysis capabilities through a user-friendly Streamlit UI.  

Key Features  
- Text Generation: Uses Gemini-1.5-flash to generate responses based on user queries.  
- AI-Powered Image Generation: Utilizes Stable Diffusion v1.5 to generate images from textual prompts.  
- Image Analysis: Allows users to upload images for AI-based analysis.  
- Streamlit UI: A web-based interface for seamless user interaction.  
- Logging & Error Handling: Implements robust logging for debugging and tracking API responses.  

Table of Contents  
1. Requirements
2. Installation
3. Project Structure
4. Usage
5. API Integration
6. Model Weights

Requirements  
pip install -r requirements.txt

You also need API keys for:  
- Google Gemini AI (`GOOGLE_API_KEY`)  
- Hugging Face Token (`HUGGINGFACE_TOKEN`)  

Installation  
Clone the repository and navigate to Task 2:  
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name/Task_2_Multi-Modal_AI_Chatbot
pip install -r requirements.txt

Set up environment variables:  
export GOOGLE_API_KEY="your_google_api_key_here"
export HUGGINGFACE_TOKEN="your_huggingface_token_here"

Project Structure  
Task_2_Multi-Modal_AI_Chatbot/
├── app.py                    # Main script for running the chatbot
├── modules/
│   ├── text_generator.py      # Handles text generation using Google Gemini AI
│   ├── image_generator.py     # Uses Stable Diffusion v1.5 for image generation
│   ├── image_analyzer.py      # Processes uploaded images
│   ├── utils.py               # Helper functions and API request handling
├── requirements.txt           # Dependencies
├── README.md                  # Documentation (this file)

Usage  
streamlit run app.py

API Integration  
This chatbot integrates:  
- Google Gemini-1.5-flash for text processing  
- Stable Diffusion v1.5 via Hugging Face for image generation  

Model Weights  
Download pre-trained model weights from:  
[Task 2 Weights](https://drive.google.com/drive/folders/1v_o14IJljgI0EaJIGbbs4_iOxo9V8V-2)  