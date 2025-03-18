Overview
This project implements a multi-modal chatbot capable of handling text summarization, sentiment analysis, and image generation. The chatbot integrates advanced AI models like Google Gemini API, Stable Diffusion, and Hugging Face Transformers to provide seamless interactions between text and images.

Key Features:

Extractive Summarization : Generates concise summaries of long documents by selecting important sentences.
Multi-Modal Chatbot : Handles both textual and visual inputs, generates relevant images, and integrates them into conversations.
Sentiment Analysis : Detects customer emotions (positive, negative, neutral) and responds appropriately.

Table of Contents

1)Requirements : pip install -r requirements.txt
2)Installation:Clone the Repository :

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
import nltk
nltk.download('punkt')
nltk.download('stopwords')
export GOOGLE_API_KEY="your-google-api-key"
from huggingface_hub import login
login("your-huggingface-token")

Project Structure
├── Task_1_Extractive_Summarization/
│   ├── model_training.ipynb          # Jupyter Notebook for summarization
│   ├── model_weights/                # Model weights (or link to Google Drive)
│   ├── saved_model/                  # Saved model files
│   └── README.md                     # Task-specific documentation

├── Task_2_MultiModal_Chatbot/
│   ├── model_training.ipynb          # Jupyter Notebook for chatbot
│   ├── model_weights/                # Model weights (or link to Google Drive)
│   ├── saved_model/                  # Saved model files
│   └── README.md                     # Task-specific documentation

├── Task_3_Sentiment_Analysis/
│   ├── model_training.ipynb          # Jupyter Notebook for sentiment analysis
│   ├── model_weights/                # Model weights (or link to Google Drive)
│   ├── saved_model/                  # Saved model files
│   └── README.md                     # Task-specific documentation

├── requirements.txt                  # List of dependencies
├── README.md                         # Main project documentation

Evaluation Metrics
The models are evaluated using the following metrics:

Accuracy : Overall correctness of predictions.
Precision : Proportion of true positives among predicted positives.
Recall : Proportion of true positives among actual positives.
F1-Score : Harmonic mean of precision and recall.
Confusion Matrix : Visual representation of classification performance.

Model Weights and Files
- Task 1 -https://drive.google.com/drive/folders/1D7n7T4ONdFagFxLmxb8Tjs17lzFquHWB
- Task 2 -https://drive.google.com/drive/folders/1v_o14IJljgI0EaJIGbbs4_iOxo9V8V-2
- Task 3 -https://drive.google.com/drive/folders/1XpmE34v6xlqynm5nleYJwoNrS6p0R1mc


