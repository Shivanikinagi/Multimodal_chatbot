Extractive Summarization

Overview
This project implements an extractive text summarization system using NLTK and Sentence Transformers. The system ranks sentences based on their semantic similarity to the original text and selects the most informative ones for the summary.

Key Features
- Extractive Summarization: Selects key sentences from the input text to generate a concise summary.
- Diversity Control: Ensures that the selected sentences are diverse to improve summary quality.
- Evaluation Metrics: Uses cosine similarity, precision, recall, and F1-score to assess summarization performance.

Table of Contents
1. Requirements
2. Installation
3. Project Structure
4. Usage
5. Evaluation Metrics
6. Model Weights

Requirements
Install dependencies using:
pip install -r requirements.txt

Installation
Clone the repository and navigate to Task 1:

git clone https://github.com/your-username/your-repo-name.git

cd your-repo-name/Task_1_Extractive_Summarization

pip install -r requirements.txt

Download necessary NLTK resources:

import nltk

nltk.download('punkt')

nltk.download('stopwords')

nltk.download('wordnet')

Project Structure

Task_1_Extractive_Summarization/
├── model_training.ipynb       # Jupyter Notebook for training and testing
├── saved_model/               # Saved model files
├── summarization.py           # Main script for text summarization
├── requirements.txt           # Dependencies
├── README.md                  # Documentation (this file)

Usage
To generate a summary:
from summarization import generate_summary

Evaluation Metrics
The summarization model is evaluated using:
- Cosine Similarity: Measures similarity between generated and reference summaries.
- Precision: Proportion of true positive words in the summary.
- Recall: Proportion of reference summary words included in the generated summary.
- F1-Score: Harmonic mean of precision and recall.
- Confusion Matrix: Visualizes classification performance.

Model Weights
Download model weights from:

[Task 1 Weights](https://drive.google.com/drive/folders/1D7n7T4ONdFagFxLmxb8Tjs17lzFquHWB)
