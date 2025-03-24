Sentiment Analysis

Overview
This project implements a sentiment analysis system using RandomForestClassifier and TF-IDF Vectorization. It analyzes customer sentiments from airline-related tweets and classifies them into Negative, Neutral, or Positive categories.  

Key Features  
-Sentiment Classification: Classifies tweets into three sentiment categories (Negative, Neutral, Positive).  
-TF-IDF Vectorization: Converts text into numerical features for machine learning.  
-Random Forest Model: Trained with optimized hyperparameters for better accuracy.  
-Model Persistence: Saves the trained classifier and vectorizer for future use.  
-Confusion Matrix Visualization: Displays model performance metrics.  

Table of Contents
1. Requirements
2. Installation
3. roject Structure
4. Usage
5. Evaluation Metrics
6. Model Persistence

Requirements  
Install dependencies using:  
pip install -r requirements.txt

Installation
Clone the repository and navigate to the project directory:  

git clone https://github.com/your-username/your-repo-name.git

cd your-repo-name/sentiment_analysis

pip install -r requirements.txt

Download the Twitter Airline Sentiment Dataset using Kaggle API:  

import kagglehub

path = kagglehub.dataset_download("crowdflower/twitter-airline-sentiment")

Usage
python sentiment_analysis.py

Evaluation Metrics
The sentiment analysis model is evaluated using:  
- Accuracy: Measures overall correctness of predictions.  
- Precision: Evaluates how many predicted positive cases were truly positive.  
- Recall: Measures how many actual positive cases were correctly classified.  
- F1-Score: Harmonic mean of precision and recall.  
- Confusion Matrix: Visualizes classification performance.  

Model Persistence
import joblib

joblib.dump(classifier, "./fine_tuned_model/sentiment_classifier.pkl")
joblib.dump(vectorizer, "./fine_tuned_model/vectorizer.pkl")
