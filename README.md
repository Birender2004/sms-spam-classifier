<div align="center">

🛡️ SMS Spam Classifier

NLP • Text Classification • Machine Learning

A lightweight machine-learning application that classifies SMS messages as Spam or Not Spam using natural language processing, TF-IDF feature extraction, and Multinomial Naive Bayes.

<p>
<img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/NLP-NLTK-4B8BBE?style=for-the-badge">
<img src="https://img.shields.io/badge/ML-Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">
<img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
<img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white">
</p>

</div>

📌 About the Project

Spam messages are a common problem in digital communication. This project demonstrates an end-to-end NLP text-classification pipeline that takes an SMS message, preprocesses the text, converts it into numerical features, and predicts whether the message is Spam or Not Spam.

The trained model and TF-IDF vectorizer are integrated into a Streamlit application, allowing users to enter a message and receive a prediction through a simple interface.

🎯 Objective

Build a practical machine-learning application that demonstrates how text preprocessing, feature engineering, classification, and model deployment can work together in a real-world NLP problem.

⚙️ How It Works

          SMS MESSAGE
               │
               ▼
       TEXT PREPROCESSING
               │
       ┌───────┼────────┐
       ▼       ▼        ▼
   Tokenize  Remove   Stemming
             Stopwords
       └───────┼────────┘
               ▼
        TF-IDF FEATURES
               │
               ▼
   MULTINOMIAL NAIVE BAYES
               │
          ┌────┴────┐
          ▼         ▼
        SPAM     NOT SPAM

🧰 Tech Stack

💻 Core Technologies

<p>
<img src="https://img.shields.io/badge/Python-Core-3776AB?style=flat-square&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white">
<img src="https://img.shields.io/badge/Docker-Containerization-2496ED?style=flat-square&logo=docker&logoColor=white">
</p>

🧠 NLP & Machine Learning

<p>
<img src="https://img.shields.io/badge/NLTK-Text%20Processing-4B8BBE?style=flat-square">
<img src="https://img.shields.io/badge/TF--IDF-Feature%20Extraction-6A5ACD?style=flat-square">
<img src="https://img.shields.io/badge/Naive%20Bayes-Classification-F7931E?style=flat-square">
<img src="https://img.shields.io/badge/Scikit--learn-ML%20Framework-F7931E?style=flat-square&logo=scikit-learn&logoColor=white">
</p>

🔍 Technology Roles

Technology

Purpose in the Project

🐍 Python

Core programming language used for the application and ML pipeline

📝 NLTK

Handles text preprocessing such as tokenization, stop-word removal, and stemming

🔢 TF-IDF

Converts processed SMS text into numerical feature vectors

🧠 Multinomial Naive Bayes

Classifies messages into Spam or Not Spam

🔬 Scikit-learn

Provides the ML utilities, vectorizer, and classification implementation

🎨 Streamlit

Provides the interactive web interface for entering messages and viewing predictions

🐳 Docker

Packages the application and its dependencies into a portable container

💾 Pickle

Stores and loads the trained ML model and TF-IDF vectorizer

✨ Key Features

📨 Classifies SMS messages as Spam or Not Spam

🧹 NLP-based text preprocessing

🔢 TF-IDF feature extraction

🧠 Multinomial Naive Bayes classification

🖥️ Interactive Streamlit interface

🐳 Docker support

⚡ Uses pre-trained model artifacts for prediction

🗂️ Project Structure

sms-spam-classifier/
│
├── main.py
├── mnb.pkl
├── tfidf.pkl
├── requirements.txt
├── Dockerfile
├── Jenkinsfile
└── README.md

File

Purpose

main.py

Streamlit application and prediction pipeline

mnb.pkl

Trained Multinomial Naive Bayes model

tfidf.pkl

Trained TF-IDF vectorizer

requirements.txt

Python dependencies

Dockerfile

Container configuration

Jenkinsfile

CI/CD configuration

README.md

Project documentation

🚀 Run Locally

1. Clone the repository

git clone https://github.com/Birender2004/sms-spam-classifier.git
cd sms-spam-classifier

2. Install dependencies

pip install -r requirements.txt

3. Start the application

streamlit run main.py

Open the application at:

http://localhost:8501

🐳 Run with Docker

Build the image

docker build -t sms-spam-classifier .

Run the container

docker run -p 8501:8501 sms-spam-classifier

Then open:

http://localhost:8501

🧠 What This Project Demonstrates

Text Processing → Feature Engineering → Machine Learning → Model Inference → Web Application → Containerization

This project demonstrates practical understanding of how a traditional NLP machine-learning model can be turned into an interactive application.

🔮 Future Improvements

Improve model evaluation and hyperparameter tuning

Experiment with additional NLP/classification models

Add model performance metrics

Improve the user interface

Deploy the application to a cloud platform

<div align="center">

Built with Python • NLP • Machine Learning • Streamlit

</div>
