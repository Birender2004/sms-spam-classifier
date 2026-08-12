# 🛡️ SMS Spam Classifier

A simple **Natural Language Processing (NLP) and Machine Learning** application that classifies SMS messages as **Spam** or **Not Spam**.

The project demonstrates how raw text can be processed, converted into numerical features using **TF-IDF**, and classified using **Multinomial Naive Bayes**. A **Streamlit** interface is used to make the model interactive.

---

## 📌 Project Overview

- Built to solve a real-world **text classification** problem.
- Takes an SMS message as input and predicts whether it is **Spam** or **Not Spam**.
- Applies NLP techniques to clean and prepare the text before prediction.
- Uses a trained machine-learning model for classification.
- Provides an interactive interface through Streamlit.

---

## 🧠 Key Concepts

- **Natural Language Processing (NLP)** — Used to process and prepare human-readable text for machine learning.
- **Text Preprocessing** — Cleans the SMS text before it is passed to the model.
- **Feature Extraction** — Converts text into numerical data that a machine-learning model can understand.
- **Text Classification** — Assigns each SMS message to one of two categories: Spam or Not Spam.
- **Model Inference** — Uses the trained model to generate predictions for new messages.

---

## 🧰 Tech Stack

- 🐍 **Python** — Core programming language used to build the application.
- 📝 **NLTK** — Used for NLP preprocessing such as tokenization, stop-word removal, and stemming.
- 🔢 **TF-IDF** — Converts SMS text into numerical feature vectors.
- 🧠 **Scikit-learn** — Provides the machine-learning and feature-extraction components.
- 📊 **Multinomial Naive Bayes** — Used to classify messages as Spam or Not Spam.
- 🎨 **Streamlit** — Provides the interactive web interface.
- 🐳 **Docker** — Used to package the application and its dependencies into a container.
- 💾 **Pickle** — Used to save and load the trained model and TF-IDF vectorizer.

---

## ⚙️ How It Works

1. The user enters an SMS message.
2. The message is cleaned and preprocessed using NLP techniques.
3. The processed text is converted into numerical features using TF-IDF.
4. The TF-IDF representation is passed to the trained Multinomial Naive Bayes model.
5. The model predicts whether the message is **Spam** or **Not Spam**.
6. The prediction is displayed through the Streamlit interface.

---

## ✨ Key Features

- 📨 Spam and Not Spam classification
- 🧹 NLP-based text preprocessing
- 🔢 TF-IDF feature extraction
- 🧠 Multinomial Naive Bayes classification
- 🖥️ Interactive Streamlit interface
- 🐳 Docker support
- ⚡ Fast prediction using a pre-trained model

---

## 🚀 Run Locally

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run main.py
```

The application will open in your browser.

---

## 🐳 Docker

Build the Docker image:

```bash
docker build -t sms-spam-classifier .
```

Run the application:

```bash
docker run -p 8501:8501 sms-spam-classifier
```

---

## 🔮 Future Improvements

- Improve model evaluation and tuning.
- Experiment with additional NLP and classification algorithms.
- Add model performance metrics.
- Improve the application interface.
- Deploy the application to a cloud platform.

---

### Built with Python • NLP • Machine Learning
