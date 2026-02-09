import streamlit as st
import pickle

import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

def transform(text):
  text= text.lower()
  text= nltk.word_tokenize(text)
  y=[]

  ps = PorterStemmer()

  for i in text:
    if(i.isalnum()):
      y.append(i)

  test=y[:]
  y.clear()

  for i in test:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text=y[:]

  y.clear()

  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)




tfidf= pickle.load(open('tfidf.pkl','rb'))
model= pickle.load(open("mnb.pkl", "rb"))

st.title("SMS Spam Classifier")

sms_input= st.text_area("Enter your message")


if st.button("Predict"):

  # Data preprocessing
  transformed_sms= transform(sms_input)

  # Vectorizing the input
  vectorized_sms= tfidf.transform([transformed_sms])

  #Prediction
  result= model.predict(vectorized_sms)[0]

  #Displaying the result
  if result==1:
      st.header("Spam")

  else:
      st.header("Not Spam")