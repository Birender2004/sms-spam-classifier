# import streamlit as st
# import pickle

# import string
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer

# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')

# def transform(text):
#   text= text.lower()
#   text= nltk.word_tokenize(text)
#   y=[]

#   ps = PorterStemmer()

#   for i in text:
#     if(i.isalnum()):
#       y.append(i)

#   test=y[:]
#   y.clear()

#   for i in test:
#     if i not in stopwords.words('english') and i not in string.punctuation:
#       y.append(i)

#   text=y[:]

#   y.clear()

#   for i in text:
#     y.append(ps.stem(i))

#   return " ".join(y)




# tfidf= pickle.load(open('tfidf.pkl','rb'))
# model= pickle.load(open("mnb.pkl", "rb"))

# st.title("SMS Spam Classifier")

# sms_input= st.text_area("Enter your message")


# if st.button("Predict"):

#   # Data preprocessing
#   transformed_sms= transform(sms_input)

#   # Vectorizing the input
#   vectorized_sms= tfidf.transform([transformed_sms])

#   #Prediction
#   result= model.predict(vectorized_sms)[0]

#   #Displaying the result
#   if result==1:
#       st.header("Spam")

#   else:
#       st.header("Not Spam")


import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from prometheus_client import start_http_server, Counter

# -------------------- PROMETHEUS SERVER (FIXED) --------------------
# Start metrics server only once (important for Streamlit reruns)
if "metrics_started" not in st.session_state:
    try:
        start_http_server(8000, addr="0.0.0.0")
    except:
        pass
    st.session_state["metrics_started"] = True

# Counter metric
request_count = Counter('app_requests_total', 'Total number of predictions made')

# -------------------- NLTK SETUP (SAFE DOWNLOAD) --------------------
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

# -------------------- TEXT PREPROCESSING --------------------
def transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []

    ps = PorterStemmer()

    for i in text:
        if i.isalnum():
            y.append(i)

    temp = y[:]
    y.clear()

    for i in temp:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# -------------------- LOAD MODEL --------------------
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
model = pickle.load(open("mnb.pkl", "rb"))

# -------------------- STREAMLIT UI --------------------
st.title("SMS Spam Classifier")

sms_input = st.text_area("Enter your message")

if st.button("Predict"):

    # Increment Prometheus counter
    request_count.inc()

    transformed_sms = transform(sms_input)
    vectorized_sms = tfidf.transform([transformed_sms])
    result = model.predict(vectorized_sms)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
