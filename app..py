import gradio as gr
import joblib
import string
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
stop = set(stopwords.words("english"))

model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

def clean(text):
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    return " ".join(w for w in text.split() if w not in stop)

def check(msg):
    v = tfidf.transform([clean(msg)])
    return model.predict(v)[0]

gr.Interface(
    fn=check,
    inputs=gr.Textbox(label="Enter SMS message"),
    outputs=gr.Textbox(label="Result"),
    title="SMS Spam Detector"
).launch()