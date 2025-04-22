import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


nltk.download('punkt_tab')






ps=PorterStemmer()

tfidf=pickle.load(open(r'C:\Users\hamza\Desktop\python\vectorizer.pkl','rb'))
model=pickle.load(open(r'C:\Users\hamza\Desktop\python\model.pkl','rb'))

st.title ('Email_Spam_Classifier')

def transform_text(text):
  text=text.lower()
  text=nltk.word_tokenize(text)
  y=[]
  for i in text:
    if i.isalnum():
      y.append(i)

  text=y[:]
  y.clear()
  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)
  text=y[:]
  y.clear()

  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)

sms=st.text_input('Enter the message here')
if st.button('Predict'):
  #preprocess
  transformed_sms=transform_text(sms)
  # Vectorize
  vector_input=tfidf.transform([transformed_sms])
  ##Predict
  result=model.predict(vector_input)[0]
  #Display
  if result == 1:
    st.header('Spam')
  else:
    st.header('Not_Spam')
