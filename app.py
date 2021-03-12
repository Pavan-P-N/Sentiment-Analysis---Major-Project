import streamlit as st
st.title('Sentiment Analysis')
import pandas as pd
import emoji
df = pd.read_csv('Re_Data.csv')
df =df.iloc[0:5000]
df.dropna(inplace=True)
import string
df['clean_comment'] = df['clean_comment'].str.replace('[^\w\s]','')
df.clean_comment = df.clean_comment.str.replace('\d+','')
x = df.iloc[:,0]
y = df.iloc[:,1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 0)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
text_model = Pipeline([('tfidf',TfidfVectorizer()),('model',SVC())]) 
text_model.fit(x_train, y_train)
select = st.text_input('Enter your message')
op = text_model.predict([select])
if op[0] == "Positive":
  st.title(op)
  st.title(emoji.emojize(':smile:'))
elif op[0]=="Neutral":
  st.title(op)
  st.title(emoji.emojize(':expressionless:'))
elif op[0]=="Negative":
  st.title(op)
  st.title(emoji.emojize(':worried:'))

