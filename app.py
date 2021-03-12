import streamlit as st
st.title('Sentiment Analysis using Python')
import pandas as pd
df = pd.read_csv('Re_Data.csv')
df = df.iloc[0:11000]
import numpy as np
df = df.replace(np.nan, ' ', regex=True)
import string
df['clean_comment'] = df['clean_comment'].str.replace('[^\w\s]','')
df.clean_comment = df.clean_comment.str.replace('\d+', '')
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem.snowball import PorterStemmer
ps = PorterStemmer()
for i in range(0, len(df)):
  df.clean_comment[i] = ps.stem(df.clean_comment[i]) # for word in df.clean_comment[i]]
x = df['category'].values
y = df['clean_comment'].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 0)
import numpy as np
np.unique(y_train,return_counts=True)
np.unique(y_test,return_counts=True)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
text_model = Pipeline([('tfidf',TfidfVectorizer()),('model',SVC())]) 
text_model.fit(x_train, y_train)
y_pred = text_model.predict(x_test)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
