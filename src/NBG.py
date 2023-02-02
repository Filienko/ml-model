# date: 01/24/2023
# name: Daniil Filienko
# description: Naive Bayes model for gender recognition of social media posters
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import logging
import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
import pickle

# Reading the data into a dataframe and selecting the columns we need
df = pd.read_csv(r"big.csv",encoding='latin1')

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text
    
df['Content'] = df['Content'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(df['Content'],df['gender'], stratify=df['gender'],test_size=0.001)

# Training a Naive Bayes model
count_vect = CountVectorizer()
X_train = count_vect.fit_transform(X_train)
clf = MultinomialNB()
clf.fit(X_train, y_train)
# Testing the Naive Bayes model
X_test = count_vect.transform(X_test)
y_predicted = clf.predict(X_test)
print(y_predicted)
pickle.dump(clf, open('model_NB_gender.pkl', 'wb'))
pickle.dump(count_vect, open('model_NB_gender_vectorizer.pkl', 'wb'))

# Reporting on classification performance
print("Accuracy: %.2f" % accuracy_score(y_test,y_predicted))