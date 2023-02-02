# date: 01/24/2023
# name: Daniil Filienko
# description: Naive Bayes model for age recognition of social media posters
import random
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

# Reading the data into a dataframe and selecting the columns we need
df = pd.read_csv(r"C:\Users\danfi\VSCode\BERT\age.csv",encoding='latin1')

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

X_train, X_test, y_train, y_test = train_test_split(df['Content'],df['age'],test_size=0.1)

# Training a Naive Bayes model
count_vect = CountVectorizer()
X_train = count_vect.fit_transform(X_train)
clf = MultinomialNB()
clf.fit(X_train, y_train)
# Testing the Naive Bayes model
X_test = count_vect.transform(X_test)
y_predicted = clf.predict(X_test)
# Reporting on classification performance
print("Accuracy: %.2f" % accuracy_score(y_test,y_predicted))