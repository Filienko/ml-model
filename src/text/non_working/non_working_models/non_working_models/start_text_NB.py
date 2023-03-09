import xml.etree.ElementTree as ET
import os
import pickle
import re
import numpy as np
from gensim.models import Doc2Vec
from sklearn import utils
import gensim
from gensim.models.doc2vec import TaggedDocument
from sklearn.model_selection import train_test_split
import pandas as pd
import tqdm 
############################################
# PICKLE
os.chdir('GreatestProjectEverScripts/text/')
pickled_model_age = pickle.load(open('model_gender_log.pkl', 'rb'))
pickled_model_gender = pickle.load(open('model_gender_log_gender.pkl', 'rb'))

############################################
# DO NOT CHANGE - PATH

# path = "data/tcss455/training/profile"

pathFrom = input()

pathFrom = pathFrom + "/text"

############################################

# DO NOT CHANGE - PATH

pathTo = input()
############################################
#CONSTRUCTING A DATAFRAME
list_contents = []
list_ids = []
os.chdir(pathFrom)

for filename in os.listdir(os.getcwd()):
    list_ids.append(filename[:-4])
    with open(os.path.join(os.getcwd(), filename), 'r',encoding='latin-1') as f: # open in readonly mode
        list_contents.append(f.read())
        
df = pd.DataFrame(list(zip(list_ids, list_contents)),
               columns =['ID', 'Content'])

print("LOADED THE DF")
############################################
#Text preprocessing
def preprocess(message):
    """
    This function takes a string as input, then performs these operations: 
        - lowercase
        - remove URLs
        - remove ticker symbols 
        - removes punctuation
        - removes any single character tokens
    Parameters
    ----------
        message : The text message to be preprocessed
    Returns
    -------
        text: The preprocessed text
    """ 
    # Lowercase the twit message
    text = message.lower()
    # Replace URLs with a space in the message
    text = re.sub('https?:\/\/[a-zA-Z0-9@:%._\/+~#=?&;-]*', ' ', text)
    # Replace ticker symbols with a space. The ticker symbols are any stock symbol that starts with $.
    text = re.sub('\$[a-zA-Z0-9]*', ' ', text)
    # Replace StockTwits usernames with a space. The usernames are any word that starts with @.
    text = re.sub('\@[a-zA-Z0-9]*', ' ', text)
    # Replace everything not a letter or apostrophe with a space
    text = re.sub('[^a-zA-Z\']', ' ', text)
    # Remove single letter words
    text = ' '.join( [w for w in text.split() if len(w)>1] )
    
    return text

df = df.reset_index()

for index, row in df.iterrows():
    message = df['Content'].iloc[index]
    processed_Text = preprocess(message)
    df.loc[index, 'Content'] = processed_Text
############################################
#VARIABLE DECLARATION

############################################

gender_group = ['male','female']
ages_group = ['xx-24', '25-34', '35-49', '50-xx']

gender_output = pickled_model_gender.predict(df['Content'])
age_output = pickled_model_age.predict(df['Content'])

print(gender_output)
for ind in range(df.shape[0]):
	id_name = df['ID'][ind]

	user = ET.Element("user", dict(

	id = id_name,

	age_group = ages_group[age_output[ind]],

	gender = gender_group[gender_output[ind]],
    extrovert = '3.5',
    neurotic = '2.7',
    agreeable = '3.6',
    conscientious = '3.4',
    open = '3.9'))
	ET.dump(user)

	tree = ET.ElementTree(user)

	id_name = id_name + ".xml"

	tree_data = ET.tostring(user)


	with open(pathTo + "/" + id_name,"wb") as f:

		f.write(tree_data)

print("Making predictions: DONE")
