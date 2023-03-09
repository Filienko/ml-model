import xml.etree.ElementTree as ET
import os
import pickle
import re
import pandas as pd
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
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 

def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    stem_words=[stemmer.stem(w) for w in filtered_words]
    lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    return " ".join(lemma_words)

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
