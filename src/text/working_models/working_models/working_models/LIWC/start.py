import xml.etree.ElementTree as ET
import os
import pickle
import re
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

############################################
# PICKLE
os.chdir('GreatestProjectEverScripts/text/LIWC')

#model_age = pickle.load(open('test_nn_LIWC.pkl', 'rb'))
model_gender = tf.keras.models.load_model('LIWC', compile = False)
model_gender.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
############################################
# DO NOT CHANGE - PATH

# path = "data/tcss455/training/profile"

pathFrom = input()

pathFrom = pathFrom + "/LIWC"
os.chdir(pathFrom)
############################################

# DO NOT CHANGE - PATH

pathTo = input()
############################################
#CONSTRUCTING A DATAFRAME

df = pd.read_csv('LIWC.csv',delimiter=",")
train_df = df.drop("userId", axis='columns')
print("LOADED THE DF")
############################################
#VARIABLE DECLARATION

############################################
gender_group = ['male','female']
ages_group = ['xx-24', '25-34', '35-49', '50-xx']
y_pred = model_gender.predict(train_df)
y_pred_norm = np.around(y_pred)
y_pred_norm = np.asarray(y_pred_norm, dtype = 'int')


for ind in range(df.shape[0]):
	id_name = df['userId'][ind]

	gender_output = str(y_pred_norm[ind][0])
	user = ET.Element("user", dict(

	id = id_name,

	age_group = ages_group[0],

	gender = gender_output,
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
