import xml.etree.ElementTree as ET

import os

from statistics import mode

import re

import tensorflow as tf

import pandas as pd

import pickle

import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import tensorflow_text as text

# load Winston's model

# unpickle Daniil's model

# connect to Justin's KNN

# average, produce output



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





def list_distance(e):

    return e[0]





def parse_attributes(file):

    p_lines = file.readlines()

    ages = {}

    genders = {}

    opes = {}

    cons = {}

    exts = {}

    agrs = {}

    neus = {}

    for line in p_lines[1:]:

        split = line.split(',')

        profile = split[1].strip().strip("\n")



        age = int(float(split[2]))

        gender = int(float(split[3]))

        ope = float(split[4])

        con = float(split[5])

        ext = float(split[6])

        agr = float(split[7])

        neu = float(split[8])



        ages[profile] = age

        genders[profile] = gender

        opes[profile] = ope

        cons[profile] = con

        exts[profile] = ext

        agrs[profile] = agr

        neus[profile] = neu

    return ages, genders, opes, cons, exts, agrs, neus





def parse_relations(file):

    r_lines = file.readlines()

    likes = {}

    for line in r_lines[1:]:

        split = line.split(',')

        profile = split[1].strip().strip("\n")

        like = split[2].strip().strip("\n")

        if profile in likes:

            likes[profile][like] = True

        else:

            likes[profile] = {like: True}

    return likes





def knn_gender(unknown, training_keys, k, train_likes, test_likes, attribute):

    neighbors = []

    for train in training_keys:

        similarity = 0

        for like in test_likes[unknown]:

            if like in train_likes[train]:

                similarity += 1

        test_length = len(test_likes[unknown])

        train_length = len(train_likes[train])

        distance = 1 - (similarity / (test_length + train_length - similarity))

        if len(neighbors) < k:

            neighbors.append([distance, train])

        else:

            neighbors.sort(reverse=True, key=list_distance)

            if distance < neighbors[0][0]:

                neighbors[0] = [distance, train]

    numerator = 0

    denominator = 0

    for neighbor in neighbors:

        distance = neighbor[0] if neighbor[0] > 0 else 0.00001

        neighbor_id = neighbor[1]

        value = attribute[neighbor_id]

        numerator += (1 / distance) * value

        denominator += 1 / distance



    prediction = round(numerator / denominator)



    return prediction





def knn_ocean(unknown, training_keys, k, train_likes, test_likes, attribute):

    neighbors = []

    for train in training_keys:

        similarity = 0

        for like in test_likes[unknown]:

            if like in train_likes[train]:

                similarity += 1

        test_length = len(test_likes[unknown])

        train_length = len(train_likes[train])

        distance = 1 - (similarity / (test_length + train_length - similarity))

        if len(neighbors) < k:

            neighbors.append([distance, train])

        else:

            neighbors.sort(reverse=True, key=list_distance)

            if distance < neighbors[0][0]:

                neighbors[0] = [distance, train]

    numerator = 0

    denominator = 0

    for neighbor in neighbors:

        distance = neighbor[0] if neighbor[0] > 0 else 0.00001

        neighbor_id = neighbor[1]

       

        value = attribute[neighbor_id]

        numerator += (1 / distance) * value

        denominator += 1 / distance



    return str(numerator / denominator)





def load_model(type, file_name):

    input_nodes = {}

    file = open("/home/itadmin/GreatestProjectEverScripts/" + file_name, "r")

    lines = file.readlines()

    for line in lines:

        split = line.split(',')

        if type == "lr":

            split = line.split(',')

            node = split[0].strip().strip("\n")

            weight = split[1].strip().strip("\n")

            input_nodes[node] = float(weight)

        elif type == "sr":

            node = split[0].strip().strip("\n")

            edge_1 = split[1].strip().strip("\n")

            edge_2 = split[2].strip().strip("\n")

            edge_3 = split[3].strip().strip("\n")

            edge_4 = split[4].strip().strip("\n")



            input_nodes[node] = {}

            input_nodes[node]["xx-24"] = float(edge_1)

            input_nodes[node]["25-34"] = float(edge_2)

            input_nodes[node]["35-49"] = float(edge_3)

            input_nodes[node]["50-xx"] = float(edge_4)

    return input_nodes





def make_predictions(test, input_nodes, likes, type):

    if type == "lr":

        input_net = input_nodes['0']

        for like in likes[test]:

            if like in input_nodes:

                input_net += input_nodes[like]

        od = (1 / (1 + np.exp(-input_net))) * 5

        prediction = round(od)

    elif type == "sr":

        age_range = [[0, 24], [25, 34], [35, 49], [50, 200]]

        edge_key = ["xx-24", "25-34", "35-49", "50-xx"]



        outputs = [1, 1, 1, 1]

        for i in range(4):

            input_net = input_nodes['0'][edge_key[i]]

            for like in likes[test]:

                if like in input_nodes:

                    input_net += input_nodes[like][edge_key[i]]

            outputs[i] = 1 / (1 + np.exp(-input_net))

        denominator = sum(outputs)

        for i in range(4):

            if denominator > 0:

                outputs[i] = outputs[i] / sum(outputs)

            else:

                outputs[i] = 0

        maximum = np.argmax(outputs)

        prediction = edge_key[maximum]

    return prediction





################################

# TRAINING

# RELATION TRAINING

#populate the map

os.chdir("..")

os.chdir("/home/itadmin/data/tcss455/training/relation")

training_file_input = open("relation.csv", "r")



os.chdir("..")

os.chdir("/home/itadmin/data/tcss455/training/profile")

profile_training = open("profile.csv", "r")



print("Generating map of training genders")

training_ages, training_genders, training_opes, training_cons, training_exts, training_agrs, training_neus = parse_attributes(profile_training)



print("Generating training relations")

training_likes = parse_relations(training_file_input)



relation_gender_model = load_model("lr", "gender_model.txt")

relation_age_model = load_model("sr", "age_model.txt")



# IMAGE LOADING

os.chdir("..")

os.chdir("/home/itadmin/GreatestProjectEverScripts/ensemble")

print("loading image model")

model_image = tf.keras.models.load_model('image_model')

# TEXT LOADING

model_text = pickle.load(open('logres_gender.pkl', 'rb'))



# LIWC LOADING

os.chdir("..")

os.chdir('/home/itadmin/GreatestProjectEverScripts/text/LIWC')

#model_age = pickle.load(open('test_nn_LIWC.pkl', 'rb'))

model_text_LIWC = tf.keras.models.load_model('LIWC', compile = False)

model_text_LIWC.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model_text_BERT = tf.keras.models.load_model('BERT_gender', compile = False)

model_text_BERT.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

############################################

# LOADING THE TEST DATA

# RELATION

pathFrom = input()

pathTo = input()



pathFromRelation = pathFrom + "/relation"



os.chdir(pathFromRelation)

test_file_likes = open("relation.csv", "r")

print("Generating map of test relations")

test_likes = parse_relations(test_file_likes)



# IMAGE

pathFromProfile = pathFrom + "/profile"

os.chdir(pathFromProfile)

profiles = pd.read_csv("profile.csv")

pathFromImage = pathFrom + "/image/"

# TEXT

pathFromText = pathFrom + "/text"

pathFromLIWC = pathFrom + "/LIWC"

os.chdir(pathFromLIWC)

df_LIWC_gender = pd.read_csv("LIWC.csv")

############################################

#This Lines variable contains all of the testing relations and IDs

pathFromProfile = pathFrom + "/profile"

os.chdir(pathFromProfile)

file_test = open("profile.csv", "r")

Lines = file_test.readlines()

############################################

############################################

#VARIABLE DECLARATION

count = 0

id_text = ""

p_age = 'xx-24',

p_extrovert = '3.5',

p_neurotic = '2.7',

p_agreeable = '3.6',

p_conscientious = '3.4',

p_open = '3.9'

############################################

############################################

ensemble_values = []

gender_group = ['male','female']

img_width = 32

img_height = 32



for line in Lines:

    if count != 0:

        split = line.split(',')

        id_text = split[1]

        # Relation Section

        #Add Relation output to ensemble

        ensemble_values.append(make_predictions(id_text.strip().strip("\n"), relation_gender_model, test_likes, "lr"))

        #Image Section

        os.chdir(pathFromImage)

        image = tf.keras.utils.load_img(pathFromImage + id_text + ".jpg")

        input_arr = tf.keras.utils.img_to_array(image)

        input_arr = np.array([input_arr])  # Convert single image to a batch.

        input_arr = tf.image.resize(input_arr, (img_height, img_width))

        predictions_image = model_image.predict(input_arr)

        val = 0

        if(predictions_image[0][0]>0.5):

            val = 1

        #Add image output to ensemble

        ensemble_values.append(val)

        #LIWC section        

        df_loc = df_LIWC_gender.loc[df_LIWC_gender['userId'] == id_text]

        df_loc_clean  = df_loc.drop("userId", axis='columns')

        val = model_text_LIWC.predict(df_loc_clean)



        ensemble_values.append(round(val[0][0]))



        #Text Section

        os.chdir(pathFromText)

        message = ''

        f = open(os.path.join(os.getcwd(), id_text+".txt"), 'r',encoding='latin-1')	

        message = preprocess(f.read())

        list_message = []

        list_message.append(message)

        df_gender = pd.DataFrame(list_message, columns=['Content'])

        pred_message = model_text.predict(df_gender.iloc[0])

        pred_message1 = model_text_BERT.predict(df_gender.iloc[0])

        ensemble_values.append(pred_message[0])

        ensemble_values.append(round(pred_message1[0][0]))

        list_message = []

        df_gender = df_gender.iloc[0:0]

        user = ET.Element("user", dict(

        id = id_text,

        age_group = make_predictions(id_text.strip().strip("\n"), relation_age_model, test_likes, "sr"),

        gender = gender_group[mode(ensemble_values)],

        extrovert = knn_ocean(id_text, training_likes.keys(), 25, training_likes, test_likes, training_exts),

        neurotic = knn_ocean(id_text, training_likes.keys(), 25, training_likes, test_likes, training_neus),

        agreeable = knn_ocean(id_text, training_likes.keys(), 25, training_likes, test_likes, training_agrs),

        conscientious = knn_ocean(id_text, training_likes.keys(), 25, training_likes, test_likes, training_cons),

        open = knn_ocean(id_text, training_likes.keys(), 25, training_likes, test_likes, training_opes)))



        ET.dump(user)

        ensemble_values = []

        tree = ET.ElementTree(user)

        id_name = id_text + ".xml"

        tree_data = ET.tostring(user)

        

        with open(pathTo + id_name,"wb") as f:

            f.write(tree_data)



        

    count += 1

    completion = count / len(test_likes.keys()) * 100

    

    print("Making predictions: " + "%.2f" % completion + "%")
