import math
import os
import random
import numpy as np


def parse_attributes(file):
    p_lines = file.readlines()
    ages = {}
    genders = {}
    opes = {}
    for line in p_lines[1:]:
        split = line.split(',')

        age = int(float(split[2]))
        gender = int(float(split[3]))
        ope = float(split[4])

        ages[split[1]] = age
        genders[split[1]] = gender
        opes[split[1]] = ope
    return ages, genders, opes


def parse_relations(file):
    r_lines = file.readlines()
    user_likes = {}
    like_ids = {}
    for line in r_lines[1:]:
        split = line.split(',')
        profile = split[1].strip().strip("\n")
        like = split[2].strip().strip("\n")
        if split[1] in user_likes:
            user_likes[profile][like] = True
        else:
            user_likes[profile] = {like: True}

        if like not in like_ids:
            like_ids[like] = 0.1
    return user_likes, like_ids


def train_model(profiles, likes, input_nodes, learning_rate, data):
    print("Training model: 0% complete", end="")
    count = 1
    for profile in profiles:
        input_net = input_nodes['0']
        for like in likes[profile]:
            input_net += input_nodes[like]
        #od = 1 / (1 + np.exp(-input_net))          # this line is for a sigmoid unit
        od = input_net if input_net >= 0 else 0     # this line is for a ReLu

        for node in input_nodes.keys():
            old_weight = input_nodes[node]
            td = data[profile]
            xd = 1 if node in likes[profile] or node == "0" else 0
            input_nodes[node] = old_weight + learning_rate * (td - od) * xd
        percent_complete = count / int(len(profiles)) * 100
        print("\r", end="")
        print("Training model: " + "%.2f" % percent_complete + "% complete", end="")
        count += 1
    print('\n')


def load_model():
    input_nodes = {}
    os.chdir(r"")               # change the directory to whichever holds the model
    file = open(r"", "r")       # pass the model's file name
    p_lines = file.readlines()
    for line in p_lines:
        split = line.split(',')
        node = split[0].strip().strip("\n")
        weight = split[1].strip().strip("\n")
        input_nodes[node] = float(weight)
    return input_nodes


def save_model(input_nodes):
    print("Saving model to text file")
    os.chdir(r"")                               # pass the directory to which the model should be saved
    with open(r"", "w") as m:                   # pass the model's file name
        for node in input_nodes.keys():
            m.write(str(node) + "," + str(input_nodes[node]) + "\n")


def make_predictions(tests, input_nodes, likes):
    print("Making predictions: 0% complete", end="")
    predictions = {}
    count = 1
    for profile in tests:
        input_net = input_nodes['0']
        for like in likes[profile]:
            if like in input_nodes:
                input_net += input_nodes[like]
        #od = (1 / (1 + np.exp(-input_net)))         # this line is for a sigmoid unit
        od = input_net if input_net >= 0 else 0      # this line is for a ReLu
        predictions[profile] = od

        percent_complete = count / int(len(tests)) * 100
        print("\r", end="")
        print("Making predictions: " + "%.2f" % percent_complete + "% complete", end="")
        count += 1
    print('\n')
    return predictions


# change the directory to where the training data is located
os.chdir(r"")
print(os.getcwd())

# open the file with the profiles and their attributes
profile = open(r"", "r")

print("Parsing attributes")
ages, genders, opes = parse_attributes(profile)
profile.close()

# open the file with relations
relations = open(r"", "r")

print("Parsing likes")
likes, input_nodes = parse_relations(relations)
relations.close()

if '0' not in input_nodes:
    input_nodes['0'] = 0.1

keys = list(likes.keys())
random.shuffle(keys)
test_index = int(len(keys) * 0.1)

train_model(keys[test_index:], likes, input_nodes, 1, opes)
save_model(input_nodes)
#input_nodes = load_model()
predictions = make_predictions(keys[:test_index], input_nodes, likes)

# print("Checking gender accuracy")
# correct = 0
# for key in predictions:
#     prediction = predictions[key]
#     real_value = genders[key]
#     if prediction == real_value:
#         correct += 1
# accuracy = correct / len(predictions) * 100
# print("Gender Accuracy: " + "%.2f" % accuracy + "%")

print("Checking ope accuracy")
summation = 0
for key in predictions:
    prediction = predictions[key]
    real_value = opes[key]
    summation += (real_value - prediction) * (real_value - prediction)
RMSE = math.sqrt(summation / len(predictions))
print("Ope RMSE: " + "%.2f" % RMSE)
