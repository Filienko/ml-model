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
        profile = split[1].strip().strip("\n")

        age = int(float(split[2]))
        gender = int(float(split[3]))
        ope = float(split[4])

        ages[profile] = age
        genders[profile] = gender
        opes[profile] = ope
    return ages, genders, opes


def parse_relations(file):
    r_lines = file.readlines()
    user_likes = {}
    input_nodes = {}

    if '0' not in input_nodes:
        input_nodes['0'] = {}
        input_nodes['0']["xx-24"] = 0.1
        input_nodes['0']["25-34"] = 0.1
        input_nodes['0']["35-49"] = 0.1
        input_nodes['0']["50-xx"] = 0.1

    for line in r_lines[1:]:
        split = line.split(',')
        profile = split[1].strip().strip("\n")
        like = split[2].strip().strip("\n")
        if profile in user_likes:
            user_likes[profile][like] = True
        else:
            user_likes[profile] = {like: True}

        if like not in input_nodes:
            input_nodes[like] = {}
            input_nodes[like]["xx-24"] = 0.1
            input_nodes[like]["25-34"] = 0.1
            input_nodes[like]["35-49"] = 0.1
            input_nodes[like]["50-xx"] = 0.1
    return user_likes, input_nodes


def train_model(profiles, likes, input_nodes, learning_rate, data):
    print("Training model: 0% complete", end="")
    count = 1
    age_range = [[0, 24], [25, 34], [35, 49], [50, 200]]
    edge_key = ["xx-24", "25-34", "35-49", "50-xx"]

    for i in range(4):
        for profile in profiles:
            edge = edge_key[i]
            input_net = input_nodes['0'][edge]
            for like in likes[profile]:
                input_net += input_nodes[like][edge_key[i]]
            od = 1 / (1 + np.exp(-input_net))

            for node in input_nodes.keys():
                old_weight = input_nodes[node][edge_key[i]]
                td = 1 if age_range[i][0] <= data[profile] <= age_range[i][1] else 0
                xd = 1 if node in likes[profile] or node == "0" else 0
                input_nodes[node][edge] = old_weight + learning_rate * (td - od) * xd
            percent_complete = count / int(len(profiles)) * 25
            print("\r", end="")
            print("Training model: " + "%.2f" % percent_complete + "% complete", end="")
            count += 1
    print('\n')


def load_model():
    input_nodes = {}
    os.chdir(r"")                       # Change the directory to where the model is located
    file = open(r"", "r")               # Pass the name of the model text file
    p_lines = file.readlines()
    for line in p_lines:
        split = line.split(',')
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


def save_model(input_nodes):
    print("Saving model to text file")
    os.chdir(r"")                           # Pass the directory to which the model is to be saved
    with open(r"", "w") as m:               # Pass the name of the file to which the model is to be saved
        for node in input_nodes.keys():
            m.write(str(node) + "," + str(input_nodes[node]["xx-24"])
                    + "," + str(input_nodes[node]["25-34"])
                    + "," + str(input_nodes[node]["35-49"])
                    + "," + str(input_nodes[node]["50-xx"])
                    + "\n")


def make_predictions(tests, input_nodes, likes):
    print("Making predictions: 0% complete", end="")
    predictions = {}
    count = 1
    age_range = [[0, 24], [25, 34], [35, 49], [50, 200]]
    edge_key = ["xx-24", "25-34", "35-49", "50-xx"]

    outputs = [1, 1, 1, 1]

    for profile in tests:
        for i in range(4):
            input_net = input_nodes['0'][edge_key[i]]
            for like in likes[profile]:
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
        predictions[profile] = age_range[maximum]

        percent_complete = count / int(len(tests)) * 100
        print("\r", end="")
        print("Making predictions: " + "%.2f" % percent_complete + "% complete", end="")
        count += 1
    print('\n')
    return predictions


# Change the directory to where the data is located
os.chdir(r"")           #
print(os.getcwd())

# Pass the name of the file with the profiles and attributes
profile = open(r"", "r")

print("Parsing attributes")
ages, genders, opes = parse_attributes(profile)
profile.close()

# Pass the name of the file with the relations
relations = open(r"", "r")

print("Parsing likes")
likes, input_nodes = parse_relations(relations)
relations.close()

if '0' not in input_nodes:
    input_nodes['0'] = {}
    input_nodes['0']["xx-24"] = 0.1
    input_nodes['0']["25-34"] = 0.1
    input_nodes['0']["35-49"] = 0.1
    input_nodes['0']["50-xx"] = 0.1

keys = list(likes.keys())
random.shuffle(keys)
test_index = int(len(keys) * 0.1)

train_model(keys, likes, input_nodes, 0.3, ages)
save_model(input_nodes)
# input_nodes = load_model()
predictions = make_predictions(keys[:test_index], input_nodes, likes)

print("Checking age accuracy")
correct = 0
for key in predictions:
    prediction = predictions[key]
    real_value = ages[key]
    if predictions[key][0] <= real_value <= predictions[key][1]:
        correct += 1
accuracy = correct / len(predictions) * 100
print("Gender Accuracy: " + "%.2f" % accuracy + "%")


