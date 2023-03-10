import os
import random
import multiprocessing
import math


def list_distance(e):
    return e[0]


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
    likes = {}
    for line in r_lines[1:]:
        split = line.split(',')
        if split[1] in likes:
            likes[split[1]][split[2]] = True
        else:
            likes[split[1]] = {split[2]: True}
    return likes


def knn(unknown, training_keys, k, likes, attribute):
    neighbors = []
    for train in training_keys:
        similarity = 0
        for like in likes[unknown]:
            if like in likes[train]:
                similarity += 1
        test_length = len(likes[unknown])
        train_length = len(likes[train])
        #similarity *= 15
        #distance = (test_length - similarity) + (train_length - similarity)
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

    return numerator / denominator


def make_predictions(training, test, shared, process, training_attribute, target_attribute):
    print("Making predictions: 0% complete", end="")
    predictions = {}
    count = 1
    for case in test:
        predictions[case] = knn(case, training, 17, training_attribute, target_attribute)
        percent_complete = count / int(len(test)) * 100
        print("\r", end="")
        print("Making predictions: " + "%.2f" % percent_complete + "% complete", end="")
        count += 1
    shared[process] = predictions


def multiprocess_predictions(training, test, training_attribute, target_attribute):
    length = int(len(test))
    split = math.ceil(length / 4)

    manager = multiprocessing.Manager()
    shared = manager.dict()
    p1 = multiprocessing.Process(target=make_predictions, args=(training, test[:split],
                                                                shared, 1, training_attribute, target_attribute,))
    p2 = multiprocessing.Process(target=make_predictions, args=(training, test[split:split * 2],
                                                                shared, 2, training_attribute, target_attribute,))
    p3 = multiprocessing.Process(target=make_predictions, args=(training, test[split * 2: split * 3],
                                                                shared, 3, training_attribute, target_attribute,))
    p4 = multiprocessing.Process(target=make_predictions, args=(training, test[split * 3: length],
                                                                shared, 4, training_attribute, target_attribute,))

    p1.start()
    p2.start()
    p3.start()
    p4.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()

    predictions = {}
    predictions.update(shared[1])
    predictions.update(shared[2])
    predictions.update(shared[3])
    predictions.update(shared[4])

    return predictions


if __name__ == '__main__':
    # Change the directory to where the data is located
    os.chdir(r"")
    print(os.getcwd())

    # pass the file with the profiles and attributes
    profile = open(r"", "r")

    print("Parsing attributes")
    ages, genders, opes = parse_attributes(profile)
    profile.close()

    # pass the file the relations
    relations = open(r"", "r")

    print("Parsing likes")
    likes = parse_relations(relations)
    relations.close()

    keys = list(likes.keys())
    random.shuffle(keys)
    test_index = int(len(keys) * 0.1)

    #gender_predictions = multiprocess_predictions(keys[test_index:], keys[:test_index], likes, gender)
    #age_predictions = multiprocess_predictions(keys[test_index:], keys[:test_index], likes, ages)
    ope_predictions = multiprocess_predictions(keys[test_index:], keys[:test_index], likes, opes)

    print()
    """
    print("Checking gender accuracy")
    correct = 0
    for key in gender_predictions:
        prediction = gender_predictions[key]
        real_value = gender[key]
        if prediction == real_value:
            correct += 1
    accuracy = correct / len(gender_predictions) * 100
    print("Gender Accuracy: " + "%.2f" % accuracy + "%") 

    print("Checking age accuracy")
    correct = 0
    for key in age_predictions:
        prediction = age_predictions[key]
        real_value = ages[key]
        if 0 <= prediction <= 24 and 0 <= real_value <= 24:
            correct += 1
        elif 25 <= prediction <= 34 and 25 <= real_value <= 34:
            correct += 1
        elif 35 <= prediction <= 49 and 35 <= real_value <= 49:
            correct += 1
        elif 50 <= prediction and 50 <= real_value:
            correct += 1
    accuracy = correct / len(age_predictions) * 100
    print("Age Accuracy: " + "%.2f" % accuracy + "%") """

    print("Checking ope accuracy")
    summation = 0
    for key in ope_predictions:
        prediction = ope_predictions[key]
        real_value = opes[key]
        summation += (real_value - prediction) * (real_value - prediction)
    RMSE = math.sqrt(summation / len(ope_predictions))
    print("Ope RMSE: " + "%.2f" % RMSE)

