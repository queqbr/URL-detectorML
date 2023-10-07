import csv
import inspect
import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

labels = {
    'benign': 0,
    'defacement': 1,
    'phishing': 2,
    'malware': 3,
}

def format_label_names(d):
    """
    Concatenate keys with the same value into a single string, demarcated by newline characters.

    :param d: Dictionary with multiple keys mapping to the same value.
    :type d: dict
    :return: List of strings where keys mapping to the same value are concatenated.
    :rtype: list
    """
    # Create a reverse mapping of values to lists of keys
    reversed_dict = {}
    for key, value in d.items():
        if value not in reversed_dict:
            reversed_dict[value] = []
        reversed_dict[value].append(key)

    # Create a list of concatenated keys
    concatenated_keys_list = []
    for _, v in sorted(reversed_dict.items()):
        concatenated_keys_list.append("\n".join(v))

    return concatenated_keys_list

def display_accuracy(target, predictions, labels, title):
    cm = confusion_matrix(target, predictions)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots()
    cm_display.plot(ax=ax)
    ax.set_title(title)
    plt.show()

class Features:

    def urlLen(inputs):
        urlLen = np.zeros(len(inputs), dtype="int32")
        for i in range(len(inputs)):
            urlLen[i] = len(inputs[i])
        return urlLen

    def http(inputs):
        http = np.zeros(len(inputs), dtype="int32")
        for i in range(len(inputs)):
            http[i] = inputs[i].count("http")
        return http

    def closeChars(inputs):
        close = np.zeros(len(inputs), dtype="int32")
        for i in range(len(inputs)):
            close[i] = inputs[i].isascii()
        return close

    def numOfNums(inputs):
        nums = np.zeros(len(inputs), dtype="int32")
        for i in range(len(inputs)):
            nums[i] = len(re.sub("[^0-9]", "", inputs[i]))
        return nums

    def numPercent(inputs):
        perc = np.zeros(len(inputs), dtype="int32")
        for i in range(len(inputs)):
            perc[i] = inputs[i].count("%")
        return perc

    def numOfPHP(inputs):
        numPHP = np.zeros(len(inputs), dtype="int32")
        for i in range(len(inputs)):
            numPHP[i] = inputs[i].count("php")
        return numPHP

    def numOfWWW(inputs):
        numWWW = np.zeros(len(inputs), dtype="int32")
        for i in range(len(inputs)):
            numWWW[i] = inputs[i].count("www")
        return numWWW

    def numOfHtml(inputs):
        numHtml = np.zeros(len(inputs), dtype="int32")
        for i in range(len(inputs)):
            numHtml[i] = inputs[i].count("html")
        return numHtml

    def numOfHyphen(inputs):
        numHy = np.zeros(len(inputs), dtype="int32")
        for i in range(len(inputs)):
            numHy[i] = inputs[i].count("-")
        return numHy

    def numOfQuestion(inputs):
        numQ = np.zeros(len(inputs), dtype="int32")
        for i in range(len(inputs)):
            numQ[i] = inputs[i].count("?")
        return numQ

    def numOfEq(inputs):
        numE = np.zeros(len(inputs), dtype="int32")
        for i in range(len(inputs)):
            numE[i] = inputs[i].count("=")
        return numE

    def numPeriod(inputs):
        numP = np.zeros(len(inputs), dtype="int32")
        for i in range(len(inputs)):
            numP[i] = inputs[i].count(".")
        return numP

    def numAmp(inputs):
        numA = np.zeros(len(inputs), dtype="int32")
        for i in range(len(inputs)):
            numA[i] = inputs[i].count("&")
        return numA

def get_data():
    filename = 'malicious_phish.csv'
    reader = csv.reader(open(filename, encoding="latin1"))
    next(reader)
    urls = [line[0] for line in reader]
    inputs = np.stack([
        f(urls) for _, f in tqdm(inspect.getmembers(
            Features, predicate=inspect.isfunction
        ), desc='Extracting features from texts')
    ], axis=1)

    reader = csv.reader(open(filename, encoding="latin1"))
    next(reader)
    target = np.array([labels[line[1]] for line in reader])

    inputs_train, inputs_test, targets_train, targets_test = train_test_split(
        inputs, target, test_size=None,
        random_state=0, shuffle=True, stratify=target
    )
    return inputs_train, inputs_test, targets_train, targets_test

def main():
    model_load_path = Path('model.pickle')
    inputs_train, inputs_test, targets_train, targets_test = get_data()
    if model_load_path.exists():
        classifier = pickle.load('model.pickle')
    else:
        classifier = RandomForestClassifier(random_state=0, verbose=1, n_estimators=100)
        classifier.fit(inputs_train, targets_train)
        pickle.dump(classifier, model_load_path.open(mode='wb'))

    results = classifier.predict(inputs_test)

    feat_names = sorted([t[0] for t in inspect.getmembers(Features, predicate=inspect.isfunction)])
    feats = sorted(zip(classifier.feature_importances_, feat_names))[::-1]
    print('Feature importances:')
    print('\n'.join([f'{t[1]:15s} {t[0]:.4f}' for t in feats]))
    display_accuracy(targets_test, results, format_label_names(labels), "Malicious URLs")
    print(f'Accuracy: {(results == targets_test).mean()}')

if __name__ == '__main__':
    main()
