import csv
import inspect
import re
from pathlib import Path
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch

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
    print('Plotting confusion matrix...')
    cm = confusion_matrix(target, predictions)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots()
    cm_display.plot(ax=ax)
    ax.set_title(title)
    plt.show()

def display_feature_importances(classifier, inputs_train, targets_train, idx):
    feat_names = np.array([t[0] for t in inspect.getmembers(Features, predicate=inspect.isfunction)])
    # Indices of most important features
    print('Feature importances:')
    print('\n'.join([f'{t[1]:15s} {t[0]:.4f}' for t in zip(classifier.feature_importances_[idx], feat_names[idx])]))
    print('Computing Permutation Importances...')
    permImportance = permutation_importance(
        classifier, inputs_train,
        targets_train, n_repeats=1
    )
    featureImpMethods = {
        'Permutation Importances': permImportance.importances_mean[idx],
        'RF Importances': classifier.feature_importances_[idx],
    }
    x = np.arange(len(feat_names))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    print('Plotting feature importances...')
    fig, ax = plt.subplots(layout='constrained')
    for attribute, measurement in featureImpMethods.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('relative importance')
    ax.set_xticks(x + width, feat_names[idx])
    ax.set_ylim(0, 1)
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

def plot_performance_curve(inputs_train, inputs_test, targets_train, targets_test, idx):

    accuracies = []
    for i in range(1, len(idx)):
        classifier = RandomForestClassifier(random_state=0)
        classifier.fit(inputs_train[:, idx[:i]], targets_train)
        accuracies.append((
            targets_test == classifier.predict(inputs_test[:, idx[:i]])
        ).mean())
    plt.plot(range(1, len(accuracies) + 1), accuracies)
    plt.xlabel('Number of most important features to include')
    plt.ylabel('Accuracy')
    plt.show()

def use_handmade_features():
    model_load_path = Path('model.pickle')
    inputs_train, inputs_test, targets_train, targets_test = get_data()
    hyperparams = {'random_state': 0, 'n_estimators': 100}
    if model_load_path.exists():
        with open('model.pickle', 'rb') as pickle_file:
            classifier = pickle.load(pickle_file)
    else:
        classifier = RandomForestClassifier(**hyperparams)
        print(f'Training {classifier.__class__.__name__}...')
        classifier.fit(inputs_train, targets_train)
        pickle.dump(classifier, model_load_path.open(mode='wb'))

    results = classifier.predict(inputs_test)
    idx = np.argsort(classifier.feature_importances_)[::-1]
    plot_performance_curve(inputs_train, inputs_test, targets_train, targets_test, idx)

    display_accuracy(targets_test, results, format_label_names(labels), "Malicious URLs")
    # display_feature_importances(classifier, inputs_train, targets_train, idx)
    print(f'Test Accuracy: {(results == targets_test).mean() * 100:.4f}%')

def use_bert_features():
    filename = 'malicious_phish.csv'
    reader = csv.reader(open(filename, encoding="latin1"))
    next(reader)
    urls = [line[0] for line in reader]
    reader = csv.reader(open(filename, encoding="latin1"))
    next(reader)
    targets = np.array([labels[line[1]] for line in reader])

    bert_data_path = Path('bert_data.npy')

    if bert_data_path.exists():
        arr_file = np.load(bert_data_path)
        inputs = arr_file['inputs']
        targets = arr_file['targets']
    else:
        inputs = []
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        # `inference_mode` should be used to wrap any use of the model when we are
        # not training the model to ensure is no memory leak
        with torch.inference_mode():
            model = AutoModel.from_pretrained("bert-base-cased")
            print('Tokenizing URLs...')
            tokenized = tokenizer(urls, return_tensors='pt', padding=True, truncation=True)
            print('Encoding URLs...')
            inputs = model(**tokenized).pooler_output
        inputs = inputs.detach().numpy()
        np.savez_compressed(bert_data_path, inputs=inputs, targets=targets)

    bert_inputs_train, bert_inputs_test, bert_targets_train, bert_targets_test = train_test_split(
        inputs, targets, test_size=None,
        random_state=0, shuffle=True, stratify=targets
    )
    model_load_path = Path('bert.pickle')
    hyperparams = {'random_state': 0}
    if model_load_path.exists():
        with open('bert.pickle', 'rb') as pickle_file:
            classifier = pickle.load(pickle_file)
    else:
        classifier = MLPClassifier(**hyperparams)
        print(f'Training {classifier.__class__.__name__}...')
        classifier.fit(bert_inputs_train, bert_targets_train)
        pickle.dump(classifier, model_load_path.open(mode='wb'))

    results = classifier.predict(bert_inputs_test)
    display_accuracy(bert_targets_test, results, format_label_names(labels), "Malicious URLs")
    print(f'Test Accuracy: {(results == bert_targets_test).mean() * 100:.4f}%')

def main():
    # use_handmade_features()
    use_bert_features()

if __name__ == '__main__':
    main()
