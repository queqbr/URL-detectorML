import csv

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np

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
    reader = csv.reader(open(filename))
    next(reader)
    urls = [line[0] for line in reader]
    PHP = numOfPHP(urls)
    WWW = numOfWWW(urls)
    HTML = numOfHtml(urls)
    Hyphen = numOfHyphen(urls)
    Question = numOfQuestion(urls)
    Equals = numOfEq(urls)
    Period = numPeriod(urls)
    Amp = numAmp(urls)
    inputs = np.stack([PHP, WWW, HTML, Hyphen, Question, Equals, Period, Amp])

    labels = {
        'benign': 0,
        'defacement': 1,
        'phishing': 2,
        'malware': 3,
    }

    reader = csv.reader(open(filename))
    next(reader)
    target = np.array([labels[line[1]] for line in reader])

    inputs_train, inputs_test, targets_train, targets_test = train_test_split(
        inputs.T, target, test_size=None,
        random_state=0, shuffle=True, stratify=target
    )
    return inputs_train, inputs_test, targets_train, targets_test

def main():
    inputs_train, inputs_test, targets_train, targets_test = get_data()

    classifier = MLPClassifier(random_state=0, verbose=1)

    # Train on all the data AFTER the first 10 (i.e. on 1787 images)
    classifier.fit(inputs_train, targets_train)

    # Test on ONLY the first 10 digits
    # (which coincidentally are themselves the digits 1,2,3,4,5,6,7,8,9 in order)
    results = classifier.predict(inputs_test)

    print(f'Accuracy: {(results == targets_test).mean()}')

if __name__ == '__main__':
    main()
