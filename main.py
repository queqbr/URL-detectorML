from sklearn.neural_network import MLPClassifier
from sklearn import datasets
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

'''def display_img(img):

    Prints the contents of an input `img` to the command line using 4 different shades of gray

    dark_black = '\u001B[40m  '
    light_black = '\u001B[100m  '
    dark_white = '\u001B[47m  '
    light_white = '\u001B[107m  '
    reset_color = '\u001B[0m'
    percs = np.percentile(np.unique(img), (25, 50, 75))
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            print(
                dark_black if img[r, c] <= percs[0]
                else light_black if img[r, c] <= percs[1]
                else dark_white if img[r, c] <= percs[2]
                else light_white, end=""
            )
        print(reset_color)
    print()
'''
def learn_digits():
    # This just prints information about the dataset. Uncomment this `print` see the description
    # print(digits_set.DESCR)

    # This is the dataset.
    # To get the inputs (pixels of each image) use `digits_set.data`.
    #     This returns (1797, 64) because there are 1797 images, and each one has 64 pixels
    #     (each image is flattened into a 1D array to make them easier to ingest for the neural
    #     network, as opposed to a 2D array which is easier to visualize)
    # To get the intended outputs (the actual number that this is supposed to be a picture of) use `digits_set.target`
    #     This returns a tuple (1797,) because there are 1797 labels (i.e. one for each image)
    digits_set = np.genfromtxt('C:\\Users\\queqb\\Downloads\\archive\\malicious_phish.csv', skip_header=1, delimiter=",", names="data, target", encoding="latin1")
    inputs = digits_set.data
    target = digits_set.target
    #print(f'Shape of input data array:  {inputs.shape}')
    #print(f'Shape of output data array: {target.shape}')
    PHP = numOfPHP(inputs)
    WWW = numOfWWW(inputs)
    HTML = numOfHtml(inputs)
    Hyphen = numOfHyphen(inputs)
    Question = numOfQuestion(inputs)
    Equals = numOfEq(inputs)
    Period = numPeriod(inputs)
    Amp = numAmp(inputs)
    newInputs = np.dstack((PHP, WWW, HTML, Hyphen, Question, Equals, Period, Amp))
    # This is the neural network
    classifier = MLPClassifier(random_state=0)
    print()
    test_size = 10

    # Train on all the data AFTER the first 10 (i.e. on 1787 images)
    classifier.fit(newInputs[test_size:], target[test_size:])

    # Test on ONLY the first 10 digits
    # (which coincidentally are themselves the digits 1,2,3,4,5,6,7,8,9 in order)
    results = classifier.predict(inputs[:test_size])

    # Print to the terminal the results
    count = 0
    for i in range(len(results)):
        print('Neural Net guessed: ' + str(results[i]))
        print('Actual value: ' + str(target[i]))
        if results[i] == target[i]:
            count += 1
        #img = inputs[i].reshape(8, 8)  # reshape to look like an 8x8 image
        #display_img(img)
    print(count)

if __name__ == '__main__':
    learn_digits()