# ECE471_Final_Project.py
# ECE 471 Dr. Hairong Qi
# written by Noah Caldwell and Austin Day
# 5/6/19
# Final Project

import csv
import numpy as np
from matplotlib import pyplot as plt
import math
import operator
import timeit
from pathlib import Path
import matplotlib.pyplot as plt

# get classifiers from here
from classifiers import MPP_Discriminant as MPP
from classifiers import kNN
from classifiers import decision_tree
from classifiers import NN
from classifiers import combined_classifier

# PCA
from pca import pca, pca_error, yeet

redwine_data_location = Path(__file__).absolute().parent / 'winequality-red.csv'
whitewine_data_location = Path(__file__).absolute().parent / 'winequality-white.csv'

np.set_printoptions(linewidth=200,precision=6)

def getData(datafile):
    with open(datafile, "r") as csvIn: # datafile should be string with .csv at end
        aReader = csv.reader(csvIn, delimiter=";", quotechar='"')
        next(aReader)
        dataIn = []
        for row in aReader:
            thisRow = []
            for cell in row:
                thisRow.append(float(cell))
            dataIn.append(thisRow)
    return np.array(dataIn)

def normalize(dataset): 
    d = len(dataset[0])
    dataset = np.array(dataset).astype(np.float) # make it a numpy array
    dataset = dataset.T
    for i in range(0, d-1): # Keep class the same
        mean = dataset[i].mean(0)
        stdev = np.std(dataset[i],0)
        for j in range(len(dataset[i])):
            dataset[i][j] = (dataset[i][j] - mean) / stdev
    return dataset.T

    # Potential issue:
    # I'm dividing the file into m equal parts but it isn't random.
    # The first part is the first 1/m part (or the first 10%) of the file,
    # second is second 10%, etc. so it's not truly random and I have no guarantee
    # that each subset contains a variety of labelled data. One subset could be
    # all mediocre quality wines, and that's no good for classification
def data_fold(m, data):
    #####
    grps = []
    each = math.floor((len(data) - 1) / m)
    for i in range(0,m):
        mrow = []
        for i in range(i*each, (i+1)*each):
            mrow.append(i)
        grps.append(mrow)
    #####
    # above is all that I changed from your code in the last project.
    # Just changed it to do as I said above, to take the first 10%,
    # second 10%, etc.
    fd = []
    for i in range(len(grps)):
        gp = []
        for g in grps[i]:
            gp.append(data[g-1])
        fd.append(np.array(gp))

    folded_data = np.array(fd)
    return folded_data

def cross_validation(classifier, params, folded_data):
    folds = len(folded_data)
    results = []
    for i in range(folds):
        testingdata = folded_data[i]
        # set the training data to the conc of all the other folds
        if (i == 0):
            trainingdata = np.concatenate(folded_data[i+1:folds])
        elif (i+1 == folds):
            trainingdata = np.concatenate(folded_data[0:i])
        else:
            trainingdata = np.concatenate((np.concatenate(folded_data[0:i]), np.concatenate(folded_data[i+1:folds])))
        results.append(obtain_accuracy(classifier, params, trainingdata, testingdata))
    return results

def obtain_accuracy(classifier, params, trainingdata, testingdata):
    guesses = classifier(trainingdata, testingdata, *params)
    actual = testingdata.T[-1]
    correct = 0
    for i in range(len(actual)):
        #if (np.abs(guesses[i]-actual[i]) <= 1): # for within 1 quality rung accuracy
        if (guesses[i] == actual[i]):  # for correct classification only
            correct += 1
    accuracy = float(correct/len(actual))
    return accuracy

def graph(averages, ran, title, stdev=0, init_range = 1):
    plt.xticks(range(init_range,ran+1))
    plt.scatter(range(init_range, ran+1), averages)
    plt.errorbar(range(init_range, ran+1), averages, yerr=stdev)
    plt.title(title)
    plt.show()
    input()
    plt.clf()

def test_classifiers(dimensions=None):
    redwine_data = normalize(getData(redwine_data_location))
    whitewine_data = normalize(getData(whitewine_data_location))
    
    # Dimensionality reduction with PCA
    # Printing error margins for dimensions
    # Reducing to input amount of dimensions
    redwine_data = yeet(redwine_data, [7,8,3,2,4])
    whitewine_data = yeet(whitewine_data, [2, 9, 8, 5,6])
 
    pca_error(redwine_data)
    pca_error(whitewine_data)
    if dimensions is not None: 
        redwine_data = pca(redwine_data, dimensions)
        whitewine_data = pca(whitewine_data, dimensions)
    
    red_folded_data = data_fold(10, redwine_data)
    white_folded_data = data_fold(10, whitewine_data)

    ###################################
    # Bayesian descriminant functions #
    ###################################
    # There's something terribly wrong with my case 2 and case 3 implementations where it keeps classifying things totally wrong. Will look into it more later, it's giving me a headache.
    
    """
    print("MPP discriminant function 10-fold cross validation:")
    for i in range(0,2): # first loop, red. second loop, white. will need stdev later for plotting
        if i == 0:
            folded_data = red_folded_data
            wine = "red wine"
        if i == 1:
            folded_data = white_folded_data
            wine = "white wine"
        results_matrix = []
        for case in range(1,4):
            case_averages = cross_validation(MPP, [case], folded_data)
            case_accuracy = np.mean(case_averages)
            print("Average accuracy on {} dataset using case {} = {}%".format(wine, case, case_accuracy*100))
        
        
        # print(results_matrix)
    """
    
    ##################
    #      kNN       #
    ##################
    """
    krange = 50
    print("k-Nearest-Neighbors approach with 10-fold cross validation:")
    for i in range(0,2): # first loop, red. second loop, white. will need stdev later for plotting
        if i == 0:
            folded_data = red_folded_data
            wine = "red wine"
        if i == 1:
            folded_data = white_folded_data
            wine = "white wine"
        results_matrix = []
        for k in range(1,krange+1):
            results_matrix.append(cross_validation(kNN, [k], folded_data))
        results_matrix = np.array(results_matrix)
        kNN_averages = np.mean(results_matrix, axis=1)
        stdev = np.std(results_matrix, axis=1)
        max_accuracy = max(kNN_averages)
        maxk = list(kNN_averages).index(max_accuracy) + 1
        print("Best accuracy on {} dataset is found at k = {}: {}%".format(wine, maxk, max_accuracy*100))
        # print(results_matrix)
        graph(kNN_averages, krange, 'average accuracy vs k', stdev=stdev)
    """

    ##################
    #      DT        #
    ##################
    # Best accuracy on red wine dataset is found at max depth = 5: 57.42138364779874%
    # Best accuracy on white wine dataset is found at max depth = 5: 51.679506933744214%
    """
    print("Decision tree approach with 10-fold cross validation:")
    for i in range(0,2):
        if i == 0:
            folded_data = red_folded_data
            wine = "red wine"
        if i == 1:
            folded_data = white_folded_data
            wine = "white wine"
        results_matrix = []
        sqrtn = int(np.sqrt(len(folded_data[0])*len(folded_data))) + 1
        for depth in range(1, sqrtn):
            results_matrix.append(cross_validation(decision_tree, [depth], folded_data))
        results_matrix = np.array(results_matrix)
        dt_averages = np.mean(results_matrix, axis=1)
        stdev = np.std(results_matrix, axis=1)
        max_accuracy = max(dt_averages)
        bestdepth = list(dt_averages).index(max_accuracy) + 1
        print("Best accuracy on {} dataset is found at max depth = {}: {}%".format(wine, bestdepth, max_accuracy*100))
        # print(results_matrix)
    
    ##################
    #      NN        #
    ##################
    # Best accuracy on red wine dataset is found with number hidden nodes = 11: 0.6018867924528302
    # Total time elapsed = 243.79368224800055 seconds
    # Best accuracy on white wine dataset is found with number hidden nodes = 11: 0.5453004622496149
    # Total time elapsed = 474.58355869500156 seconds
    hidden_node_range = 12
    print("Neural Network approach with m-fold cross validation:")
    for i in range(0,2):
        if i == 0:
            folded_data = red_folded_data
            wine = "red wine"
        if i == 1:
            folded_data = white_folded_data
            wine = "white wine"
        

        start = timeit.default_timer()
        results_matrix = []
        for k in range(5, hidden_node_range+1):
            print("Trying with {} hidden layer(s) of size {}".format(1, k))
            res = cross_validation(NN, [k, 1], folded_data)
            print("Accuracy: {}%".format(np.average(np.array(res))*100))
            results_matrix.append(res)

        results_matrix = np.array(results_matrix)
        NN_averages = np.mean(results_matrix, axis=1)
        stdev = np.std(results_matrix, axis=1)
        max_accuracy = max(NN_averages)
        maxk = list(NN_averages).index(max_accuracy) + 2
        print("Best accuracy on {} dataset is found with hidden layer size = {}: {}%".format(wine, maxk, max_accuracy*100))
        
        stop = timeit.default_timer()
        print("Total time elapsed = {} seconds".format(stop-start))
        graph(kNN_averages, hidden_node_range, 'average accuracy vs number of hidden layers', stdev=stdev, init_range=5)
    """

def run_combined_classifier():
    redwine_data = normalize(getData(redwine_data_location))
    whitewine_data = normalize(getData(whitewine_data_location))
    
    # Dimensionality reduction with PCA
    # Printing error margins for dimensions
    # Reducing to input amount of dimensions
    redwine_data = yeet(redwine_data, [7,8,3,2,4])
    whitewine_data = yeet(whitewine_data, [2, 9, 8, 5,6])
    #pca_error(redwine_data)
    #pca_error(whitewine_data)
    
    redwine_data = pca(redwine_data, 5)
    whitewine_data = pca(whitewine_data, 5)

    
    red_folded_data = data_fold(10, redwine_data)
    white_folded_data = data_fold(10, whitewine_data)

    # Configuration
    redweights = [0.3, 1, 0.3, 1.5, 1.5, 2]
    whiteweights = [0.2, 0.7, 0.1, 1.2, 1, 2]

    # Begin timing process
    start = timeit.default_timer()

    averages_ = cross_validation(combined_classifier, ["red", redweights], red_folded_data)
    accuracy_ = np.mean(averages_)
    print("Average accuracy on red wine dataset: {}%".format(accuracy_*100))

    stop = timeit.default_timer()
    print("Total time elapsed = {} seconds".format(stop-start))

    # Begin timing process - white wine
    start = timeit.default_timer()

    averages_ = cross_validation(combined_classifier, ["white", whiteweights], white_folded_data)
    accuracy_ = np.mean(averages_)
    print("Average accuracy on white wine dataset: {}%".format(accuracy_*100))

    stop = timeit.default_timer()
    print("Total time elapsed = {} seconds".format(stop-start))

    #graph(averages_, stdev=0, )
    

if __name__ == '__main__':
    run_combined_classifier()
    #test_classifiers(5)
    
