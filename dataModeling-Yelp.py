from __future__ import division
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
import numpy
import string
import csv
import operator
import math
import matplotlib.pyplot as plt


##############################################################################
#
#                           Data Modeling
#                               
##############################################################################
#find_features('yelp_features_2.txt','yelp_ features_1800.csv',vocab('yelp_features_2.txt', 1800))


# this function calculates probability
# parameters: a dataset, smoothing parameter, number of vocabularies
def occurrences(dataset, m, vocab):
    no_of_examples = len(dataset)
    prob = dict(Counter(dataset))
    for key in prob.keys():
        prob[key] = math.log(prob[key] + m) / (float(no_of_examples) + m*vocab)
    return prob

# this function 
def naive_bayes_train(x_train, y_train, m, vocab):
    classes = np.unique(y_train)
    rows, cols = np.shape(x_train)
    likelihoods = {}
    for cls in classes:
        likelihoods[cls] = defaultdict(list)

    for cls in classes:
        row_indices = np.where(y_train == cls)[0]
        subset = x_train[row_indices, :]
        r, c = np.shape(subset)
        for j in range(0, c):
            likelihoods[cls][j] += list(subset[:, j])

    for cls in classes:
        for j in range(0, cols):
            likelihoods[cls][j] = occurrences(likelihoods[cls][j], m, vocab)
    return likelihoods
#test(train(x_train, y_train, m, vocab), y_train, x_test, m, vocab)

def naive_bayes_test(likelihoods, y_train, x_test, m, vocab):
    classes = np.unique(y_train)
    class_probabilities = occurrences(y_train, m, vocab)

    results = {}
    final = []

    for item in x_test:
        for cls in classes:
            class_probability = class_probabilities[cls]
            for i in range(0, len(x_test[0])):
                relative_values = likelihoods[cls][i]
                if item[i] in relative_values.keys():
                    class_probability += relative_values[item[i]]
                else:
                    class_probability += 0
                results[cls] = class_probability
        if results[0] > results[1]:
            final.append(0)
        else:
            final.append(1)
    return final

#this function evaluates the classifier by providing error rate
#parameters: Y testing set, Y prediction set
def evaluate(y_test, y_pred):
    y_diff = [x1 - x2 for (x1, x2) in zip(y_test, y_pred)]
    acc = y_diff.count(0)
    accuracy_rate = acc / 100
    error_rate = 1 - accuracy_rate
    return error_rate

#this function does stratified cross validation
#parameters: iteration times, the proportion of test set and train set, random seed
def cross_validation(X, y, m, vocab):
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
    errorrates = []
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        likelihood = naive_bayes_train(X_train, y_train, m, vocab)
        y_pred = naive_bayes_test(likelihood, y_train, X_test, m, vocab)
        errorrate = evaluate(y_test, y_pred)
        errorrates.append(errorrate)
    
    mean = sum(errorrates) / 10.0
    print ('The error rate of my classifier is :' + str(mean))

# Using pandas to preprocess the .csv files
x_train = pd.read_csv('yelp_ features_1800.csv', header=None)
y_train = pd.read_csv('yelp_labels.csv', header=None)

# Combines the features and labels
x_train['Y'] = y_train
trainingset = x_train
train = trainingset.values

# Data structure transformation: from DataFrame to list
dataset_list = trainingset.values.tolist()

if __name__ == "__main__":
    X = train[0::, :-1]
    y = train[0::, -1]

    cross_validation(X, y,0.1,1800)

    
plt.scatter([0.1,0.5,2.5,12.5],[0.235,0.225,0.219,0.233])
plt.xlabel('Smoothing Factor')
plt.ylabel('Corss Validation Performance')
    
    
plt.scatter([125,250,500,1000,1800],[0.318,0.257,0.213,0.214,0.219])
plt.xlabel('Vocabulary Size')
plt.ylabel('Corss Validation Performance')

