import re
import math
import random
import operator

# Open the file
dataset = open('yelp_labelled.txt').read()

# Lower the case
dataset = dataset.lower()

# Use regular expression to remove punctuation
dataset = re.sub(r'[^\w\s]', '', dataset)

# Split each instance
dataset_lists = dataset.split('\n')
print type(dataset_lists)

# Separate features and labels
features = []
labels = []

for line in dataset_lists:
    label = line[-1:]
    feature = line[:-2]
    labels.append(label)
    features.append(feature)

# Remove the digits
features_set = []
for instance in features:
    temp = ''.join([i for i in instance if not i.isdigit()])
    features_set.append(temp)

# Create the new dataset
new_dataset = []
for i in range(len(features_set)):
    temp = features_set[i] + ' ' + labels[i]
    new_dataset.append(temp)

# Partition
def partition(dataset, k, seed=None):
    dataset_size = math.ceil(len(dataset) / float(k))
    partitions = [[] for i in range(k)]

    for instance in dataset:
        x = instance_assign(partitions, k, dataset_size, seed)
        partitions[x].append(instance)

    return partitions

# Assign instance to each partition
def instance_assign(partitions, k, dataset_size, seed=None):
    if seed is not None:
        random.Random(seed)
    x = random.randint(0, k - 1)
    while len(partitions[x]) >= dataset_size:
        x = random.randint(0, k - 1)
    return x

# Split training set and validation set
def KFold(dataset, k, seed=None):
    global trainingSet
    partitions = partition(dataset, k, seed)

    for i in range(k):
        trainingSet = []

        for j in range(k):
            if j != i:
                trainingSet.append(partitions[j])

        trainingSet = [item for instance in trainingSet for item in instance]
        testSet = partitions[i]

    return trainingSet,testSet

# Document to a vector
def vocab(filename_features, num_vocab):
    di = dict()
    for line in filename_features:
        line = line.rstrip()
        words = line.split()
        for word in words:
            di[word] = di.get(word, 0) + 1
    sorted_di = sorted(di.items(), key=operator.itemgetter(1), reverse=True)
    vocabs = [x[0] for x in sorted_di][11:11 + num_vocab]
    return vocabs
