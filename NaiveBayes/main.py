# Naive Bayes On The Iris Dataset
from csv import reader
from random import seed
from random import randrange
from math import sqrt
from math import exp
from math import pi
import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load a CSV file
def load_csv(filename):
    dataset = pandas.read_csv(filename, sep = ';', header = None)
    df = dataset.iloc[1:]
    # import missingno as msno
    # msno.bar(df, figsize=(8, 6), color='skyblue')
    # plt.show()
    # g = sns.relplot(x=0, y=1, data=dataset, hue=4, style=4)
    # g.fig.set_size_inches(10, 5)
    # plt.show()
    return df


# Split the dataset by class values, returns a dictionary
def separate_by_class(X_train, y_train):
    separated = dict()
    #print("X_train: ", X_train.iloc[1])
    for i in range(len(X_train)):
        vector = X_train.iloc[i]
        #print("Vector: ",vector)
        class_value = y_train.iloc[i]
        #print("class_value: ",class_value)
        if (class_value not in separated):
            separated[class_value] = []
        separated[class_value].append(vector)
    return separated


def prepareTraining(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    #print('X: ', X.shape, ' y: ', y.shape)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #print('X_train: ', X_train.shape, ' y_train: ', y_train.shape)
    #print('X_test: ', X_test.shape, ' y_test: ', y_test.shape)
    return X_train, X_test, y_train, y_test

def naiveBayesAuto(X_train, X_test, y_train, y_test):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import confusion_matrix

    gaussian = GaussianNB()
    gaussian.fit(X_train, y_train)
    Y_pred = gaussian.predict(X_test)
    accuracy_nb = round(accuracy_score(y_test, Y_pred) * 100, 2)
    acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)
    cm = confusion_matrix(y_test, Y_pred)
    accuracy = accuracy_score(y_test, Y_pred)
    precision = precision_score(y_test, Y_pred, average='micro')
    recall = recall_score(y_test, Y_pred, average='micro')
    f1 = f1_score(y_test, Y_pred, average='micro')
    print('Confusion matrix for Naive Bayes\n', cm)
    print('accuracy_Naive Bayes: %.3f' % accuracy)
    print('precision_Naive Bayes: %.3f' % precision)
    print('recall_Naive Bayes: %.3f' % recall)
    print('f1-score_Naive Bayes : %.3f' % f1)


# Calculate the mean of a list of numbers
def mean(numbers):
    return sum(numbers) / float(len(numbers))

def stddev(list):
    # Standard deviation of list
    mean = sum(list) / len(list)
    variance = sum([((x - mean) ** 2) for x in list]) / len(list)
    res = variance ** 0.5
    return res


# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
	summaries = [(mean(column), stddev(column), len(column)) for column in zip(*dataset)]
	del(summaries[-1])
	return summaries


def summarizebyclass(X_train, y_train):
    separated = separate_by_class(X_train, y_train)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries


# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stddev):
    exponent = exp(-((x - mean) ** 2 / (2 * stddev ** 2)))
    return (1 / (sqrt(2 * pi) * stddev)) * exponent


# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])                           # 83+ 128 = 211 rows in X_train
    probabilities = dict()
    #print("Total rows in X_train: ",total_rows)
    #print("Current row: ",row)
    for class_value, class_summaries in summaries.items():
        #print("class_value: ",class_value)
        #print("class_summaries: ", class_summaries)
        probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
        #print("probabilities: ", probabilities)
        for i in range(len(class_summaries)):
            mean, std, count = class_summaries[i]
            probability = calculate_probability(row[i], mean, std)
            #print("Probability: ",probability)
            probabilities[class_value] *=  probability
    return probabilities


# Predict the class for a given row
def predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label

# Naive Bayes Algorithm
def naive_bayes(train, test):
    summarize = summarizebyclass(train)
    predictions = []
    for row in test:
        output = predict(summarize, row)
        predictions.append(output)
    return(predictions)


filename = 'Features.csv'
dataset = load_csv(filename)
X_train, X_test, y_train, y_test = prepareTraining(dataset)
#print(y_train)
Sum = []
std = []
for i in range(4):
    list = X_train[i].tolist()
    Sum.append(sum(list))
    std.append(stddev(list))
# print(Sum)
# print(std)
#separate = separate_by_class(X_train,y_train)
summary = summarizebyclass(X_test,y_test)
print(summary)
correct = 0
for i in range(len(y_test)):
    test = predict(summary, X_test.iloc[i])
    print(i, ". Ergebnis: ",test)
    print(i, ". Soll: ",y_test.iloc[i])
    if test == y_test.iloc[i]:
        correct +=1
print("Accuracy: ", correct/float(len(y_test)) * 100.0)




