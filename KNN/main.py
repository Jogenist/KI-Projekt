# k nearest neighbor classifier
from math import sqrt
import pandas
from random import randrange


# Load a CSV file
def load_csv(filename):
    dataset = pandas.read_csv(filename, sep = ';', header = None)
    df = dataset.iloc[1:]
    return df

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)

# split the dataset in train and test
def prepareTraining(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    #print('X: ', X.shape, ' y: ', y.shape)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Locate the most similar neighbors
def get_neighbors(X_train, test_row, num_neighbors):
    distances = list()
    for i in range(len(X_train)-1):
        dist = euclidean_distance(test_row, X_train.iloc[i])
        distances.append((X_train.iloc[i], dist))
    distances.sort(key=lambda tup: tup[1])
    #print(distances)
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

# return class of located neighbors
def get_neighbors_class(neighbors):
    #print(neighbors)
    neighbors_class = list()
    for row in neighbors:
        neighbors_class.append(row[4].tolist())
    return neighbors_class

# find most representated value in list
def most_frequent(List):
    counter = 0
    num = List[0]
    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i
    return num

# Make a classification prediction with neighbors
def predict_classification(X_train, test_row, num_neighbors):
    neighbors = get_neighbors(X_train, test_row, num_neighbors)
    neighbors_class = get_neighbors_class(neighbors)
    #print(neighbors_class)
    prediction = most_frequent(neighbors_class)
    return prediction

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = dataset[1].tolist()
    print("Datasetcopy: ",dataset_copy)
    fold_size = int(len(dataset) / n_folds)
    print(fold_size)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors):
    predictions = list()
    for row in range(len(test)):
        #print("Row: ",test.iloc[row])
        output = predict_classification(train, test.iloc[row], num_neighbors)
        predictions.append(output)
    return(predictions)

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


filename = 'Features.csv'
dataset = load_csv(filename)                                                                                            #load features
X_train, X_test, y_train, y_test = prepareTraining(dataset)                                                             #split dataset in train and test
X_train[4] = y_train                                                                                                    #ergebnisvektor wieder in den allgemeinen
X_test[4] = y_test                                                                                                      #dataset einfügen
# evaluate algorithm
num_neighbors = 3
prediction = k_nearest_neighbors(X_train, X_test, num_neighbors)
actual = X_test[4].tolist()
for i in range(len(prediction)):
    print("Expected %d, Got %d" % (actual[i],prediction[i]))
score = accuracy_metric(actual, prediction)
print("Score: ",score)