import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
#from keras.models import Sequential
#from keras.layers.core import Dense, Dropout, Activation, Flatten
#from keras.layers.convolutional import Convolution2D, MaxPooling2D
#from keras.optimizers import SGD,RMSprop,adam
#from keras.utils import np_utils
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import numpy as np
import matplotlib
import os
import theano
from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
#from sklearn.externals.six import StringIO
from six import StringIO
import pydotplus

path1 = '/InputData/'    #path of folder of images
path2 = '/InputData_resized/'  #path of folder to save images
rootdir = os.path.dirname(os.path.abspath(__file__))
listing = os.listdir(os.path.join(rootdir + path1))
num_samples = size(listing)
print(num_samples)

# input image dimensions
img_rows, img_cols = 200, 200

# number of channels
img_channels = 1

for file in listing:
    im = Image.open(os.path.join(rootdir + path1) + file)
    img = im.resize((img_rows,img_cols))
    gray = img.convert('L')
                #need to do some more processing here
    gray.save(os.path.join(rootdir + path2)  + file, "JPEG")

imlist = os.listdir(os.path.join(rootdir + path2))

im1 = array(Image.open(os.path.join(rootdir + path2) + imlist[0])) # open one image to get size
m,n = im1.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images

# create matrix to store all flattened images
immatrix = array([array(Image.open(os.path.join(rootdir + path2) + im2)).flatten()
                  for im2 in imlist], 'f')



label = np.ones((num_samples,), dtype=int)
label[0:160] = 1  # bürste
label[161:] = 0   # kamm

data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]

# %%
(X, y) = (train_data[0], train_data[1])
# STEP 1: split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)



# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


from IPython.display import Image

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('DecisionTree_graph.png')
Image(graph.create_png())
