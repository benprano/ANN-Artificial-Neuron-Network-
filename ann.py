# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 13:06:20 2019

@author: Mamadou Ben Hamidou
"""

import tensorflow as tf
import theano
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Data preprocessing 
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:,13].values

# Encoding the categorical variables
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Building the ANN   m  
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
# Initialising the ANN Artificial Neurons Networks
classifier = Sequential()
# Adding the input layers and the first hidden layers with dropout
classifier.add(Dense(output_dim = 8, init= 'uniform', activation= 'relu', input_dim = 11))
classifier.add(Dropout(P = 0.1))
# Adding the input layers and the second hidden layers
classifier.add(Dense(output_dim = 8, init= 'uniform', activation= 'relu'))
classifier.add(Dropout(P = 0.1))
# Adding the input layers and the third hidden layers
classifier.add(Dense(output_dim = 8, init= 'uniform', activation= 'relu'))
classifier.add(Dropout(P = 0.1))
# Adding the input layers and the fourth hidden layers
classifier.add(Dense(output_dim = 8, init= 'uniform', activation= 'relu'))
classifier.add(Dropout(P = 0.1))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init= 'uniform', activation= 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer= 'adam', loss= 'binary_crossentropy' , metrics= ['accuracy'])
# Fitting the ANN to the training set
classifier.fit(X_train , y_train, batch_size= 10 , nb_epoch = 200)
# Predicting the output 
y_pred = classifier.predict(X_test)
# Defining a threshold to define if a customer is leaving or staying 
y_pred = (y_pred > 0.5)

# Single prediction
""" New single prediction
Geography : France
Credit card score = 600
Gender : Male
Age : 40
Tenure : 3
Balance : 60000
Number of products : 2
Has credit card : Yes
Is actived Member : Yes
Estimated Salary : 50000
"""
new_prediction = classifier.predict(sc.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1 , 1, 50000 ]])))
new_prediction = (new_prediction > 0.5)
# The confusion matrix

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Improving the model by the k-folds cross-validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
def buildclassifier():
    # Initialising the ANN Artificial Neurons Networks
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init= 'uniform', activation= 'relu', input_dim = 11))
    classifier.add(Dense(output_dim = 6, init= 'uniform', activation= 'relu'))
    classifier.add(Dense(output_dim = 6, init= 'uniform', activation= 'relu'))
    #classifier.add(Dense(output_dim = 6, init= 'uniform', activation= 'relu'))
    classifier.add(Dense(output_dim = 1, init= 'uniform', activation= 'sigmoid'))
    classifier.compile(optimizer= 'adam', loss= 'binary_crossentropy' , metrics= ['accuracy'])
    
    return classifier

classifier = KerasClassifier(build_fn = buildclassifier , batch_size= 10 , nb_epoch = 100 )
accuracies = cross_val_score(estimator = classifier, X = X_train , y = y_train, cv= 10, n_jobs= 1)

mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN
# Dropout regularization technique to reduce overfitting if needed
# Parameter tunning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
def buildclassifier(optimizer):
    # Initialising the ANN Artificial Neurons Networks
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init= 'uniform', activation= 'relu', input_dim = 11))
    #classifier.add(Dropout(P = 0.1))
    classifier.add(Dense(output_dim = 6, init= 'uniform', activation= 'relu'))
    #classifier.add(Dropout(P = 0.1))
    #classifier.add(Dense(output_dim = 6, init= 'uniform', activation= 'relu'))
    #classifier.add(Dropout(P = 0.1))
    #classifier.add(Dense(output_dim = 6, init= 'uniform', activation= 'relu'))
    classifier.add(Dense(output_dim = 1, init= 'uniform', activation= 'sigmoid'))
    classifier.compile(optimizer= optimizer, loss= 'binary_crossentropy' , metrics= ['accuracy'])
    
    return classifier

classifier = KerasClassifier(build_fn = buildclassifier)
parameters  = {'batch_size': [25, 32 ],
               'nb_epoch': [100, 500],
               'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid= parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train , y_train) 
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_