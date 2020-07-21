import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf


# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)
le1 = LabelEncoder()
X[:,2]=le1.fit_transform(X[:,2])
le2 =LabelEncoder()
X[:,4]= le2.fit_transform(X[:,4])
X = X[:,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Building the ANN

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#Creating a nn object
classifier = Sequential()

#Creating a hidden Layer
classifier.add(Dense(output_dim = 6,init = 'uniform', activation= 'relu',input_dim = 11))
classifier.add(Dropout(p=0.1))

#Creating a second hidden layer
classifier.add(Dense(output_dim = 6,init = 'uniform', activation= 'relu'))
classifier.add(Dropout(p=0.1))

#Creating output layer
classifier.add(Dense(output_dim = 1,init = 'uniform', activation= 'sigmoid'))

#Compiling ann
classifier.compile(optimizer= 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])

#Fitting the ann
classifier.fit(X_train, y_train, batch_size=10, nb_epoch = 100)

#prediction
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

new_pred = classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))



#evaluation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6,init = 'uniform', activation= 'relu',input_dim = 11))
    classifier.add(Dense(output_dim = 6,init = 'uniform', activation= 'relu'))
    classifier.add(Dense(output_dim = 1,init = 'uniform', activation= 'sigmoid'))
    classifier.compile(optimizer= 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])
    return classifier

classifier= KerasClassifier(build_fn= build_classifier, batch_size=10, nb_epoch = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

mean = accuracies.mean()
variance = accuracies.std()

#Improving ANN
#Dropout Regularisation
#from keras.layers import Dropout

#Tuning the ANN

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6,init = 'uniform', activation= 'relu',input_dim = 11))
    classifier.add(Dense(output_dim = 6,init = 'uniform', activation= 'relu'))
    classifier.add(Dense(output_dim = 1,init = 'uniform', activation= 'sigmoid'))
    classifier.compile(optimizer= optimizer, loss = 'binary_crossentropy',metrics = ['accuracy'])
    return classifier

classifier= KerasClassifier(build_fn= build_classifier)

parameters = {'batch_size':[25, 32],
              'nb_epoch':[100, 500],
              'optimizer': ['adam','rmsprop']}

grid_search = GridSearchCV(estimator = classifier,param_grid=parameters, scoring = 'accuracy', cv =10)
grid_search = grid_search.fit(X_train,y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


#takes a lot of time











