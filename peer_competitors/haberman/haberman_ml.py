# This file is created to compare ML Classifiers with NLC
# Copyright (c) 2021 Hamit Taner Ünal and Prof.Fatih Başçiftçi

# Classification Task for Haberman's Survival Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import std, median
from numpy import mean
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, make_scorer, accuracy_score, RocCurveDisplay, auc
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, cross_validate
import warnings
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger('tensorflow').disabled = True
warnings.filterwarnings('ignore')

#Determine seed
seed = 7
np.random.seed(seed)

# Read data
list_cols = ['Age', "year_of_surgery", "axillary_nodes", "survival_status"]
data = pd.read_csv("haberman.csv", names=list_cols)

# Print data summary
print(data.head())
print(data.info())

# Split the columns to input and output
y = data['survival_status']
x = data.drop('survival_status', axis = 1)
X = x.astype('float32')

# Label encode output to 0/1
y = LabelEncoder().fit_transform(y)


#Use seed to reproduce the results
print("Using seed ",seed)

# Define models to be used (with best hyperparameters)
model1 = LogisticRegression(C=0.1, max_iter=100, penalty='l2',solver='liblinear')
model2 = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=4, bootstrap = True, max_features = 'log2')
model3 = Sequential()
model3.add(Dense(9, input_dim=X.shape[1], activation='relu'))
model3.add(Dense(3, activation='relu'))
model3.add(Dense(1, activation="sigmoid"))
model3.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])
model4 = DecisionTreeClassifier(criterion='entropy',max_depth=2,max_features='log2',min_samples_leaf=3,min_samples_split=3)
model5 = GaussianNB(var_smoothing=0.005336699231206307)
model6 = KNeighborsClassifier(leaf_size=35, metric='chebyshev',n_neighbors=27, p = 1,weights='uniform')
model7 = SVC(C=1,gamma=0.1,kernel='rbf')

#Define model3 (ANN) as Keras Classifier
# Prepare for ANN
def create_ANN_model(optimizer='RMSprop'):
    model = Sequential()
    model.add(Dense(9, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

model3_keras = KerasClassifier(build_fn=create_ANN_model, verbose=0, epochs=150, batch_size=20)
model3_keras._estimator_type = "classifier"

# Define Stratified k-fold Cross Validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

# Initialize scores for each classifier
overall_score1 = []
overall_score2 = []
overall_score3 = []
overall_score4 = []
overall_score5 = []
overall_score6 = []
overall_score7 = []

# Start k-fold cross validation loop
# Print Classification Reports
csv1 = []
csv2 = []
csv3 = []
csv4 = []
csv5 = []
csv6 = []
csv7 = []
print("Classification Report for Logistic Regression:")
print("----------------------------------------------")
for score in [ "accuracy", "recall", "precision", "f1", "balanced_accuracy"]:
    steps = list()
    steps.append(('scaler', StandardScaler()))
    steps.append(('model', model1))
    pipeline = Pipeline(steps=steps)
    cv_score = cross_val_score(pipeline, X, y, scoring=score, cv=kf)
    print(score + " Mean: %.8f" % mean(cv_score) + " STD: %.8f" % std(cv_score) + " Median: %.8f" % median(cv_score))
    csv1.append(str(mean(cv_score)))
    csv1.append(str(std(cv_score)))
print("----------------------------------------------")

print("Classification Report for Random Forest:")
print("----------------------------------------------")
for score in [ "accuracy", "recall", "precision", "f1", "balanced_accuracy"]:
    steps = list()
    steps.append(('scaler', StandardScaler()))
    steps.append(('model', model2))
    pipeline = Pipeline(steps=steps)
    cv_score = cross_val_score(pipeline, X, y, scoring=score, cv=kf)
    print(score + " Mean: %.8f" % mean(cv_score) + " STD: %.8f" % std(cv_score) + " Median: %.8f" % median(cv_score))
    csv2.append(str(mean(cv_score)))
    csv2.append(str(std(cv_score)))
print("----------------------------------------------")

print("Classification Report for ANN:")
print("----------------------------------------------")
for score in [ "accuracy", "recall", "precision", "f1", "balanced_accuracy"]:
    steps = list()
    steps.append(('scaler', StandardScaler()))
    steps.append(('model', model3_keras))
    pipeline = Pipeline(steps=steps)
    cv_score = cross_val_score(pipeline, X, y, scoring=score, cv=kf)
    print(score + " Mean: %.8f" % mean(cv_score) + " STD: %.8f" % std(cv_score) + " Median: %.8f" % median(cv_score))
    csv3.append(str(mean(cv_score)))
    csv3.append(str(std(cv_score)))
print("----------------------------------------------")

print("Classification Report for CART:")
print("----------------------------------------------")
for score in [ "accuracy", "recall", "precision", "f1", "balanced_accuracy"]:
    steps = list()
    steps.append(('scaler', StandardScaler()))
    steps.append(('model', model4))
    pipeline = Pipeline(steps=steps)
    cv_score = cross_val_score(pipeline, X, y, scoring=score, cv=kf)
    print(score + " Mean: %.8f" % mean(cv_score) + " STD: %.8f" % std(cv_score) + " Median: %.8f" % median(cv_score))
    csv4.append(str(mean(cv_score)))
    csv4.append(str(std(cv_score)))
print("----------------------------------------------")

print("Classification Report for Naive Bayes:")
print("----------------------------------------------")
for score in [ "accuracy", "recall", "precision", "f1", "balanced_accuracy"]:
    steps = list()
    steps.append(('scaler', StandardScaler()))
    steps.append(('model', model5))
    pipeline = Pipeline(steps=steps)
    cv_score = cross_val_score(pipeline, X, y, scoring=score, cv=kf)
    print(score + " Mean: %.8f" % mean(cv_score) + " STD: %.8f" % std(cv_score) + " Median: %.8f" % median(cv_score))
    csv5.append(str(mean(cv_score)))
    csv5.append(str(std(cv_score)))
print("----------------------------------------------")

print("Classification Report for kNN:")
print("----------------------------------------------")
for score in [ "accuracy", "recall", "precision", "f1", "balanced_accuracy"]:
    steps = list()
    steps.append(('scaler', StandardScaler()))
    steps.append(('model', model6))
    pipeline = Pipeline(steps=steps)
    cv_score = cross_val_score(pipeline, X, y, scoring=score, cv=kf)
    print(score + " Mean: %.8f" % mean(cv_score) + " STD: %.8f" % std(cv_score) + " Median: %.8f" % median(cv_score))
    csv6.append(str(mean(cv_score)))
    csv6.append(str(std(cv_score)))
print("----------------------------------------------")

print("Classification Report for SVC:")
print("----------------------------------------------")
for score in [ "accuracy", "recall", "precision", "f1", "balanced_accuracy"]:
    steps = list()
    steps.append(('scaler', StandardScaler()))
    steps.append(('model', model7))
    pipeline = Pipeline(steps=steps)
    cv_score = cross_val_score(pipeline, X, y, scoring=score, cv=kf)
    print(score + " Mean: %.8f" % mean(cv_score) + " STD: %.8f" % std(cv_score) + " Median: %.8f" % median(cv_score))
    csv7.append(str(mean(cv_score)))
    csv7.append(str(std(cv_score)))
print("----------------------------------------------")
print("***** Printing CSV Data to export *************")
print(csv1)
print(csv2)
print(csv3)
print(csv4)
print(csv5)
print(csv6)
print(csv7)
print("******** End of CSV ***************************")
