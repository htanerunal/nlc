# This file is created to compare ML Classifiers with NLC
# Copyright (c) 2021 Hamit Taner Ünal and Prof.Fatih Başçiftçi

# Finding best hyperparameters for classification task on Haberman's Survival dataset
from keras import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Read data
list_cols = ['Age', "year_of_surgery", "axillary_nodes", "survival_status"]
data = pd.read_csv("dataset.csv", names=list_cols)

# Print data summary
print(data.head())
print(data.info())

# Split the columns to input and output
y = data['survival_status']
x = data.drop('survival_status', axis = 1)
x = x.astype('float32')

# Label encode output to 0/1
y = LabelEncoder().fit_transform(y)

# Transform input data for better classification
scaler = StandardScaler()
X = scaler.fit_transform(x)

# Params for LogisticRegression()
param_grid1 = {
    'C':[1e-5, 1e-4, 1e-3, 1e-2, 1e-1,1,10,100],
    'penalty':['l1','l2'],
    'solver' : ['newton-cg','lbfgs','liblinear'],
    'max_iter':[100,200,300,500]
}

# Params for Random Forest
param_grid2 = {
    'bootstrap': [False,True],
    'n_estimators': [100,200,300,400,500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

# Prepare for ANN
def create_ANN_model(optimizer='adam'):
    model = Sequential()
    model.add(Dense(9, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Params for Keras ANN Classifier
param_grid3 = {
    'batch_size': [10, 20, 50, 100],
    'epochs': [10, 50, 100, 150],
    'optimizer': ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
}

# Params for DecisionTreeClassifier()
param_grid4 = {
    'min_samples_split':range(1,10),
    'min_samples_leaf':range(1,10),
    'max_depth' : range(1,10),
    'max_features': ['sqrt','log2','none'],
    'criterion' :['gini', 'entropy']
}

# Params for GaussianNB()
param_grid5 = {
    'var_smoothing': np.logspace(0,-9, num=100)
}

# Params for KNeighborsClassifier()
param_grid6 = {
    'leaf_size': range(1,50),
    'n_neighbors': range(1,30),
    'p': [1,2],
    'weights': ['uniform', 'distance'],
    'metric': ['minkowski', 'chebyshev']
}

# Params for Support Vector Classifier (SVC)
param_grid7 = {
    'C': [0.1,1,10],
    'gamma': [1,0.1,0.01,0.001],
    'kernel': ['rbf', 'poly', 'sigmoid']
}


print("Fitting model1...LR")
model1 = GridSearchCV(estimator=LogisticRegression(), param_grid=param_grid1,scoring='accuracy', refit=True, n_jobs=-1,verbose=3, cv= 5)
model1.fit(X, y)
print('Best config for Logistic Regression: %s' % model1.best_params_)
print("------------------------------------------------------------------------------------------")
print("Fitting model2...RF")
model2 = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid2, scoring='accuracy', refit=True, n_jobs=-1, verbose=3, cv= 5)
model2.fit(X, y)
print('Best config for Random Forest: %s' % model2.best_params_)
print("------------------------------------------------------------------------------------------")
model3 = GridSearchCV(estimator=KerasClassifier(build_fn=create_ANN_model, verbose=0), param_grid=param_grid3, n_jobs=-1,scoring='accuracy', verbose=3, cv= 3)
model3.fit(X, y)
print('Best config for ANN: %s' % model3.best_params_)
print("------------------------------------------------------------------------------------------")
print("Fitting model4...CART")
model4 = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param_grid4, scoring='accuracy', refit=True, n_jobs=-1,verbose=3, cv= 5)
model4.fit(X, y)
print('Best config for Decision Trees: %s' % model4.best_params_)
print("------------------------------------------------------------------------------------------")
print("Fitting model5...NB")
model5 = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid5,scoring='accuracy', refit=True, verbose=3, n_jobs=-1, cv= 5)
model5.fit(X, y)
print('Best config for NaiveBayes: %s' % model5.best_params_)
print("------------------------------------------------------------------------------------------")
print("Fitting model6...KNN")
model6 = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid6, scoring='accuracy', refit=True, verbose=3,n_jobs=-1, cv= 5)
model6.fit(X, y)
print('Best config for KNN: %s' % model6.best_params_)
print("------------------------------------------------------------------------------------------")
print("Fitting model7...SVC")
model7 = GridSearchCV(estimator=SVC(), param_grid=param_grid7, scoring='accuracy', refit=True, n_jobs=-1,verbose=2, cv= 5)
model7.fit(X, y)
print('Best config for SVC: %s' % model7.best_params_)
print("------------------------------------------------------------------------------------------")

print("***************  GRID SEARCH RESULTS FOR ML ALGORITHMS ***********************************")
print('Best config for Logistic Regression: %s' % model1.best_params_)
print('Best config for Random Forest: %s' % model2.best_params_)
print('Best config for ANN: %s' % model3.best_params_)
print('Best config for Decision Trees: %s' % model4.best_params_)
print('Best config for Naive Bayes: %s' % model5.best_params_)
print('Best config for KNN: %s' % model6.best_params_)
print('Best config for SVC: %s' % model7.best_params_)

print("*****************  END OF REPORT *********************************************************")
