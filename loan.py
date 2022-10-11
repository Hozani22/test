# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from statistics import mean, median,variance , stdev, quantiles
##import matplotlib as plt


# loading the dataset to pandas DataFrame
##loan_dataset = pd.read_csv('dataset.csv')
file = 'C:/Users/Hussam/VScode/MWS_ADM/loan_data_set.csv'
loan_dataset = pd.read_csv(file)


def open_File():
    # printing the first 5 rows of the dataframe
    print("-----------printing the first 5 rows of the dataframe-------------\n")
    print(loan_dataset.head())

open_File()

# number of rows and columns
print('-------number of rows and columns------\n')
print(loan_dataset.shape)

# statistical measures
print('-------statistical measures------\n')
print(loan_dataset.describe())

# number of missing values in each column
print('------number of missing values in each column-------\n')
print(loan_dataset.isnull().sum())

###loan_dataset.isnull().replace({"LoanAmount":{'N':0,'Y':1}},inplace=True)

# dropping the missing values
print('------dropping the missing values-------\n')
loan_dataset = loan_dataset.dropna()

# number of missing values in each column
print('------number of missing values in each column-------\n')
print(loan_dataset.isnull().sum())

# values replacement
print('-----values replacement--------\n')
loan_dataset.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)

# printing the first 5 rows of the dataframe
print('-----printing the first 5 rows of the dataframe--------\n')
print(loan_dataset.head())

# Dependent column values
print('------Dependent column values-------\n')
print(loan_dataset['Dependents'].value_counts())

# replacing the value of 3+ to 4
print('------ replacing the value of 3+ to 4-------\n')
loan_dataset = loan_dataset.replace(to_replace='3+', value=3)

# dependent values
print("--------------dependent values--------------------------")
print(loan_dataset['Dependents'].value_counts())

"""Data Visualization"""

# education & Loan Status
print("---------------education & Loan Status---------------")
print(sns.countplot(x='Education',hue='Loan_Status',data=loan_dataset))
sns.countplot(x='Education',hue='Loan_Status',data=loan_dataset)

# marital status & Loan Status
print("------------------marital status & Loan Status------------------")
sns.countplot(x='Married',hue='Loan_Status',data=loan_dataset)
print(sns.countplot(x='Married',hue='Loan_Status',data=loan_dataset))
##plt. show(sns.countplot(x='Married',hue='Loan_Status',data=loan_dataset))

# convert categorical columns to numerical values
loan_dataset.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)

print("--------convert categorical columns to numerical values--------\n")
print(loan_dataset.head())

# separating the data and label
print("-----------------------separating the data and label---------------------")
X = loan_dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y = loan_dataset['Loan_Status']


print("\n--------x---------\n")
print("X=: \n",X)
print("\n--------y---------\n")
print("Y=: ",Y)

"""Train Test Split"""

X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)

print("Xshape: ",X.shape,"\n  X_train.shape: ", X_train.shape,"\n  X_test.shape: ", X_test.shape)

"""
Training the model:
Support Vector Machine Model
"""

classifier = svm.SVC(kernel='linear')

#training the support Vector Macine model  ## to check
print("---------------training the support Vector Macine model----------------")
classifier.fit(X_train,Y_train)

"""Model Evaluation"""

# accuracy score on training data
print("---------------------accuracy score on training data-----------------")
X_train_prediction = classifier.predict(X_train)
training_data_accuray = accuracy_score(X_train_prediction,Y_train)

print('Accuracy on training data : ', training_data_accuray)

# accuracy score on Test data
print("-----------------accuracy score on Test data------------------------")
X_test_prediction = classifier.predict(X_test)
test_data_accuray = accuracy_score(X_test_prediction,Y_test)
print('Accuracy on test data : ', test_data_accuray)

"""Making a predictive system"""
print("---------------------End Prog")
