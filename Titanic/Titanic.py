# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 21:06:00 2023

@author: Reynaldo Etkins
"""

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from pandas_profiling import ProfileReport
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

# Load in Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Analyze Data
df_num = train[['Age', 'Parch', 'Fare', 'SibSp']]
df_cat = train[['Sex', 'Survived', 'Cabin', 'Embarked', 'Ticket', 'Pclass']]
train.describe()
train.shape
train.value_counts()
train.info()
train.isnull().sum()

# Graph numerical columns
for i in df_num.columns:
    plt.hist(df_num[i])
    plt.title(i)
    plt.show()

# Drop and fill null values
train.Age = train.Age.fillna(train.Age.mean())
train.Fare = train.Fare.fillna(train.Fare.median())

train.drop(['Cabin', 'PassengerId', 'Name', 'Ticket'], inplace=True, axis=1)
train.dropna(subset = ['Embarked'], inplace=True)

# Label Encode df so all dtypes are numerical
le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
train['Embarked'] = le.fit_transform(train['Embarked'])

# Create X and y
y = train['Survived']
x = train.iloc[:, 1:8].values

# Drop and fill null values
test.Age = test.Age.fillna(test.Age.mean())
test.Fare = test.Fare.fillna(test.Fare.median())

test.drop(['Cabin', 'PassengerId', 'Name', 'Ticket'], inplace=True, axis=1)
test.dropna(subset = ['Embarked'], inplace=True)

# Label Encode df so all dtypes are numerical
le = LabelEncoder()
test['Sex'] = le.fit_transform(test['Sex'])
test['Embarked'] = le.fit_transform(test['Embarked'])

# Create X and y
x_test = train.iloc[:, 1:8].values

# Create Models
def model(x,y,x_test):
    # Logistic Regression Model
    lr = LogisticRegression()
    lr.fit(x,y)
    lr_pred = lr.predict(x_test)
    
    # KNeighbor Classifier
    knn = KNeighborsClassifier(n_neighbors=5, p=2)
    knn.fit(x,y)
    kp = knn.predict((x_test))
    
    # Gaussian Model
    gauss = GaussianNB()
    gauss.fit(x,y)
    
    # Random Forest Model
    forest = RandomForestClassifier(n_estimators=10, criterion='entropy')
    forest.fit(x,y)
    
    # Decision Tree Model
    tree = DecisionTreeClassifier(criterion='entropy')
    tree.fit(x,y)
    
    # Prints Accuracy of all Models
    print('[0]Logistic Regression Traning Accuracy: ', lr.score(x,y))
    print('[1]Knn Traning Accuracy: ', knn.score(x,y))
    print('[2]Guass Traning Accuracy: ', gauss.score(x,y))
    print('[3]Forest Traning Accuracy: ', forest.score(x,y))
    print('[4]Tree Traning Accuracy: ', tree.score(x,y))
    
    return lr, knn, gauss, forest, tree

model = model(x,y,x_test)

plt.scatter(train['Sex'], train['Age'], s=train['Survived'])
