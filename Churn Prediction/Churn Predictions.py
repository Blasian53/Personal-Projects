# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 17:57:10 2023

@author: Reynaldo Etkins
"""

# Imports all libraries excludes pandas ProfileReport
#!pip install pandas_profiling
#from pandas_profiling import ProfileReport
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Loads in data and saves it to df
data = '/content/WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(data, encoding='unicode_escape') # Add the encoding since it wont run without
df.head()

# Analyze data
df.describe()
df.info()
df.isna().sum()
df.duplicated().sum()
df.shape
df.customerID.value_counts()
df.SeniorCitizen.value_counts(normalize=True)

#ProfileReport(df)

# Function to clean the data
def wrangle(df):
  # Drop High Cardinality Columns
  df.drop(['customerID'], inplace=True, axis=1)
  
  # Converts "TotalCharges" to a numerical value
  df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')
  
  # Drops Null Values
  df.dropna(inplace=True)
  
  # Drops Duplicates
  df.drop_duplicates(inplace=True)
  
  # One Hot Encodes the Churn column seperately
  # so it doesent create two seperate columns when using get_dummies
  df.Churn.replace(to_replace='Yes', value=1, inplace=True)
  df.Churn.replace(to_replace='No', value=0, inplace=True)
  return df
df = wrangle(df)

# Visualizations for the data
df.Contract.value_counts().plot(kind='barh')
df.gender.value_counts().plot(kind='pie')
sns.pairplot(df)

# One Hot Encodes the whole dataframe
df = pd.get_dummies(df)
df.head()

# Plots the correlation of all features to the Churn column 
plt.figure(figsize=(15,15))
df.corr()['Churn'].sort_values().plot(kind='barh')
plt.show()

# Creates the X and y
target = 'Churn'
y = df[target].values
X = df.drop(columns=target)

# Scales the data using MinMaxScaler
features = X.columns.values
scaler = MinMaxScaler()
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X))
X.columns = features

# Splits data with train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

# Creates LogisticRegression model
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
lr_pred = model_lr.predict(X_test)

# Evaluates LR Model
accuracy_score(y_test, lr_pred)
confusion_matrix(y_test, lr_pred)
lr_score = cross_val_score(model_lr, X_train, y_train, cv=10, scoring='accuracy', n_jobs=-1)
lr_score.mean()
print(classification_report(y_test,lr_pred))

# Visualization for LR Model
weights = pd.Series(model_lr.coef_[0], index=X.columns)
weights.sort_values()[-10:].plot(kind='barh')

# Creates DecisionTreeClassifer Model
model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train, y_train)
dt_pred = model_dt.predict(X_test)

# Evaluates DT Model
accuracy_score(y_test, dt_pred)
confusion_matrix(y_test, dt_pred)
dt_score = cross_val_score(model_dt, X_train, y_train, n_jobs=-1, cv=10, scoring='accuracy')
dt_score.mean()
print(classification_report(y_test, dt_pred))

# Visualization for DT Model
dt_importance = model_dt.feature_importances_
weights = pd.Series(dt_importance, index=X.columns)
weights.sort_values()[-10:].plot(kind='barh')

# Creates RandomForestClassifer Model
model_rf = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42, verbose=2)
model_rf.fit(X_train, y_train)
rf_pred = model_rf.predict(X_test)

# Evaluates RF Model
accuracy_score(y_test, rf_pred)
confusion_matrix(y_test, rf_pred)
rf_score = cross_val_score(model_rf, X_train, y_train, cv=10, scoring='accuracy')
rf_score.mean()
print(classification_report(y_test, rf_pred))

# Visualization for RF Model
rf_importance = model_rf.feature_importances_
weights = pd.Series(rf_importance, index=X.columns.values)
weights.sort_values()[-10:].plot(kind='barh')

# Creates KNeighborClassifier Model
model_kn = KNeighborsClassifier(n_neighbors=10)
model_kn.fit(X_train, y_train)
kn_pred = model_kn.predict(X_test)

# Evaluates KNN Model
accuracy_score(y_test,kn_pred)
confusion_matrix(y_test, kn_pred)
print(classification_report(y_test, kn_pred))