# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 18:29:13 2020
Mushroom Classification
@author: Santosh
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#% matplotlib inline


df = pd.read_csv(r"D:\BeCode\ML\SL\Challenge\mushrooms.csv")
print(df.head())

# checking possible values each variable can take and for stalk-root we notice that '?' is also a possible value.
for c in df.columns:    
    if (df[c].dtype==object):
        print(c,df[c].unique())
        

#here we are just selection features with categorical data into a list named cv
cv=[]
for c in df.columns:
    if df[c].dtype==object:
        cv.append(c)        
        
#here we map values of stalk root to integers and '?' to nan so that we can use mean imputation on stalk- root later        
df['stalk-root']=df['stalk-root'].map({'?':np.nan,'e':1,'c':2,'b':3,'r':4})        

cv.remove('stalk-root')
print(cv)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for c in cv:
    df[c]=le.fit_transform(df[c])
    
from sklearn.impute import SimpleImputer
im=SimpleImputer()
m=im.fit_transform(df)
d=pd.DataFrame(m,columns=df.columns)   


l=[]
for col in d.columns:
    if d[col].std()==0:
        l.append(col)
print(l)
d.drop('veil-type',axis=1,inplace=True)



# input 
x = d.iloc[:, 1:22].values 
  
# output 
y = d.iloc[:, 0].values 

#===Splitting the dataset to train and test. 75% of data is used for training the 
# model and 25% of it is used to test the performance of our model.

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25, random_state = 0) 


'''from sklearn.preprocessing import StandardScaler 
sc_x = StandardScaler() 
xtrain = sc_x.fit_transform(xtrain) 
xtest = sc_x.transform(xtest) 

print (xtrain[0:10, :]) '''

#============ Logistic Regression===========#

from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state = 0) 
classifier.fit(xtrain, ytrain) 

y_pred = classifier.predict(xtest) 

#======

from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(ytest, y_pred) 
  
print ("Confusion Matrix for LR : \n", cm) 

#=======Performance measure – Accuracy

from sklearn.metrics import accuracy_score 
print ("Accuracy of LR : ", accuracy_score(ytest, y_pred)) 



##=======KNN===========##

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)

# Train the model using the training sets
model.fit(xtrain, ytrain)

y_pred = model.predict(xtest) 

#======

from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(ytest, y_pred) 
  
print ("Confusion Matrix for KNN: \n", cm) 

#=======Performance measure – Accuracy

from sklearn.metrics import accuracy_score 
print ("Accuracy of KNN : ", accuracy_score(ytest, y_pred)) 

#=============SVM Model=============#

#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='rbf') # Linear Kernel

#Train the model using the training sets
clf.fit(xtrain, ytrain)

#Predict the response for test dataset
y_pred = clf.predict(xtest)

#============Evaluating the Model===========#

#Import scikit-learn metrics module for accuracy calculation

from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(ytest, y_pred) 
  
print ("Confusion Matrix for SVM: \n", cm) 

from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy of SVM:",metrics.accuracy_score(ytest, y_pred))

