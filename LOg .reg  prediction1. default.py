#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 11:09:09 2023

@author: myyntiimac
"""
#Logistic regression
# we build logistic model on historical data, then we pass the futre data to our developed model , so that it predcts future
#import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv("/Users/myyntiimac/Desktop/logit classification.csv")
df.head()
df.shape
df.isnull().any()
#So in our dataframe column contain 400 rows and 5 column, from taht column , we decided our dependent variable is puchased
#and from others independent variables we  wiil select only relavent column , and think that gender and user id  is not important to predict futre
# so we will delete unrelavent column during variable define , when slicing
X = df.iloc[:,[2,3]].values
X
Y = df.iloc[:,-1]
# now split the X;Y into train and test data with test size 25% and random state-0
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X, Y, test_size = 0.25,random_state = 0 )

#Feature scaling
# we saw in our independent variable , the attribute in different range , so we need to it normalize , lets try first with std.scaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# remember scaling need for independent variable 

#now build the model with train data
# call  the logistic reg function , ab´nd define it , then fit and train the model with your train data
# with deafault function without parameter change
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# then predict the y_pred for our X_test data
y_pred = classifier.predict(X_test)

# now compare the value of y_test and Ypred, how model predict
# we saww there is missclassification happen , lets try to find the how many missclassifiacation happen(false postive, false negative)
# so here come , confusion matrix , to build this confusion matrix we need to import co. matrix from sklearn.matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#We find that 11 missclassification happen where FP-3, FN-8
#Lets find the accuracy of model
from sklearn.metrics import accuracy_score 
ac = accuracy_score(y_test, y_pred)
print(ac)

#we found that accuracy score of our model is 89%., its seem good model
#Lets find the classification report
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
cr
# in this report we found both precision, recall and accracy are .89, seems look good model
#Lets try to find bias and variance
bias = classifier.score(X_train, y_train)
bias
variance = classifier.score(X_test, y_test)
variance

# bias is ,.82, and varaince is .89, almost similer , its  agood model 
# lets see in graph , how its look like
##Future prdeiction
df1=pd.read_csv("/Users/myyntiimac/Desktop/Future prediction models comparision.csv")
df1.head()
df1.shape
#copy df1 for assignn the prediction value after prediction
FD=df1.copy()
FD
#Then the defined the dataset for futyre prediction
X1= df1.iloc[:,[3,4]].values
X1
#scalize the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 
C= sc.fit_transform(X1)
C

# create an empty dataframe for assign the predic value Then predict
y_pred2=classifier.predict(X1)
y_pred2


#now call your copy dataframe FD where you want to assignpredicted value as new column 
# so create anew column in FD
FD['predict'] = pd.Series(y_pred2)
FD
#Now save it 
FD.to_csv("logpred.csv")



# can we improve this model ?
#we can try to improve the model , by test split (15%, 20%), we can also imperove by feature scaling by normalizatin or minmax scaler
#we can also try to improve wtth parameter tuning inLR function#Lets try

