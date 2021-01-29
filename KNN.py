#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##### Author : Amir Shokri
##### github link : https://github.com/amirshnll/Abalone
##### dataset link : http://archive.ics.uci.edu/ml/datasets/Abalone
##### email : amirsh.nll@gmail.com


# In[9]:


import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matri


# In[10]:


#read file
df = pd.read_csv("D:\\abalone.txt", header=None)
for char in df:
    df = df.replace('M','1')
    df = df.replace('F','-1')
    df = df.replace('I','0')
df

#separate the feature columns from the target column.
features = [0,1,2,3,4,5,6,7]
X = df[features]
y = df[8]
print(X)
print(y)


# In[11]:


#separate the Training data and Test data
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1,test_size=0.2)
# Feature scaling
scaler = StandardScaler()  
scaler.fit(X_train)
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)


# In[12]:


# Finally for the knn-k nearest neighbor(k=1,3,5,7,9)
test=[] 
train=[] 
knn=[]
for i in range(1, 10, 2):

    # Finally for the KNN
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(X_train, y_train)
   
    #In the prediction step, the model is used to predict the response for given data.
    predictions =classifier.predict(X_test)
    print(predictions)

    # Last thing: evaluation of algorithm performance in classifying 
    print(confusion_matrix(y_test,predictions))  
    print(classification_report(y_test,predictions))

    # mean accuracy on the given test data and labels.
    knn.append(i)
    test.append(classifier.score(X_test,y_test))
    train.append(classifier.score(X_train,y_train))
    print("Accuracy Test  k[",i,"]",classifier.score(X_test,y_test))
    print("Accuracy Train k[",i,"]",classifier.score(X_train,y_train))
    


# In[13]:


plt.plot(knn, test)
plt.plot(knn, train)
plt.show()

