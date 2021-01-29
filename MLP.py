#!/usr/bin/env python
# coding: utf-8

# In[1]:


##### Author : Amir Shokri
##### github link : https://github.com/amirshnll/Abalone
##### dataset link : http://archive.ics.uci.edu/ml/datasets/Abalone
##### email : amirsh.nll@gmail.com


# In[ ]:


import sklearn
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


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


# In[ ]:


#separate the Training data and Test data
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1, test_size=0.2)
# Feature scaling
scaler = StandardScaler()  
scaler.fit(X_train)
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)


# In[ ]:


# Finally for the MLP- Multilayer Perceptron
mlp = MLPClassifier(max_iter=1000)  
mlp.fit(X_train, y_train.values.ravel())


# In[ ]:


#In the prediction step, the model is used to predict the response for given data.
predictions = mlp.predict(X_test)
print(predictions)


# In[23]:


# Last thing: evaluation of algorithm performance in classifying 
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))


# In[ ]:




