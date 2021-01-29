#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##### Author : Amir Shokri
##### github link : https://github.com/amirshnll/Abalone
##### dataset link : http://archive.ics.uci.edu/ml/datasets/Abalone
##### email : amirsh.nll@gmail.com


# In[2]:


import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler  
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix


# In[3]:


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


# In[4]:


#separate the Training data and Test data
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1, test_size=0.2)
# Feature scaling
scaler = StandardScaler()  
scaler.fit(X_train)
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)


# In[5]:


# Finally for the Logistic Regression
lgisticRegr = LogisticRegression(solver='newton-cg', random_state=0 ,max_iter=2000)
lgisticRegr.fit(X_train, y_train.values.ravel())


# In[6]:


#In the prediction step, the model is used to predict the response for given data.
predictions = lgisticRegr.predict(X_test)
print(predictions)


# In[7]:


# Last thing: evaluation of algorithm performance in classifying 
cnf_matrix =confusion_matrix(y_test,predictions)
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))


# In[8]:


# create heatmap
class_names=[0,1] 
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

