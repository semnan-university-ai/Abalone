#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##### Author : Amir Shokri
##### github link : https://github.com/amirshnll/Abalone
##### dataset link : http://archive.ics.uci.edu/ml/datasets/Abalone
##### email : amirsh.nll@gmail.com


# In[5]:


import sklearn
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix


# In[6]:


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


# In[7]:


#separate the Training data and Test data
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1, test_size=0.2)
# Feature scaling
scaler = StandardScaler()  
scaler.fit(X_train)
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)


# In[8]:


# Finally for the Naive Bayes
NB = GaussianNB()
NB.fit(X_train, y_train);


# In[9]:


#In the prediction step, the model is used to predict the response for given data.
predictions = NB.predict(X_test)
print(predictions)


# In[10]:


#Last thing: evaluation of algorithm performance in classifying 
matrix=confusion_matrix(y_test,predictions)

print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))


# In[11]:


#create heatmap
class_names=[0,1] 
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[ ]:




