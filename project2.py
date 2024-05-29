#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


data = pd.read_csv(r"C:\Users\Akshara\Desktop\pyth\creditcard_2023.csv")


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


fraud = data.loc[data['Class'] == 1]
normal = data.loc[data['Class'] == 0]


# In[6]:


fraud.count()


# In[7]:


len(fraud)


# In[8]:


len(normal)


# In[9]:


data.describe()


# In[10]:


sns.relplot(x ='Amount', y ='id',hue = "Class", data=data)


# In[12]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split


# In[15]:


x = data.iloc[:,:-1]
y = data['Class']


# In[16]:


x_train,x_test, y_train,y_test = train_test_split(x,y,test_size=0.35)


# In[17]:


clf = linear_model.LogisticRegression(C=1e5)


# In[18]:


clf.fit(x_train,y_train)


# In[20]:


y_pred = np.array(clf.predict(x_test))
y = np.array(y_test)


# In[22]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[23]:


print(confusion_matrix(y_test,y_pred))


# In[25]:


print(accuracy_score(y_test,y_pred))


# In[26]:


print(classification_report(y_test,y_pred))


# In[ ]:




