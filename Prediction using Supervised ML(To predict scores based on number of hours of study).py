#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


url = "http://bit.ly/w-data"
df = pd.read_csv(url)
df.head()


# In[5]:


# selecting the dependent and independent variables
X = df.iloc[:, :-1].values  
Y = df.iloc[:, 1].values


# In[6]:


# splitting data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# In[7]:


#importing the LinearRegression Model and fitting it
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[8]:


y_pred = regressor.predict(X_test)


# In[9]:


print(y_pred)


# In[10]:


# predicting the score if the hours of study is 9.25
own_pred=regressor.predict([[9.25]])
print(own_pred)


# In[11]:


#comparing the actual and predicted value through visualization
sns.distplot(y_test,hist=False,color="purple",label="actual")
sns.distplot(y_pred,hist=False,color="green",label="Predicted")


# In[12]:


# model evaluation
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) 


# In[ ]:




