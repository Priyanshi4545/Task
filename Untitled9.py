#!/usr/bin/env python
# coding: utf-8

# In[3]:


### IMPORTING THE LIBRARIES ###

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings as wg
import sklearn.metrics as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.utils import shuffle


# # Step 1 : Reading the data

# In[4]:


### Reading the data from import link ###

student_info=pd.read_csv('http://bit.ly/w-data')
print(student_info)


# In[5]:


student_info.info()


# In[6]:


student_info.describe()


# In[7]:


student_info.head()


# In[13]:


print(student_info.shape)


# # Step 2 : Data visualization

# In[2]:


student_info(x='hours',y='scores')
plt.title("hours vs scores")
plt.xlable('hours')
plt.ylable('scores')
plt.scatter(x,y)
plt.show()


# In[3]:


# here the heatmap shows positive correlation between the hour column and scores column

plt.figure(figsize=(12,6))
sns.heatmap(student_info[['scores','hours']].corr(),annot=True)
plt.title('the relation matrix',fontsize=14)
plt.show()


# In[4]:


get_ipython().system('pip install matplotlib')

