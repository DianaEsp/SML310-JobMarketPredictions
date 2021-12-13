#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn
import sklearn.datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


# read input csv
degrees = pd.read_csv('degrees-that-pay-back.csv')
salaries_college = pd.read_csv('salaries-by-college-type.csv')
salaries_region = pd.read_csv('salaries-by-region.csv')


# In[4]:


degrees.head()


# In[5]:


salaries_college.head()


# In[8]:


print(degrees.min())
print(degrees.max())


# In[15]:


degrees[degrees['Percent change from Starting to Mid-Career Salary'] > 90]


# In[28]:


# https://stackoverflow.com/questions/32464280/converting-currency-with-to-numbers-in-python-pandas
salaries_college[salaries_college.columns[2:]] = salaries_college[salaries_college.columns[2:]].replace('[\$,]', '', regex=True).astype(float)
salaries_college[salaries_college['Mid-Career Median Salary'] > 110000]


# In[27]:


salaries_college[salaries_college['Starting Median Salary'] > 60000]


# In[ ]:




