#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle


# In[2]:


data = pd.read_csv("C:/Users/Reshma/Downloads/BODY_FITNESS_DATASET.csv")
data


# In[4]:


from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
lb = LabelEncoder()
data['bool_of_active'] = lb.fit_transform(data['bool_of_active'])
data1 =data
data1


# In[5]:


y =data1.iloc[:,5]
y


# In[6]:


data.drop('weight_kg', axis=1, inplace=True)
data


# In[7]:


data.drop('date', axis=1, inplace=True)
data


# In[8]:


data.drop('bool_of_active', axis=1, inplace=True)
data


# In[9]:


data.drop('step_count', axis=1, inplace=True)
data


# In[10]:


from sklearn.compose import ColumnTransformer 
ct = ColumnTransformer([("mood", OneHotEncoder(),[0])], remainder="passthrough") # The last arg ([0]) is the list of columns you want to transform in this step
x=ct.fit_transform(data)
x 


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)


# In[12]:


from sklearn.naive_bayes import GaussianNB
nb =GaussianNB()
model =nb.fit(X_train,y_train)


# In[15]:


# Saving model to disk
pickle.dump(nb, open('model.pkl','wb'))


# In[17]:


# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
z = (model.predict([[0, 1, 0,181,5]]))
if z == 0:
    print ("unfit")
else:
    print ("fit")

