#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests


# In[ ]:


url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Sad':0, 'Nuetral':0, 'Happy':1, 'calories':4, 'hours_of_sleep':2 })

print(r.json())

