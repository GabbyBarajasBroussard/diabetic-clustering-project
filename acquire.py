#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os


# In[ ]:


def get_diabetic_data():
    df= pd.read_csv('diabetic_data.csv', index_col=0)
    return df


# In[ ]:


def get_id_data():
    df2= pd.read_csv('IDs_mapping.csv', index_col=0)
    return df2
