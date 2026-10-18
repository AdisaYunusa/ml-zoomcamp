#!/usr/bin/env python
# coding: utf-8

# # Week 5

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import math
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

get_ipython().system('pip install tqdm')
from tqdm.auto import tqdm


# In[2]:


get_ipython().run_cell_magic('script', 'true # Skips the cell to prevent repeated dataset downloads.', "\ndata = 'https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv'\n\n!wget $data\n")


# In[3]:


df_raw = pd.read_csv('course_lead_scoring_2026.csv')


# In[4]:


df = df_raw.copy()
df


# ## Data Preparation

# In[5]:


df.isna().sum()


# In[6]:


df.dtypes


# In[7]:


categorical_features = list(df.dtypes[df.dtypes == 'str'].index)
categorical_features


# In[8]:


numerical_features = list(df.dtypes[(df.dtypes != 'str') & (df.dtypes.index != 'converted')].index)
numerical_features


# In[9]:


df[categorical_features] = df[categorical_features].fillna('NA')

df[numerical_features] = df[numerical_features].fillna(0.0)


# In[10]:


df.isna().sum()


# ### Data Split

# In[11]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


# In[12]:


y_train = df_full_train.converted.values
y_test = df_test.converted.values


# In[13]:


del df_full_train['converted']


# ## Methods for Training and Validation

# In[14]:


def train(df, y, C=1.0):
    cat = df[categorical_features + numerical_features].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(cat)

    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
    model.fit(X, y)

    return dv, model


# In[21]:


def predict(df, dv, model):
    cat = df[categorical_features + numerical_features].to_dict(orient='records')

    X = dv.transform(cat)

    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# ## Training The Model (Deployment Exercise)

# In[16]:


c = 0.001


# In[17]:


dv_deploy, model_deploy = train(df_full_train, y_train, C=c)


# In[16]:


output_file = 'model.bin'


# In[19]:


get_ipython().run_cell_magic('script', 'true', "\n# f_out = open(output_file, 'wb')\n# pickle.dump((dv_deploy, model_deploy), f_out)\n# f_out.close()\n\n# f_in = open(output_file, 'rb')\n# model = pickle.load(f_in)\n# f_in.close()\n")


# In[20]:


# Saving the model.

with open(output_file, 'wb') as model_file:
    pickle.dump((dv_deploy, model_deploy), model_file)


# In[17]:


# Loading the model.

with open(output_file, 'rb') as model_file:
    (dv_deploy, model_deploy) = pickle.load(model_file)


# In[18]:


dv_deploy


# In[19]:


model_deploy


# In[22]:


y_pred = predict(df_test, dv_deploy, model_deploy)


# In[23]:


auc_deploy= round(roc_auc_score(y_test, y_pred), 3)
print(f"The AUC for this model on the validation dataset is {auc_deploy}.")


# In[24]:


t = 0.5
tp = ((y_pred >= t) & (y_test == 1)).sum()
fp = ((y_pred >= t) & (y_test == 0)).sum()
fn = ((y_pred < t) & (y_test == 1)).sum()
tn = ((y_pred < t) & (y_test == 0)).sum()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
print(f"Precision: {precision}.")
print()
print(f"Recall: {recall}.")


# In[ ]:




