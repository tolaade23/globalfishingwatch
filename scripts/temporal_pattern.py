#!/usr/bin/env python
# coding: utf-8

# In[2]:


## 4b: Temporal Pattern Recognition

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import os


# In[3]:


# Load the combined data
base_dir = os.path.join(os.path.expanduser("~"), 'Downloads', 'globalfishingwatch')
data = pd.read_csv(os.path.join(base_dir, 'data', 'combined_data.csv'))



# In[4]:


# Decompose the time series
data['event_start'] = pd.to_datetime(data['event_start'])
data.set_index('event_start', inplace=True)
result = seasonal_decompose(data['duration'], model='multiplicative', period=30)


# In[5]:


# Plot the decomposed components
result.plot()
plt.show()

