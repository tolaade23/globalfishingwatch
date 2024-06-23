#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Step 1: Data Preprocessing

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import joblib


# In[2]:


# Specify the base directory
base_dir = os.path.join(os.path.expanduser("~"), 'Downloads', 'globalfishingwatch')

# Load the datasets
ports_data = pd.read_csv(os.path.join(base_dir, 'data', 'CVP_ports_20240614.csv'))
loitering_data = pd.read_csv(os.path.join(base_dir, 'data', 'CVP_loitering_20240614.csv'))
encounters_data = pd.read_csv(os.path.join(base_dir, 'data', 'CVP_encounters_20240614.csv'))


# In[3]:


# Verify that the data contains the 'event_start' and 'event_end' columns
required_columns = ['event_start', 'event_end']
for dataset in [ports_data, loitering_data, encounters_data]:
    for column in required_columns:
        if column not in dataset.columns:
            raise KeyError(f"The required column '{column}' is missing in the dataset")


# In[4]:


# Combine all the datasets
data = pd.concat([ports_data, loitering_data, encounters_data], ignore_index=True)

# Convert event_start and event_end to datetime
data['event_start'] = pd.to_datetime(data['event_start'])
data['event_end'] = pd.to_datetime(data['event_end'])


# In[5]:


# Feature Engineering to calculate the duration of events
data['duration'] = (data['event_end'] - data['event_start']).dt.total_seconds() / 3600.0

# Feature selection for Anomaly detection
features = ['lat_mean', 'lon_mean', 'duration', 'event_type', 'event_start']
data = pd.get_dummies(data[features])


# In[6]:


# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data.drop(columns=['event_start']))


# In[7]:


# Save the scaler
scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')
os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
joblib.dump(scaler, scaler_path)


# In[9]:


# Save the combined data
combined_data_path = os.path.join(base_dir, 'data', 'combined_data.csv')
os.makedirs(os.path.dirname(combined_data_path), exist_ok=True)
data.to_csv(combined_data_path, index=False)


# In[ ]:




