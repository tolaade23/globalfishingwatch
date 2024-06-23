#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Step 2: Anomaly Detection

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib
import os


# In[2]:


# Load the combined data
base_dir = os.path.join(os.path.expanduser("~"), 'Downloads', 'globalfishingwatch')
data = pd.read_csv(os.path.join(base_dir, 'data', 'combined_data.csv'))


# In[3]:


# Standardize features
scaler = joblib.load(os.path.join(base_dir, 'models', 'scaler.pkl'))
X_scaled = scaler.transform(data.drop(columns=['event_start']))


# In[4]:


# Initialize and fit Isolation Forest
model = IsolationForest(contamination=0.01, random_state=42)
model.fit(X_scaled)


# In[5]:


# Predict anomalies
data['anomaly'] = model.predict(X_scaled)
data['anomaly'] = data['anomaly'].map({1: 0, -1: 1})


# In[6]:


# Save the model
model_path = os.path.join(base_dir, 'models', 'anomaly_detection_model.pkl')
joblib.dump(model, model_path)


# In[7]:


# Analyze anomalies
anomalies = data[data['anomaly'] == 1]
print(anomalies)


# In[ ]:





# In[ ]:




