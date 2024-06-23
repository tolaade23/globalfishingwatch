#!/usr/bin/env python
# coding: utf-8

# In[17]:


## Step 3: Clustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os
import pandas as pd
import warnings


# In[18]:


# Load the combined data
base_dir = os.path.join(os.path.expanduser("~"), 'Downloads', 'globalfishingwatch')
data = pd.read_csv(os.path.join(base_dir, 'data', 'combined_data.csv'))


# In[19]:


# Standardize features
scaler = joblib.load(os.path.join(base_dir, 'models', 'scaler.pkl'))
X_scaled = scaler.transform(data.drop(columns=['event_start']))


# In[20]:


# Perform K-Means Clustering
n_clusters = 10
# The environment variable to match n_init
os.environ["LOKY_MAX_CPU_COUNT"] = str(n_clusters)
os.environ["OMP_NUM_THREADS"] = str(n_clusters)

warnings.filterwarnings("ignore", category=FutureWarning)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data['cluster'] = kmeans.fit_predict(X_scaled)


# In[21]:


# Save the clustered data including 'cluster' column
clustered_data_path = os.path.join(base_dir, 'data', 'clustered_data.csv')
data.to_csv(clustered_data_path, index=False)


# In[22]:


# Save the model
model_path = os.path.join(base_dir, 'models', 'kmeans_model.pkl')
joblib.dump(kmeans, model_path)


# In[ ]:




