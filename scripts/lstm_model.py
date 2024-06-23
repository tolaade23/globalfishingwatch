#!/usr/bin/env python
# coding: utf-8

# In[4]:


## 4a: LSTM Model for Vessel Movement Prediction
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from datetime import datetime
import os


# In[5]:


# Load the combined data
base_dir = os.path.join(os.path.expanduser("~"), 'Downloads', 'globalfishingwatch')
data = pd.read_csv(os.path.join(base_dir, 'data', 'combined_data.csv'))


# In[6]:


# Prepare the data
data['event_start'] = pd.to_datetime(data['event_start'])
data = data.sort_values('event_start')
data['timestamp'] = data['event_start'].map(datetime.timestamp)



# In[7]:


# Features and target for prediction
features = ['lat_mean', 'lon_mean', 'timestamp']
target = ['lat_mean', 'lon_mean']


# In[8]:


# Create sequences
def create_sequences(data, seq_length=10):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[features].iloc[i:(i + seq_length)].values
        y = data[target].iloc[i + seq_length].values
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X, y = create_sequences(data)


# In[9]:


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[10]:


# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50))
model.add(Dense(2))

model.compile(optimizer='adam', loss='mse')


# In[11]:


# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1)

# Save the model
model_path = os.path.join(base_dir, 'models', 'vessel_movement_model.h5')
model.save(model_path)


# In[ ]:




