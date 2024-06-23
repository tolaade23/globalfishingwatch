#!/usr/bin/env python
# coding: utf-8

# In[1]:


## 4c: Geospatial Analysis and Visualization

import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
import os


# In[2]:


# Load the combined data
base_dir = os.path.join(os.path.expanduser("~"), 'Downloads', 'globalfishingwatch')
data = pd.read_csv(os.path.join(base_dir, 'data', 'combined_data.csv'))


# In[3]:


# Create a GeoDataFrame
gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.lon_mean, data.lat_mean))


# In[4]:


# Plot the heatmap
m = folium.Map([gdf.lat_mean.mean(), gdf.lon_mean.mean()], zoom_start=2)
heat_data = [[row['lat_mean'], row['lon_mean']] for index, row in gdf.iterrows()]
HeatMap(heat_data).add_to(m)
m.save(os.path.join(base_dir, 'data', 'heatmap.html'))

