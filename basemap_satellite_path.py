from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

blob_df = pd.read_csv(r'C:\Data/blob_data_2532019.csv')

#print(blob_df.head())

fig = plt.figure(figsize=(6,5))

m = Basemap(projection='mill',
            llcrnrlat=4,
            urcrnrlat=14,
            llcrnrlon=2,
            urcrnrlon=15)

m.drawcoastlines()
m.drawcountries(linewidth=3)
m.drawparallels(np.arange(2,16,2), labels=[True,False,False,False])
m.drawmeridians(np.arange(2,16,2), labels=[False,False,False,True])

sites_lat_y = blob_df['Latitude'].tolist()
sites_lon_x = blob_df['Longitude'].tolist()

# NASRDA lat and long
nasrda_lat = 8.9896
nasrda_lon = 7.3869
xpt1, ypt1 = m(nasrda_lon, nasrda_lat)

# VT-NigerBEAR lat and long
bowen_lat = 7.6236
bowen_lon = 4.1890
xpt2, ypt2 = m(bowen_lon, bowen_lat)

# SWARM Passage
m.scatter(sites_lon_x,sites_lat_y, latlon=True, color='blue')
plt.annotate('$\\bf{SWARM - A}$', m(7.8,10.3), xytext=m(10,11),
             arrowprops=dict(color='blue', width=1))

# NASRDA location
m.plot(xpt1,ypt1, 'c*', markersize=8, color='red')
plt.text(xpt1, ypt1, '$\\bf{NASRDA}$', fontsize=8)

# Bowen University
m.plot(xpt2,ypt2, 'c*', markersize=8, color='green')
plt.text(xpt2, ypt2, '$\\bf{VT-NigerBEAR}$', fontsize=8)
plt.title("SWARM - A Spacecraft passing over Nigeria | 25/3/2019")
plt.savefig(r'C:/path_2532019.png',
            dpi=200, bbox_inches='tight')
plt.show()
