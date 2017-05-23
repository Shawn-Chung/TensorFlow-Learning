import numpy as np
import matplotlib.pyplot as plt
import urllib
from mpl_toolkits import Basemap

feed = "http://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/"

#significant earthquakes in the last 30 days
url = urllib.request.urlopen(feed + "significant_month.csv")

#Magnitude > 4.5
url = urllib.request.urlopen(feed + "4.5_month.csv")

#Magnitude > 2.5
url = urllib.request.urlopen(feed + "2.5_month.csv")

#Magnitude > 1.0
url = urllib.request.urlopen(feed + "1.0_month.csv")

#reading and storage of data
data = url.read()
data = data.split(b'\n')[+1:-1]
E = np.zeros(len(data), dtype=[('position', float, 2),
                               ('magnitude', float, 1)])
for i in range(len(data)):
    row = data[i].split(',')
    E['position'][i] = float(row[2]),float(row[1])
    E['magnitude'][i] = float(row[4])