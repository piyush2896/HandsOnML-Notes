import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(tarfile_name, housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, tarfile_name)
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

if not os.path.isfile(os.path.join(HOUSING_PATH, "housing.csv")):
    fetch_housing_data("housing.tgz")

housing = pd.read_csv('./datasets/housing.csv')

pop = housing['population'].values
mhv = housing['median_house_value'].values
lon = housing['longitude'].values
lat = housing['latitude'].values

fig = plt.figure(figsize=(8,8))
m = Basemap(lon_0=-119.6564896,
            lat_0=36.6477949,
            llcrnrlat=min(lat),
            urcrnrlat=max(lat),
            llcrnrlon=min(lon),
            urcrnrlon=max(lon),
            projection='stere',
            resolution='l')

m.drawcoastlines()
m.drawcountries(linewidth=2)
m.drawstates(color='black')
m.drawrivers()

s = pop / 100
c = mhv
cmap=plt.get_cmap("jet")
x, y = m(lon, lat)
sc = m.scatter(x, y, marker='o', s=s, c=c, cmap=cmap, label='Population')

tick_values = np.linspace(mhv.min(), mhv.max(), 11)
cbar = plt.colorbar()
cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
cbar.set_label('Median House Value', fontsize=16)

plt.legend(fontsize=16)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
