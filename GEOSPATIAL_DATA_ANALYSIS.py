"""
Faye

"""

import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from shapely.geometry import Point # Shapely for converting latitude/longtitude to geometry
import geopandas
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys

import cartopy
import cartopy.crs as ccrs

from itertools import combinations
from datetime import datetime
import random
import warnings
import cartopy.crs as crs
import cartopy.feature as cfeature

import matplotlib.pyplot as plt
import seaborn as sns 
import csv
import plotly.express as px
from iso3166 import countries

data = pd.read_csv("C:/Users/Faye/Desktop/Master/data_geo_dissertation.csv")
print(data)

from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="https://www.google.com/maps")
countries = ["GR", "NL","IT", "FR","RO","PRT","ES", "DE", "AT", "FI", "NO", "LT", "IE", "BE","RS", "EU" ]

for country in countries:
    location = geolocator.geocode(country)
    print(location.latitude, location.longitude)


df = pd.DataFrame(
    {
        "Latitude": [
            38.9953683,
            52.5001698,
            42.6384261,
            46.603354,
            45.9852129,
            39.3262345,
            51.0834196,
            47.2000338,
            63.2467777,
            60.5000209,
            55.3500003,
            44.1534121,
            50.6402809,
        ],
        "Longitude": [
            21.9877132,
            5.7480821,
            12.674297,
            1.8883335,
            24.6859225,
            -4.8380649,
            10.4234469,
            13.199959,
            25.9209164,
            9.0999715,
            23.7499997,
            20.55144,
            4.6667145,
        ],
    }
)
print(df)


# creating a geometry column 
gdf = geopandas.GeoDataFrame(
    df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude))

print(gdf)

world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

gdf.plot()

ax = gdf.plot(color='r', zorder=2)
world.plot(ax=ax, zorder=1)

results_per_country_lon_lat="C:/Users/Faye/Desktop/Master/test2.xlsx"
map1=pd.read_excel(results_per_country_lon_lat) #, names=['Codes', 'Total', 'Latitude', 'Longitude'])
print(map1)
type(map1)

results = map1.groupby(by="S").count()[["Total"]].rename(columns={"Total":"Count"})
print(results)

fig = plt.figure(figsize=(12,10))

ax = fig.add_subplot(1,1,1, projection=crs.PlateCarree())

ax.stock_img()
ax.coastlines()
ax.add_feature(cfeature.BORDERS)

ax.set_extent([-10.67,34.5,31.55,71.05],
              crs=crs.PlateCarree()) ## Important


plt.scatter(x=map1.Longitude, y=map1.Latitude,
            color="orangered",
            s=results.Count,
            alpha=1,
            transform=crs.PlateCarree()) ## Important

plt.show()

map1.S.unique()

print(countries.get('us'))

def rename(country):
    try:
        return countries.get(country).alpha3
    except:
        return (np.nan)

old_sample_number = map1.S.shape[0]

countriesData = map1.S.apply(rename)
countriesData = map1.S.dropna()

new_sample_number = map1.S.shape[0]
print('we lost', old_sample_number-new_sample_number, 'samples after converting')

print(countriesData)


country_df = pd.DataFrame(data=[countriesData.value_counts().index, countriesData.value_counts().values],index=['country','count']).T


#Converting count values to int because this will be important for plotly
country_df['count']=pd.to_numeric(country_df['count'])

fig = px.scatter_geo(country_df, locations="country", size='count',
                     hover_name="country", color='country',
                     projection="natural earth")
fig.show()
