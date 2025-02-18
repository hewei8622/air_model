import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.geometry import Polygon
import json
import contextily as ctx
from shapely.ops import unary_union

# External modules
import os, sys
import matplotlib.pyplot as plt


# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.models.icestupaClass import Icestupa
from src.utils.settings import config
from src.utils import setup_logger
import logging, coloredlogs

if __name__ == "__main__":

    # Main logger
    logger = logging.getLogger(__name__)
    # logger.setLevel("ERROR")
    logger.setLevel("WARNING")
    region = "Peru"
    # Load the Ladakh boundary GeoJSON
    with open('data/' + region + '/map.geojson', 'r') as f:
        ladakh_geojson = json.load(f)
        
# Convert GeoJSON to GeoDataFrame
    ladakh = gpd.GeoDataFrame.from_features(ladakh_geojson["features"], crs="EPSG:4326")
# Load and process the data
    with open('data/'+region+'/consolidated_results.json', 'r') as f:
        data = json.load(f)

# Create DataFrame from location results
    df = pd.DataFrame(data['location_results'])

# Convert ice volume to lakh litres
    df['iceV_lakh_litres'] = df['iceV_max'] * 1000 / 100000

# Create geometry for points
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# Reproject to Web Mercator for contextily basemap
    ladakh = ladakh.to_crs(epsg=3857)
    gdf = gdf.to_crs(epsg=3857)

# Create figure
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    # Remove axes
    ax.set_axis_off()

# Plot Ladakh boundary
    ladakh.plot(ax=ax, alpha=0.4, color='lightgray')

# Create a custom colormap for ice volume
    scatter = ax.scatter(
        gdf.geometry.x,
        gdf.geometry.y,
        c=gdf['iceV_lakh_litres'],
        cmap='Blues',
        s=200,
        alpha=0.7
    )

    if region == 'Ladakh':
# Add cities
        cities = {
            'Leh': (77.58, 34.17),
            'Kargil': (76.13, 34.57),
            'Pangong Lake': (78.67, 33.83),
            'Nubra Valley': (77.27, 34.62),
            'Zanskar': (76.83, 33.72)
        }
    elif region =='Peru':
        cities = {
            'Cusco': (-71.98, -13.52),
            'Huaraz': (-77.53, -9.53),
            'Puno': (-70.02, -15.84),
            'Cerro de Pasco': (-76.27, -10.69),
            'Juliaca': (-70.13, -15.50)
        }

# Convert city coordinates to Web Mercator and plot
    for city, coords in cities.items():
        point = gpd.GeoDataFrame(
            geometry=[Point(coords[0], coords[1])], 
            crs="EPSG:4326"
        ).to_crs(epsg=3857)
        ax.plot(point.geometry.x, point.geometry.y, 'ro', markersize=8)
        ax.annotate(
            city,
            (point.geometry.x.iloc[0], point.geometry.y.iloc[0]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            color='black',
            weight='bold'
        )

# Add value labels
    for idx, row in gdf.iterrows():
        ax.annotate(
            f'{row.iceV_lakh_litres:.0f}',
            (row.geometry.x, row.geometry.y),
            xytext=(0, -20),
            textcoords="offset points",
            ha='center',
            fontsize=8,
            color='black',
            weight='bold'
        )

# Add colorbar
    plt.colorbar(scatter, label='Ice Volume (Lakh Litres)')

# # Add basemap
#     ctx.add_basemap(ax, source=ctx.providers.OpenTopoMap)
# Add basemap using CartoDB Positron (light, minimal style)
    ctx.add_basemap(ax, 
                    source=ctx.providers.CartoDB.Positron,
                    zoom=8)

# Set title and labels
    plt.title('Ice Volume Potential in '+ region, pad=20, fontsize=14)

# Adjust layout
    plt.tight_layout()

# Save the map
    plt.savefig('data/'+region+'/'+region+'.png', dpi=300, bbox_inches='tight')
    plt.close()
