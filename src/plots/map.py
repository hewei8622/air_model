import geopandas as gpd
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use non-GUI backend
import numpy as np
from shapely.geometry import Point
import json
import contextily as ctx
import os
import sys
import logging
import argparse

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.utils import setup_logger
import logging, coloredlogs

def get_country_geometry(country, use_geojson=False):
    """
    Retrieve country geometry from either Natural Earth Data or a local GeoJSON file.
    
    Parameters:
    -----------
    country : str
        Name of the country
    use_geojson : bool, optional
        Flag to prefer local GeoJSON over Natural Earth Data
        
    Returns:
    --------
    GeoDataFrame: Country geometry or empty GeoDataFrame if not found
    """
    logger = logging.getLogger(__name__)
    
    # Option 1: Try local GeoJSON if use_geojson is True or Natural Earth fails
    if use_geojson:
        geojson_path = os.path.join('data', country, f'{country}.geojson')
        if os.path.exists(geojson_path):
            try:
                logger.info(f"Loading country geometry from local GeoJSON: {geojson_path}")
                country_gdf = gpd.read_file(geojson_path)
                if not country_gdf.empty:
                    return country_gdf
            except Exception as e:
                logger.warning(f"Error reading local GeoJSON: {e}")
    
    # Option 2: Try Natural Earth Data
    try:
        country_gdf = gpd.read_file('data/ne_110m_admin_0_countries.zip', 
                                    engine='pyogrio', 
                                    use_arrow=True,
                                    where=f"ADMIN = '{country}'")
        
        if not country_gdf.empty:
            logger.info(f"Found country in Natural Earth Data: {country}")
            return country_gdf
    except Exception as e:
        logger.error(f"Error retrieving country from Natural Earth Data: {e}")
    
    # Fallback: Check map.geojson in country folder
    geojson_path = os.path.join('data', country, 'map.geojson')
    if os.path.exists(geojson_path):
        try:
            logger.info(f"Loading country geometry from map.geojson: {geojson_path}")
            country_gdf = gpd.read_file(geojson_path)
            if not country_gdf.empty:
                return country_gdf
        except Exception as e:
            logger.warning(f"Error reading map.geojson: {e}")
    
    logger.error(f"Country '{country}' not found in any source")
    return gpd.GeoDataFrame()

def get_cities_from_dataset(region, country_gdf, max_cities=10):
    """
    Extract cities for the specified region by manually loading and filtering the populated places dataset.
    Uses the 'name' column for city names with appropriate fallbacks.
    
    Parameters:
    -----------
    region : str
        The name of the region/country
    country_gdf : GeoDataFrame
        The GeoDataFrame containing the country/region to intersect with
    max_cities : int
        Maximum number of cities to return
        
    Returns:
    --------
    dict: Dictionary with city names as keys and (longitude, latitude) tuples as values
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Load the populated places dataset
        cities_gdf = gpd.read_file('data/ne_110m_populated_places_simple.zip')
        logger.info(f"Loaded {len(cities_gdf)} cities from dataset")
        
        # Make sure country_gdf is valid
        if country_gdf.empty:
            logger.error("Country GeoDataFrame is empty")
            return {}
            
        # Make sure we're working in the correct CRS for spatial operations
        if cities_gdf.crs != country_gdf.crs:
            logger.debug(f"Converting cities from {cities_gdf.crs} to {country_gdf.crs}")
            cities_gdf = cities_gdf.to_crs(country_gdf.crs)
        
        # Create a unified country geometry for more efficient operations
        country_shape = country_gdf.unary_union
        logger.debug("Created unified country geometry for spatial filtering")
        
        # Now find cities within the country boundary
        logger.debug("Filtering cities by spatial relationship with country boundary")
        within_cities = cities_gdf[cities_gdf.geometry.within(country_shape)]
        
        if len(within_cities) > 0:
            logger.info(f"Found {len(within_cities)} cities within country boundaries")
            filtered_cities = within_cities
        else:
            logger.warning("No cities found within country boundaries, trying with intersects")
            intersect_cities = cities_gdf[cities_gdf.geometry.intersects(country_shape)]
            if len(intersect_cities) > 0:
                logger.info(f"Found {len(intersect_cities)} cities intersecting country boundaries")
                filtered_cities = intersect_cities
            else:
                # Try a bounding box approach
                logger.warning("No cities found with spatial operations, trying bounding box")
                # Get the bounding box of the country
                minx, miny, maxx, maxy = country_gdf.total_bounds
                logger.info(f"Country bounding box: {minx}, {miny}, {maxx}, {maxy}")
                
                # Convert cities to same CRS as country
                cities_bbox = cities_gdf.copy()
                
                # Filter by bounding box
                bbox_cities = cities_bbox[
                    (cities_bbox.geometry.x >= minx) & 
                    (cities_bbox.geometry.x <= maxx) & 
                    (cities_bbox.geometry.y >= miny) & 
                    (cities_bbox.geometry.y <= maxy)
                ]
                
                if len(bbox_cities) > 0:
                    logger.info(f"Found {len(bbox_cities)} cities within country bounding box")
                    filtered_cities = bbox_cities
                else:
                    logger.error(f"No cities found for {region} using any method")
                    return {}
        
        # Sort by population if available
        population_columns = ['pop_max', 'pop_min', 'pop_other']
        for pop_col in population_columns:
            if pop_col in filtered_cities.columns:
                logger.debug(f"Sorting cities by {pop_col}")
                filtered_cities = filtered_cities.sort_values(pop_col, ascending=False)
                break
        
        # Limit to max_cities
        filtered_cities = filtered_cities.head(max_cities)
        
        # Create dictionary with city names and coordinates
        cities_dict = {}
        
        # Check if 'name' column exists
        if 'name' not in filtered_cities.columns:
            logger.error("'name' column not found in dataset")
            return {}
            
        # Extract city names and coordinates
        for _, city in filtered_cities.iterrows():
            # Get the coordinates in EPSG:4326 (lat/lon)
            point_gdf = gpd.GeoDataFrame(geometry=[city.geometry], crs=filtered_cities.crs)
            point_4326 = point_gdf.to_crs(epsg=4326)
            coords = (point_4326.geometry.x.iloc[0], point_4326.geometry.y.iloc[0])
            
            # Use the 'name' column for city names
            city_name = str(city['name'])
            cities_dict[city_name] = coords
            logger.info(f"Found city: {city_name} at coordinates: {coords}")
        
        logger.info(f"Extracted {len(cities_dict)} cities for {region}: {list(cities_dict.keys())}")
        return cities_dict
        
    except Exception as e:
        logger.error(f"Error extracting cities: {str(e)}")
        return {}

def plot_icestupa_map(country, output_path=None, max_cities=10):
    """
    Generate and save a map of ice volume potential for the specified country.
    
    Parameters:
    -----------
    country : str
        The name of the country to map
    output_path : str, optional
        Path to save the output image. If None, uses default path.
    max_cities : int, optional
        Maximum number of cities to display on the map (default: 10)
    """
    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel("INFO")
    
    logger.info(f"\nMapping {country}")
    
    # Load country boundaries with fallback to local sources
    country_gdf = get_country_geometry(country)
    
    if country_gdf.empty:
        # Try with GeoJSON source if Natural Earth Data fails
        country_gdf = get_country_geometry(country, use_geojson=True)
        
        if country_gdf.empty:
            logger.error(f"Country '{country}' not found in any data source")
            return False
    
    # Load and process the data
    try:
        with open(f'data/{country}/consolidated_results.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Results file not found for {country}")
        return False

    # Determine the structure of the data and load location results
    if isinstance(data, list):
        # If data is a list, use it directly
        df = pd.DataFrame(data)
    elif isinstance(data, dict):
        # If data is a dictionary, try to get location results
        if 'location_results' in data:
            df = pd.DataFrame(data['location_results'])
        else:
            # If no location_results, try to find a key that contains list of dictionaries
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                    df = pd.DataFrame(value)
                    break
            else:
                logger.error(f"Could not find suitable location data in JSON for {country}")
                return False
    else:
        logger.error(f"Unexpected data type in JSON for {country}")
        return False

    # Ensure required columns exist
    required_columns = ['iceV_max', 'longitude', 'latitude']
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Missing required column: {col}")
            return False

    # Convert ice volume to million liters
    df['iceV_litres'] = df['iceV_max'] * 1000 / 1000000

    # Create geometry for points
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Reproject to Web Mercator for contextily basemap
    country_gdf = country_gdf.to_crs(epsg=3857)
    gdf = gdf.to_crs(epsg=3857)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.set_axis_off()

    # Plot country boundary
    country_gdf.plot(ax=ax, alpha=0.4, color='lightgray')

    # Create a custom colormap for ice volume
    scatter = ax.scatter(
        gdf.geometry.x,
        gdf.geometry.y,
        c=gdf['iceV_litres'],
        cmap='Blues',
        s=200,
        alpha=0.7
    )

    # Get cities by intersecting with country geometry
    cities = get_cities_from_dataset(country, country_gdf, max_cities)
    
    # Plot cities - make them more visible
    for city_name, coords in cities.items():
        logger.debug(f"Plotting city: {city_name} at {coords}")
        
        # Convert to Web Mercator for display
        point = gpd.GeoDataFrame(
            geometry=[Point(coords[0], coords[1])], 
            crs="EPSG:4326"
        ).to_crs(epsg=3857)
        
        # Larger, more visible markers
        ax.plot(point.geometry.x, point.geometry.y, 'ro', markersize=10, 
                markeredgecolor='black', markeredgewidth=1.5)
        
        # Clearer labels with background
        ax.annotate(
            city_name,
            (point.geometry.x.iloc[0], point.geometry.y.iloc[0]),
            xytext=(7, 7),  # Offset text slightly more
            textcoords="offset points",
            fontsize=12,  # Larger font
            color='black',
            weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8)
        )

    # Add value labels for ice volumes
    for idx, row in gdf.iterrows():
        ax.annotate(
            f'{row.iceV_litres:.0f}',
            (row.geometry.x, row.geometry.y),
            xytext=(0, -20),
            textcoords="offset points",
            ha='center',
            fontsize=10,  # Slightly larger font
            color='black',
            weight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
        )

    # Add colorbar
    plt.colorbar(scatter, label='Ice Volume (Million Litres)')

    # Add basemap using CartoDB Positron (light, minimal style)
    ctx.add_basemap(ax, 
                   source=ctx.providers.CartoDB.Positron,
                   zoom=8)

    # Set title
    plt.title(f'Ice Volume Potential in {country}', pad=20, fontsize=16)

    # Adjust layout
    plt.tight_layout()

    # Save the map
    if output_path is None:
        output_path = f'data/{country}/{country}.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Map saved to {output_path}")
    
    # Also display city list in console for reference
    print(f"\nCities shown on the map for {country}:")
    for city_name in cities.keys():
        print(f"  - {city_name}")
    
    plt.close()
    return True

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate ice volume potential maps for different countries")
    
    parser.add_argument("--country", 
                        required=False, 
                        default="Tajikistan",
                        help="Country to map (e.g., Tajikistan, Peru, India)")
    
    parser.add_argument("--output", 
                        required=False, 
                        help="Output file path for the map")
    
    parser.add_argument("--max-cities",
                        type=int,
                        default=10,
                        help="Maximum number of cities to display on the map")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel("INFO")
    
    # Parse command line arguments
    args = parse_args()
    
    # Add a command line parameter for the max number of cities
    logger.info(f"Mapping country: {args.country}")
    logger.info(f"Maximum cities to show: {args.max_cities}")
    
    # Update function call to include max_cities parameter
    plot_icestupa_map(args.country, args.output, args.max_cities)
