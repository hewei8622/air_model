import os
import glob
import shutil
import geopandas as gpd
from shapely.geometry import Point
import argparse
import logging
import sys
import re

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.utils import setup_logger
import logging, coloredlogs

def sanitize_filename(filename):
    """
    Sanitize filename by removing special characters and replacing spaces
    
    Args:
        filename (str): Original filename
    
    Returns:
        str: Sanitized filename safe for filesystem
    """
    # Remove or replace special characters
    # Replace spaces with underscores
    # Remove quotes
    sanitized = re.sub(r'[^\w\-_\. ]', '_', filename)
    sanitized = sanitized.replace(' ', '_')
    sanitized = sanitized.replace("'", "")
    sanitized = sanitized.replace('"', "")
    
    return sanitized.strip()

def parse_coordinates(filename):
    """
    Parse coordinates from filename in format lat_lon_alt.csv
    Returns (lat, lon, alt) tuple or None if parsing fails
    """
    # Remove .csv extension and split by underscore
    coords = os.path.splitext(filename)[0].split('_')
    try:
        lat = float(coords[0])
        lon = float(coords[1])
        alt = float(coords[2])
        return lat, lon, alt
    except:
        return None

def load_countries():
    """
    Load country boundaries from Natural Earth dataset
    Returns GeoDataFrame with country boundaries
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Check if the shape file exists
        shapefile_path = 'data/ne_110m_admin_0_countries.zip'
        if not os.path.exists(shapefile_path):
            logger.error(f"Country shapefile not found at {shapefile_path}")
            logger.info("Please download from https://www.naturalearthdata.com/downloads/110m-cultural-vectors/")
            return None
            
        # Load country boundaries - adjust path as needed
        countries = gpd.read_file(shapefile_path)
        logger.info(f"Loaded {len(countries)} countries from shapefile")
        return countries
    except Exception as e:
        logger.error(f"Error loading country boundaries: {str(e)}")
        return None

def classify_by_country(input_folder, output_base_folder, min_alt=None, bounding_box=None):
    """
    Classify CSV files by country and copy to country-specific folders
    
    Parameters:
    -----------
    input_folder : str
        Folder containing CSV files with lat_lon_alt.csv naming format
    output_base_folder : str
        Base folder where country-specific folders will be created
    min_alt : float, optional
        Minimum altitude filter
    bounding_box : tuple, optional
        Bounding box filter (min_lat, min_lon, max_lat, max_lon)
    """
    logger = logging.getLogger(__name__)
    
    # Create output base folder if it doesn't exist
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)
        logger.info(f"Created output base folder: {output_base_folder}")
    
    # Load countries
    countries_gdf = load_countries()
    if countries_gdf is None:
        logger.error("Failed to load country boundaries. Exiting.")
        return
    
    # Get list of all CSV files in input folder
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))
    logger.info(f"Found {len(csv_files)} CSV files in {input_folder}")
    
    # Statistics counters
    files_processed = 0
    files_classified = 0
    country_counts = {}
    
    # Process each file
    for file in csv_files:
        try:
            filename = os.path.basename(file)
            coords = parse_coordinates(filename)
            
            if coords:
                lat, lon, alt = coords
                
                # Apply altitude filter if specified
                if min_alt is not None and alt < min_alt:
                    logger.debug(f"Skipping {filename} - altitude {alt} below minimum {min_alt}")
                    continue
                
                # Apply bounding box filter if specified
                if bounding_box is not None:
                    min_lat, min_lon, max_lat, max_lon = bounding_box
                    if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
                        logger.debug(f"Skipping {filename} - outside bounding box")
                        continue
                
                # Create point geometry
                point = Point(lon, lat)  # Note: GIS standard is (longitude, latitude)
                point_gdf = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326")
                
                # Find which country the point belongs to
                for idx, country in countries_gdf.iterrows():
                    if point_gdf.within(country.geometry).values[0]:
                        country_name = country['ADMIN']  # Use ADMIN field for country name
                        
                        # Sanitize country name to remove quotes and special characters
                        sanitized_country_name = sanitize_filename(country_name)
                        
                        # Create country folder and era5 subfolder if they don't exist
                        country_folder = os.path.join(output_base_folder, sanitized_country_name)
                        era5_folder = os.path.join(country_folder, "era5")
                        
                        if not os.path.exists(era5_folder):
                            os.makedirs(era5_folder)
                            logger.info(f"Created era5 folder for {sanitized_country_name}")
                        
                        # Copy file to country/era5 folder
                        output_path = os.path.join(era5_folder, filename)
                        shutil.copy2(file, output_path)
                        
                        # Update statistics
                        files_classified += 1
                        country_counts[sanitized_country_name] = country_counts.get(sanitized_country_name, 0) + 1
                        
                        logger.debug(f"Classified {filename} as {sanitized_country_name}")
                        break
                else:
                    logger.warning(f"Could not classify {filename} to any country")
            
            files_processed += 1
            if files_processed % 100 == 0:
                logger.info(f"Processed {files_processed}/{len(csv_files)} files")
                
        except Exception as e:
            logger.error(f"Error processing {file}: {str(e)}")
    
    # Log summary statistics
    logger.info(f"\nProcessing complete:")
    logger.info(f"Total files processed: {files_processed}")
    logger.info(f"Files classified to countries: {files_classified}")
    logger.info(f"Country statistics:")
    for country, count in sorted(country_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  - {country}: {count} files")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Classify weather data CSV files by country")
    
    parser.add_argument("--input", 
                        required=True,
                        help="Input folder containing CSV files named as lat_lon_alt.csv")
    
    parser.add_argument("--output", 
                        required=True,
                        help="Output base folder where country folders will be created")
    
    parser.add_argument("--min-alt",
                        type=float,
                        help="Minimum altitude filter (optional)")
    
    parser.add_argument("--bbox",
                        nargs=4,
                        type=float,
                        metavar=('MIN_LAT', 'MIN_LON', 'MAX_LAT', 'MAX_LON'),
                        help="Bounding box filter: min_lat min_lon max_lat max_lon (optional)")
    
    parser.add_argument("--log-level",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO",
                        help="Logging level")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Set up logger
    logger = logging.getLogger(__name__)
    logger.setLevel("INFO")
    
    # Run classification
    classify_by_country(
        input_folder=args.input,
        output_base_folder=args.output,
        min_alt=args.min_alt,
        bounding_box=args.bbox
    )
