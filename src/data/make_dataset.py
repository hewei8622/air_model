"""Compile raw data from the location, meteoswiss or ERA5
"""

# External modules
import sys, os, json, argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from scipy import stats
import multiprocessing as mp
from functools import partial

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.utils.settings import config
from src.data.field import get_field
from src.data.era5 import get_era5
from src.data.meteoswiss import get_meteoswiss
from src.plots.data import plot_input
from src.utils import setup_logger

def e_sat(T, surface="water", a1=611.21, a3=17.502, a4=32.19):
    T += 273.16
    if surface == "ice":
        a1 = 611.21  # Pa
        a3 = 22.587  # NA
        a4 = -0.7  # K
    return a1 * np.exp(a3 * (T - 273.16) / (T - a4))

def parse_args():
    parser = argparse.ArgumentParser(description="Command line interface to create or display Icestupa class")

    parser.add_argument("--start_year", required=False, help="Specify the start year (e.g., 2019)")
    parser.add_argument("--end_year", required=False, help="Specify the end year (e.g., 2020)")
    parser.add_argument("--country", required=True, help="Specify the country name")
    parser.add_argument("--list_countries", action="store_true", help="List all available countries")

    return parser.parse_args()

def get_available_countries():
    """
    Get a list of available countries by looking at directories
    
    Returns:
        list: List of country names (directories)
    """
    data_base_dir = os.path.join(dirname, 'data')
    
    if not os.path.exists(data_base_dir):
        logger.warning(f"Data directory not found at {data_base_dir}")
        return []
    
    # Get all directories in the data folder
    return [d for d in os.listdir(data_base_dir) 
            if os.path.isdir(os.path.join(data_base_dir, d)) and d != "world"]

def list_available_countries():
    """List all available countries by checking existing directories"""
    countries = get_available_countries()
    
    if countries:
        print("\nAvailable countries:")
        for i, country in enumerate(sorted(countries), 1):
            print(f"  {i}. {country}")
        print(f"\nTotal: {len(countries)} countries\n")
    else:
        print("No country directories found in data folder")

def process_datasets_parallel(filenames, args, num_processes=None):
    """
    Process multiple datasets in parallel
    """
    if num_processes is None:
        num_processes = mp.cpu_count()

    logger.info(f"Starting parallel processing with {num_processes} processes for country: {args.country}")
    
    # Create a partial function with fixed args
    process_func = partial(process_single_dataset, args=args)
    
    # Create a process pool and map the work
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_func, filenames)
    
    # Count successful processes
    successful = sum(1 for r in results if r)
    logger.info(f"Processed {successful} out of {len(filenames)} datasets successfully for {args.country}")

def extract_location_details(filename):
    # Strip the file extension if present
    if filename.endswith('.csv'):
        location = filename[:-4]  # Remove .csv extension
    else:
        location = filename

    # Split the string based on underscore
    try:
        lat, long, alt = location.split('_')
        coords = [float(lat), float(long)]
        alt = float(alt)
    except ValueError:
        raise ValueError("Filename should be in the format lat_long_alt.csv (e.g., 34.216_77.606_4009.csv)")

    return coords, alt, location

def get_data_filenames(country=None):
    """Find all CSV files in the data directory for the specified country"""
    if country is None:
        raise ValueError("Country must be specified")
    
    data_base_dir = os.path.join(dirname, 'data')
    country_dir = os.path.join(data_base_dir, country)
    era5_dir = os.path.join(country_dir, 'era5')
    
    # Check if the directory exists
    if not os.path.exists(era5_dir):
        raise FileNotFoundError(f"Era5 directory not found: {era5_dir}")
    
    # Look for CSV files in the data directory
    csv_files = [f for f in os.listdir(era5_dir) if f.endswith('.csv')]
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {era5_dir}")
    
    return csv_files, country_dir

def process_single_dataset(filename, args):
    """Process a single location's dataset"""
    try:
        logger.info(f"\nProcessing {filename}")
        coords, alt, location = extract_location_details(filename)

        # Build the data directory path from country
        country_dir = os.path.join(dirname, 'data', args.country)

        with open("constants.json") as f:
            CONSTANTS = json.load(f)

        SITE, FOLDER = config(location, start_year=args.start_year, end_year=args.end_year, 
                              alt=alt, coords=coords, datadir=country_dir)

        loc = location
        df = pd.read_csv(
            FOLDER["raw"] + loc + ".csv",
            sep=",",
            header=0,
            parse_dates=["time"],
        )

        # Process the dataframe
        df = df.set_index("time")
        df = df[SITE["start_date"]:SITE["expiry_date"]]

        time_steps = 60 * 60 
        df["ssrd"] /= time_steps
        df['wind'] = np.sqrt(df.u10**2 + df.v10**2)
        
        # Derive RH
        df["t2m"] -= 273.15
        df["d2m"] -= 273.15
        df["t2m_RH"] = df["t2m"].copy()
        df["d2m_RH"] = df["d2m"].copy()
        df= df.apply(lambda x: e_sat(x) if x.name == "t2m_RH" else x)
        df= df.apply(lambda x: e_sat(x) if x.name == "d2m_RH" else x)
        df["RH"] = 100 * df["d2m_RH"] / df["t2m_RH"]
        df = df.drop(['u10', 'v10', 't2m_RH', 'd2m_RH', 'd2m'], axis=1)
        df = df.reset_index()

        # CSV output
        df.rename(
            columns={
                "t2m": "temp",
                "ssrd": "SW_global",
            },
            inplace=True,
        )

        df = df.round(3)
        df["Discharge"] = 1000000

        cols = [
            "time",
            "temp",
            "RH",
            "wind",
            # "tcc",
            "SW_global",
            "Discharge",
        ]
        df_out = df[cols]

        if df_out.isna().values.any():
            logger.warning(df_out[cols].isna().sum())
            df_out = df_out.interpolate(method='ffill', axis=0)

        df_out = df_out.round(3)
        if len(df_out[df_out.index.duplicated()]):
            logger.error("Duplicate indexes")

        logger.info(df_out.head())
        logger.info(df_out.tail())
        
        # Create necessary directories
        location_dir = os.path.join(country_dir, loc)
        interim_dir = os.path.join(location_dir, "interim")
        processed_dir = os.path.join(location_dir, "processed")
        figs_dir = os.path.join(location_dir, "figs")
        
        if not os.path.exists(location_dir):
            logger.warning(f"Creating folders for {loc}")
            os.makedirs(interim_dir, exist_ok=True)
            os.makedirs(processed_dir, exist_ok=True)
            os.makedirs(figs_dir, exist_ok=True)

        df_out.to_csv(FOLDER["input"] + "aws.csv", index=False)
        plot_input(df_out, FOLDER['fig'], SITE["name"])
        
        logger.info(f"Completed processing {filename}")
        return True
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")
        return False

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("INFO")

    args = parse_args()
    
    # Check if we should list countries and exit
    if args.list_countries:
        list_available_countries()
        sys.exit(0)
    
    # Check if the country directory exists
    data_base_dir = os.path.join(dirname, 'data')
    country_dir = os.path.join(data_base_dir, args.country)
    
    if not os.path.exists(country_dir):
        print(f"Error: Country directory not found: {country_dir}")
        print("Run with --list_countries to see all available countries")
        sys.exit(1)
    
    try:
        # Get all filenames from data directory for this country
        filenames, _ = get_data_filenames(args.country)
        logger.info(f"Found {len(filenames)} CSV files to process for {args.country}")

        # Process all datasets in parallel
        process_datasets_parallel(filenames, args)
        
        print(f"\nSuccessfully processed data for country: {args.country}")
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
