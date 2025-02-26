"""Command line interface to create or display Icestupa class
"""

# External modules
import os, sys, shutil, time, argparse, json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import multiprocessing as mp
from functools import partial

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.models.icestupaClass import Icestupa
from src.utils.settings import config
from src.utils import setup_logger
from src.utils.eff_criterion import nse
import logging, coloredlogs


def process_locations_parallel(filenames, args, num_processes=None):
    """
    Process multiple locations in parallel
    """
    if num_processes is None:
        num_processes = mp.cpu_count()

    logger.info(f"Starting parallel processing with {num_processes} processes")
    
    # Create a partial function with fixed args
    process_func = partial(process_single_location, args=args)
    
    # Create a process pool and map the work
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_func, filenames)
    
    # Count successful processes
    successful = sum(1 for r in results if r)
    logger.info(f"Processed {successful} out of {len(filenames)} locations successfully")

def get_data_filenames(datadir=None):
    """Find all CSV files in the data directory"""
    if datadir is None:
        datadir = os.path.join(dirname, 'data')
    
    # Look for CSV files in the data directory
    csv_files = [f for f in os.listdir(datadir+"/era5/") if f.endswith('.csv')]
    
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the data directory")
    
    return csv_files


def parse_args():
    parser = argparse.ArgumentParser(description="Command line interface to create or display Icestupa class")

    # parser.add_argument("--location", required=True, help="Specify the location filename as lat_long_alt (e.g., 34.216_77.606_4009.csv)")
    parser.add_argument("--start_year", required=False, help="Specify the start year (e.g., 2019)")
    parser.add_argument("--end_year", required=False, help="Specify the end year (e.g., 2020)")
    parser.add_argument("--datadir", required=True, help="Specify the data folder")

    return parser.parse_args()

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

def process_single_location(filename, args):
    """Process a single location's data"""
    try:
        logger.info(f"\nProcessing {filename}")
        coords, alt, location = extract_location_details(filename)

        SITE, FOLDER = config(location, start_year=args.start_year, end_year=args.end_year, 
                            alt=alt, coords=coords, datadir=args.datadir)
        
        icestupa = Icestupa(SITE, FOLDER)
        icestupa.sim_air(test=False)
        # icestupa.summary_figures()
        
        logger.info(f"Completed processing {filename}")
        return True
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")
        return False

def consolidate_results(filenames, output_dir):
    """
    Consolidate results from multiple locations into a single JSON file.
    
    Args:
        filenames (list): List of CSV filenames processed
        output_dir (str): Directory to save the consolidated results
    """
    consolidated_results = []
    
    for filename in filenames:
        # Extract location details from filename
        if filename.endswith('.csv'):
            location = filename[:-4]
        else:
            location = filename
            
        try:
            lat, lon, alt = location.split('_')
            lat = float(lat)
            lon = float(lon)
            alt = float(alt)
            
            # Read the individual results.json for this location
            location_dir = os.path.join(output_dir, location, "processed")
            with open(os.path.join(location_dir, "results.json"), "r") as f:
                location_results = json.load(f)
            
            # Create consolidated entry
            result_entry = {
                "latitude": lat,
                "longitude": lon,
                "altitude": alt,
                "iceV_max": location_results["iceV_max"],
                "survival_days": location_results["survival_days"]
            }
            
            consolidated_results.append(result_entry)
            
        except (ValueError, FileNotFoundError, KeyError) as e:
            logger.error(f"Error processing results for {filename}: {str(e)}")
            continue
    
    # Save consolidated results
    with open(os.path.join(output_dir, "consolidated_results.json"), "w") as f:
        json.dump(consolidated_results, f, indent=4, sort_keys=True)
    
    logger.info(f"Consolidated results saved to {output_dir}/consolidated_results.json")
    
    return consolidated_results



def process_single_result(filename, datadir):
    """
    Process a single location's results.
    
    Args:
        filename (str): CSV filename to process
        datadir (str): Base output directory
    
    Returns:
        dict: Location results with coordinates and metrics
    """
    try:
        # Extract location details from filename
        if filename.endswith('.csv'):
            location = filename[:-4]
        else:
            location = filename
            
        lat, lon, alt = location.split('_')
        lat = float(lat)
        lon = float(lon)
        alt = float(alt)
        
        # Read the individual results.json for this location
        location_dir = os.path.join(datadir, location, "processed")
        with open(os.path.join(location_dir, "results.json"), "r") as f:
            location_results = json.load(f)
        
        # Create consolidated entry
        return {
            "latitude": lat,
            "longitude": lon,
            "altitude": alt,
            "iceV_max": location_results["iceV_max"],
            "survival_days": location_results["survival_days"]
        }
        
    except (ValueError, FileNotFoundError, KeyError) as e:
        logger.error(f"Error processing results for {filename}: {str(e)}")
        return None

def consolidate_results_parallel(filenames, datadir, num_processes=None):
    """
    Consolidate results from multiple locations in parallel.
    
    Args:
        filenames (list): List of CSV filenames processed
        datadir (str): Directory to save the consolidated results
        num_processes (int, optional): Number of processes to use. Defaults to CPU count.
    """
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    logger.info(f"Starting results consolidation...")
    
    # Create a partial function with fixed datadir
    process_func = partial(process_single_result, datadir=datadir)
    
    # Create a process pool and map the work
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_func, filenames)
    
    # Filter out None results (from errors) and consolidate
    consolidated_results = [r for r in results if r is not None]
    
    # Sort results by latitude for consistency
    consolidated_results.sort(key=lambda x: x["latitude"])
    
    # Add some statistics to the consolidated results
    summary_stats = {
        "total_locations": len(consolidated_results),
        "average_survival_days": sum(r['survival_days'] for r in consolidated_results) / len(consolidated_results),
        "max_iceV": max(r['iceV_max'] for r in consolidated_results),
        "min_iceV": min(r['iceV_max'] for r in consolidated_results),
        "average_iceV": sum(r['iceV_max'] for r in consolidated_results) / len(consolidated_results),
        "average_altitude": sum(r['altitude'] for r in consolidated_results) / len(consolidated_results)
    }
    
    # Save consolidated results with summary stats
    output_data = {
        "summary_statistics": summary_stats,
        "location_results": consolidated_results
    }
    
    output_file = os.path.join(datadir, "/consolidated_results.json")
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4, sort_keys=True)
    
    logger.info(f"Consolidated results saved to {output_file}")
    logger.info(f"Processed {len(consolidated_results)} locations successfully")
    
    return consolidated_results

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("WARNING")
    st = time.time()

    args = parse_args()
    
    # Get all filenames from data directory
    filenames = get_data_filenames(args.datadir)
    logger.info(f"Found {len(filenames)} CSV files to process")

    # Process all locations in parallel
    process_locations_parallel(filenames, args)
    
    # Consolidate results after parallel processing
    logger.info("Consolidating results from all locations...")
    consolidated_results = consolidate_results_parallel(filenames, args.datadir)
    
    # Print summary of consolidated results
    print("\nSummary of results across all locations:")
    print(f"Total locations processed: {len(consolidated_results)}")
    avg_survival = sum(r['survival_days'] for r in consolidated_results) / len(consolidated_results)
    max_icev = max(r['iceV_max'] for r in consolidated_results)
    print(f"Average survival days: {avg_survival:.2f}")
    print(f"Maximum iceV across all locations: {max_icev:.2f}")

    # get the end time and print total execution time
    et = time.time()
    elapsed_time = (et - st)/60
    print(f'\n\tTotal execution time: {round(elapsed_time,2)} min\n')
