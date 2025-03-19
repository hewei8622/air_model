"""Script to run simulations for all locations and all spray methods
"""

# External modules
import os, sys, shutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import argparse
import subprocess
import json
import re

# Locals
dirname = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dirname)
from src.models.icestupaClass import Icestupa
from src.utils.eff_criterion import nse
import logging, coloredlogs

# Define the locations and spray methods
LOCATIONS = [
    "guttannen20",
    "guttannen21",
    "guttannen22",
    "gangles21"
]

SPRAY_METHODS = [
    "scheduled_field",
    "unscheduled_field",
    "scheduled_wue",
    "scheduled_icv"
]

# Mapping from code name to display name
LOCATION_DISPLAY_NAMES = {
    "guttannen20": "Guttannen 2020",
    "guttannen21": "Guttannen 2021",
    "guttannen22": "Guttannen 2022",
    "gangles21": "Gangles 2021"
}

def create_output_dirs():
    """Create output directories if they don't exist."""
    base_dir = "results_comparison"
    
    # Create the base directory
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create location directories
    for location in LOCATIONS:
        location_dir = os.path.join(base_dir, location)
        if not os.path.exists(location_dir):
            os.makedirs(location_dir)
        
        # Create spray method directories
        for method in SPRAY_METHODS:
            method_dir = os.path.join(location_dir, method)
            if not os.path.exists(method_dir):
                os.makedirs(method_dir)
    
    return base_dir

def run_simulation(location, spray_method, base_dir):
    """Run a simulation for a specific location and spray method."""
    print(f"\n=== Running simulation for {LOCATION_DISPLAY_NAMES[location]} with {spray_method} ===\n")
    
    try:
        # Create Icestupa object
        stupa = Icestupa(config_json={"location": location, "scheduler": spray_method})
        
        # Generate input
        stupa.gen_input()
        
        # Simulate air
        stupa.sim_air()
        
        # Generate output
        stupa.gen_output()
        
        # Summarize figures
        stupa.summarize_figures()
        
        # Calculate metrics
        metrics = {}
        
        # Store the maximum ice volume
        metrics["ice_volume_max"] = stupa.df["iceV"].max()
        
        # Get the water input
        metrics["total_water_input"] = stupa.df["Discharge"].sum() * stupa.DT / 60  # L/min to kg
        
        # Get the total water that froze
        metrics["total_water_frozen"] = stupa.df["fountain_froze"].max()
        
        # Get direct waste (water that was sprayed but didn't freeze)
        metrics["total_direct_waste"] = stupa.df["wastewater"].max()
        
        # Get total meltwater
        metrics["total_meltwater"] = stupa.df["meltwater"].max()
        
        # Get total sublimation
        if "vapour" in stupa.df.columns:
            metrics["total_sublimation"] = stupa.df["vapour"].iloc[-1]
        else:
            metrics["total_sublimation"] = 0
        
        # Final ice mass
        metrics["final_ice_mass"] = stupa.df["ice"].iloc[-1]
        
        # Calculate efficiency metrics
        
        # Water use efficiency: ratio of final ice mass to total water input
        metrics["water_use_efficiency"] = (metrics["final_ice_mass"] / metrics["total_water_input"]) * 100
        
        # Freezing efficiency: ratio of total water frozen to total water input
        metrics["freezing_efficiency"] = (metrics["total_water_frozen"] / metrics["total_water_input"]) * 100
        
        # Stupa retention efficiency: ratio of final ice mass to total water frozen
        if metrics["total_water_frozen"] > 0:
            metrics["stupa_retention_efficiency"] = (metrics["final_ice_mass"] / metrics["total_water_frozen"]) * 100
        else:
            metrics["stupa_retention_efficiency"] = 0
        
        # Direct waste ratio: ratio of direct waste to total water input
        metrics["direct_waste_ratio"] = (metrics["total_direct_waste"] / metrics["total_water_input"]) * 100
        
        # Overall water loss ratio: ratio of (direct waste + meltwater) to total water input
        metrics["overall_water_loss_ratio"] = ((metrics["total_direct_waste"] + metrics["total_meltwater"]) / metrics["total_water_input"]) * 100
        
        # Save outputs to comparison directory
        output_dir = os.path.join(base_dir, location, spray_method)
        
        # Copy CSV
        shutil.copy('output.csv', os.path.join(output_dir, 'output.csv'))
        print(f"Copied results to {os.path.join(output_dir, 'output.csv')}")
        
        # Copy PNGs
        for png_file in ['output.png', 'Vol_Validation.png', 'Discharge.png']:
            if os.path.exists(png_file):
                shutil.copy(png_file, os.path.join(output_dir, png_file))
                print(f"Copied {png_file} to {os.path.join(output_dir, png_file)}")
        
        return {
            "location": LOCATION_DISPLAY_NAMES[location],
            "spray": spray_method,
            **metrics
        }
    
    except Exception as e:
        print(f"Error running simulation for {location} with {spray_method}: {e}")
        return None

def main():
    # Create output directories
    base_dir = create_output_dirs()
    
    # Store results
    all_results = []
    
    # Run simulations for each location and spray method
    for location in LOCATIONS:
        location_results = []
        
        for spray_method in SPRAY_METHODS:
            # Run simulation
            result = run_simulation(location, spray_method, base_dir)
            
            if result:
                location_results.append(result)
                all_results.append(result)
        
        # Save location results to CSV
        if location_results:
            df = pd.DataFrame(location_results)
            csv_path = os.path.join(base_dir, f"{location}_summary.csv")
            df.to_csv(csv_path, index=False)
            print(f"\nSaved results for {LOCATION_DISPLAY_NAMES[location]} to {csv_path}")
    
    # Save all results to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(base_dir, "all_results_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved all results to {csv_path}")
        
        print("\nSummary of results:")
        print(df[['location', 'spray', 'ice_volume_max', 'water_use_efficiency', 'freezing_efficiency']])

if __name__ == "__main__":
    main() 