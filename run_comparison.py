"""Script to run simulations for comparison across different spray methods
"""

# External modules
import os, sys, shutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import argparse

# Locals
dirname = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dirname)
from src.models.icestupaClass import Icestupa
from src.utils.eff_criterion import nse
import logging, coloredlogs

def run_comparison(location):
    """Run comparison for a specific location"""
    print(f"\n\n======= RUNNING COMPARISON FOR {location} =======\n")
    
    # All spray methods
    sprays = ["scheduled_field", "unscheduled_field", "scheduled_wue", "scheduled_icv"]

    # Results summary
    results_summary = []
    
    for spray in sprays:
        print(f"\n\n=== Running simulation for {location} with {spray} ===\n")
        
        # Create Icestupa object and run simulation
        try:
            # Create Icestupa object
            icestupa = Icestupa(location=location, spray=spray)
            
            # Generate input
            icestupa.gen_input()
            
            # Simulate air
            icestupa.sim_air()
            
            # Generate output
            icestupa.gen_output()
            
            # Summarize figures
            icestupa.summary_figures()
            
            # Read the output DataFrame
            df = icestupa.df
            
            # Debug: print column names to see what data is available
            print("\nAvailable columns in the DataFrame:")
            print(df.columns.tolist())
            
            # Calculate metrics
            metrics = {
                "location": location,
                "spray": spray,
            }
            
            # Store the maximum ice volume
            metrics["ice_volume_max"] = df["iceV"].max()
            
            # ---------------------------------------------
            # Water Conservation Metrics
            # ---------------------------------------------
            
            # 1. Total water input (from discharge)
            # NOTE: Discharge is an instantaneous value (L/min) at each timestep
            # We sum (Discharge * timestep duration) across all timesteps to get total volume
            if 'Discharge' in df.columns:
                # Convert from L/min to kg, assuming density of water = 1000 kg/m³
                # Multiply by DT (minutes) and divide by 60 to get volume per timestep
                metrics["total_water_input"] = (df["Discharge"] * icestupa.DT / 60).sum()
            else:
                metrics["total_water_input"] = 0
                
            # Get the initial ice mass (already accounted for in the model but not in our input calculation)
            # This is important for water balance calculations
            initial_ice_mass = df["ice"].iloc[1] if len(df) > 1 else 0
            metrics["initial_ice_mass"] = initial_ice_mass
                
            # 2. Total water that froze to build the stupa
            # NOTE: fountain_froze is cumulative in the dataset; each timestep adds new freeze amount
            # Take the last value which represents the total across the simulation
            if 'fountain_froze' in df.columns:
                metrics["total_water_frozen"] = df["fountain_froze"].iloc[-1]
            else:
                metrics["total_water_frozen"] = 0
                
            # Include precipitation (M_ppt) in metrics
            if 'M_ppt' in df.columns:
                metrics["M_ppt"] = df["M_ppt"].iloc[-1]
            else:
                # Check if there are precipitation-related columns
                if 'ppt' in df.columns:
                    # Calculate total precipitation (convert from mm to kg using area)
                    # Multiply by the cone area (m²) and by 1000 (mm to m) to get volume in m³, then by density of water (1000 kg/m³)
                    if 'A_cone' in df.columns:
                        # Use average cone area over the simulation period for more accuracy
                        avg_area = df['A_cone'].mean()
                        # Sum precipitation over time (mm) and convert to kg
                        metrics["M_ppt"] = (df['ppt'] * avg_area / 1000 * 1000).sum()
                        print(f"DEBUG: Precipitation calculation for {spray}:")
                        print(f"  - Average cone area: {avg_area:.2f} m²")
                        print(f"  - Total precipitation (sum): {df['ppt'].sum():.2f} mm")
                        print(f"  - Total precipitation mass: {metrics['M_ppt']:.2f} kg")
                    else:
                        # Assume a fixed area if A_cone is not available
                        metrics["M_ppt"] = (df['ppt'] * 50.0 / 1000 * 1000).sum()  # Assuming 50 m² area
                        print(f"DEBUG: Using fixed area (50 m²) for precipitation calculation for {spray}")
                        print(f"  - Total precipitation (sum): {df['ppt'].sum():.2f} mm")
                        print(f"  - Total precipitation mass: {metrics['M_ppt']:.2f} kg")
                else:
                    precip_cols = [col for col in df.columns if 'precip' in col.lower() or 'ppt' in col.lower() or 'snow' in col.lower()]
                    if precip_cols:
                        metrics["M_ppt"] = df[precip_cols[0]].sum()
                    else:
                        metrics["M_ppt"] = 0
                
            # Include deposition (M_dep) in metrics
            if 'M_dep' in df.columns:
                metrics["M_dep"] = df["M_dep"].iloc[-1]
            else:
                # Check if there are deposition-related columns
                if 'dep' in df.columns:
                    # Get the sum of all deposition values (assume they are already in kg)
                    metrics["M_dep"] = df['dep'].sum()
                else:
                    dep_cols = [col for col in df.columns if 'dep' in col.lower()]
                    if dep_cols:
                        metrics["M_dep"] = df[dep_cols[0]].sum()
                    else:
                        metrics["M_dep"] = 0
                
            # 3. Total water wasted directly (not frozen)
            # NOTE: wastewater is cumulative in the dataset; each timestep adds new waste amount
            # Take the last value which represents the total across the simulation
            if 'wastewater' in df.columns:
                metrics["total_direct_waste"] = df["wastewater"].iloc[-1]
            else:
                metrics["total_direct_waste"] = 0
                
            # 4. Total meltwater (ice that melted)
            # NOTE: meltwater is cumulative in the dataset; each timestep adds new melt amount
            # Take the last value which represents the total across the simulation
            if 'meltwater' in df.columns:
                metrics["total_meltwater"] = df["meltwater"].iloc[-1]
            else:
                metrics["total_meltwater"] = 0
                
            # 5. Total sublimation (ice that sublimated directly to vapor)
            # NOTE: vapour is cumulative in the dataset; each timestep adds new sublimation amount
            # Take the last value which represents the total across the simulation
            if 'vapour' in df.columns:
                metrics["total_sublimation"] = df["vapour"].iloc[-1]
            else:
                metrics["total_sublimation"] = 0
                
            # 6. Final ice mass
            # NOTE: ice is a cumulative mass balance; updated at each timestep with freezing, melting, etc.
            # Take the last value which represents the final ice mass
            if 'ice' in df.columns:
                metrics["final_ice_mass"] = df["ice"].iloc[-1]
            else:
                metrics["final_ice_mass"] = 0
            
            # Verify that the water components make physical sense
            # For example, total frozen should not exceed total input
            if metrics["total_water_frozen"] > metrics["total_water_input"] * 1.05:  # Allow 5% tolerance
                print(f"WARNING: total_water_frozen ({metrics['total_water_frozen']:.2f} kg) exceeds total_water_input ({metrics['total_water_input']:.2f} kg)")
                # Adjust to maintain physical consistency if needed
                # metrics["total_water_frozen"] = metrics["total_water_input"]
                
            # Water balance check
            # Include initial ice mass, precipitation, and deposition in the total water balance calculation
            total_inputs = (metrics["total_water_input"] + 
                           metrics["initial_ice_mass"] + 
                           metrics["M_ppt"] + 
                           metrics["M_dep"])
            total_outputs = (metrics["final_ice_mass"] + 
                            metrics["total_direct_waste"] + 
                            metrics["total_meltwater"] + 
                            metrics["total_sublimation"])
            
            water_balance = abs(total_inputs - total_outputs)
            water_balance_pct = water_balance / total_inputs * 100 if total_inputs > 0 else float('nan')
            
            # Store water balance information
            metrics["water_balance_diff"] = total_inputs - total_outputs
            metrics["water_balance_pct"] = water_balance_pct
            
            # Check if water conservation is maintained within reasonable tolerance
            if water_balance_pct > 5:  # More than 5% imbalance
                print(f"WARNING: Water balance check failed. Imbalance: {water_balance_pct:.2f}%")
                print(f"  Total inputs: {total_inputs:.2f} kg")
                print(f"    - Water input: {metrics['total_water_input']:.2f} kg")
                print(f"    - Initial ice: {metrics['initial_ice_mass']:.2f} kg")
                print(f"    - Precipitation: {metrics['M_ppt']:.2f} kg")
                print(f"    - Deposition: {metrics['M_dep']:.2f} kg")
                print(f"  Total outputs: {total_outputs:.2f} kg")
                print(f"    - Final ice: {metrics['final_ice_mass']:.2f} kg")
                print(f"    - Direct waste: {metrics['total_direct_waste']:.2f} kg")
                print(f"    - Meltwater: {metrics['total_meltwater']:.2f} kg")
                print(f"    - Sublimation: {metrics['total_sublimation']:.2f} kg")
                print("  Note: Water balance discrepancies may be due to physical processes not fully accounted for in the model, ")
                print("        such as changes in snow density, refreezing of meltwater, or other model simplifications.")
            
            # Calculate the efficiency metrics based on corrected values
            
            # Water use efficiency: ratio of final ice mass to total water input
            if metrics["total_water_input"] > 0:
                metrics["water_use_efficiency"] = (metrics["final_ice_mass"] / metrics["total_water_input"]) * 100
            else:
                metrics["water_use_efficiency"] = float('nan')
            
            # Freezing efficiency: ratio of total water frozen to total water input
            if metrics["total_water_input"] > 0:
                metrics["freezing_efficiency"] = (metrics["total_water_frozen"] / metrics["total_water_input"]) * 100
            else:
                metrics["freezing_efficiency"] = float('nan')
            
            # Stupa retention efficiency: ratio of final ice mass to total water frozen
            if metrics["total_water_frozen"] > 0:
                metrics["stupa_retention_efficiency"] = (metrics["final_ice_mass"] / metrics["total_water_frozen"]) * 100
            else:
                metrics["stupa_retention_efficiency"] = float('nan')
            
            # Direct waste ratio: percentage of input water directly wasted
            if metrics["total_water_input"] > 0:
                metrics["direct_waste_ratio"] = (metrics["total_direct_waste"] / metrics["total_water_input"]) * 100
            else:
                metrics["direct_waste_ratio"] = float('nan')
            
            # Overall water loss ratio: percentage of input water lost (direct waste + meltwater)
            if metrics["total_water_input"] > 0:
                metrics["overall_water_loss_ratio"] = ((metrics["total_direct_waste"] + metrics["total_meltwater"]) / 
                                                       metrics["total_water_input"]) * 100
            else:
                metrics["overall_water_loss_ratio"] = float('nan')
            
            # Save outputs to comparison directory
            location_dir = location.lower().replace(" ", "")
            comparison_dir = f"{dirname}/results_comparison/{location_dir}/{spray}"
            os.makedirs(comparison_dir, exist_ok=True)
            
            # Copy output CSV
            output_csv = "output.csv"
            if os.path.exists(output_csv):
                dest_file = f"{comparison_dir}/{output_csv}"
                shutil.copy(output_csv, dest_file)
                print(f"Copied results to {dest_file}")
            
            # Copy key figures
            for fig in ["output.png", "Vol_Validation.png", "Discharge.png"]:
                if os.path.exists(fig):
                    dest_file = f"{comparison_dir}/{fig}"
                    shutil.copy(fig, dest_file)
                    print(f"Copied {fig} to {dest_file}")
            
            # Add metrics to summary
            results_summary.append(metrics)
            
            # Print summary of water components for verification
            print(f"\nWater Components Summary for {location} with {spray}:")
            print(f"  - Total Water Input: {metrics['total_water_input']:.2f} kg")
            print(f"  - Initial Ice Mass: {metrics['initial_ice_mass']:.2f} kg")
            print(f"  - Precipitation: {metrics['M_ppt']:.2f} kg")
            print(f"  - Deposition: {metrics['M_dep']:.2f} kg")
            print(f"  - Total Water Frozen: {metrics['total_water_frozen']:.2f} kg")
            print(f"  - Total Direct Waste: {metrics['total_direct_waste']:.2f} kg")
            print(f"  - Total Meltwater: {metrics['total_meltwater']:.2f} kg")
            print(f"  - Total Sublimation: {metrics['total_sublimation']:.2f} kg")
            print(f"  - Final Ice Mass: {metrics['final_ice_mass']:.2f} kg")
            print(f"  - Water Balance Difference: {metrics['water_balance_diff']:.2f} kg ({water_balance_pct:.2f}%)")
            
            # Print the key efficiency metrics
            print(f"\nEfficiency Metrics:")
            print(f"  - Water Use Efficiency: {metrics['water_use_efficiency']:.2f}%")
            print(f"  - Freezing Efficiency: {metrics['freezing_efficiency']:.2f}%")
            print(f"  - Stupa Retention Efficiency: {metrics['stupa_retention_efficiency']:.2f}%")
            print(f"  - Direct Waste Ratio: {metrics['direct_waste_ratio']:.2f}%")
            
        except Exception as e:
            print(f"ERROR running simulation for {location} with {spray}: {str(e)}")
                    
    # Save summary as CSV
    summary_file = f"{dirname}/results_comparison/{location.lower().replace(' ', '')}_summary.csv"
    if results_summary:
        pd.DataFrame(results_summary).to_csv(summary_file, index=False)
        print(f"\nAll simulations complete. Results saved to {summary_file}")
        print("\nSummary of results:")
        print(pd.DataFrame(results_summary))
    else:
        print(f"\nNo successful simulations for {location}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run comparison for Guttannen 2022')
    parser.add_argument('--location', type=str, default="guttannen22", 
                      help='Location code (default: guttannen22)')
    args = parser.parse_args()
    
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("WARNING")
    
    # Create the comparison directory if it doesn't exist
    os.makedirs(f"{dirname}/results_comparison", exist_ok=True)
    
    # Run comparison
    run_comparison(args.location) 