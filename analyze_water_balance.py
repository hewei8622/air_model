"""
Analyze water balance details for each spray method.
This script reads the summary CSV and generates a detailed report on
water inputs, outputs, and efficiency metrics.
"""

import pandas as pd
import os

def analyze_water_balance(location="guttannen22"):
    """Generate a detailed water balance report for each spray method."""
    # Load summary CSV
    dirname = os.path.dirname(os.path.realpath(__file__))
    comparison_dir = f"{dirname}/results_comparison"
    summary_file = f"{comparison_dir}/{location}_summary.csv"
    
    if not os.path.exists(summary_file):
        print(f"No results found for {location}")
        return
    
    results = pd.read_csv(summary_file)
    
    # Print detailed water balance for each spray method
    for _, row in results.iterrows():
        spray = row['spray']
        
        # Get water components directly from the CSV, using any available columns
        # This ensures we capture all data including precipitation
        components = {}
        
        # Input components
        components['discharge'] = row.get('total_water_input', 0)
        components['precipitation'] = row.get('M_ppt', 0) if 'M_ppt' in row else 0
        components['initial_ice'] = row.get('initial_ice_mass', 0)
        components['deposition'] = row.get('M_dep', 0) if 'M_dep' in row else 0
        
        # Output components
        components['direct_waste'] = row.get('total_direct_waste', 0)
        components['final_ice'] = row.get('final_ice_mass', 0)
        components['meltwater'] = row.get('total_meltwater', 0)
        components['sublimation'] = row.get('total_sublimation', 0)
        
        # Calculate totals
        total_inputs = (components['discharge'] + 
                        components['precipitation'] + 
                        components['initial_ice'] + 
                        components['deposition'])
        
        total_outputs = (components['direct_waste'] + 
                        components['final_ice'] + 
                        components['meltwater'] + 
                        components['sublimation'])
        
        balance = total_inputs - total_outputs
        balance_pct = (balance / total_inputs) * 100 if total_inputs > 0 else 0
        
        print(f"\n===== Water Balance Details for {spray} =====")
        print("\nINPUTS:")
        print(f"  Fountain Discharge: {components['discharge']:.2f} kg ({components['discharge']/total_inputs*100:.2f}% of inputs)")
        
        if components['precipitation'] > 0:
            print(f"  Precipitation: {components['precipitation']:.2f} kg ({components['precipitation']/total_inputs*100:.2f}% of inputs)")
        else:
            print(f"  Precipitation: Not available or 0.00 kg (0.00% of inputs)")
            
        print(f"  Initial Ice: {components['initial_ice']:.2f} kg ({components['initial_ice']/total_inputs*100:.2f}% of inputs)")
        
        if components['deposition'] > 0:
            print(f"  Deposition: {components['deposition']:.2f} kg ({components['deposition']/total_inputs*100:.2f}% of inputs)")
        else:
            print(f"  Deposition: Not available or 0.00 kg (0.00% of inputs)")
            
        print(f"  TOTAL INPUTS: {total_inputs:.2f} kg")
        
        print("\nOUTPUTS:")
        print(f"  Direct Waste: {components['direct_waste']:.2f} kg ({components['direct_waste']/total_outputs*100:.2f}% of outputs)")
        print(f"  Final Ice Mass: {components['final_ice']:.2f} kg ({components['final_ice']/total_outputs*100:.2f}% of outputs)")
        print(f"  Meltwater: {components['meltwater']:.2f} kg ({components['meltwater']/total_outputs*100:.2f}% of outputs)")
        print(f"  Sublimation: {components['sublimation']:.2f} kg ({components['sublimation']/total_outputs*100:.2f}% of outputs)")
        print(f"  TOTAL OUTPUTS: {total_outputs:.2f} kg")
        
        print("\nBALANCE:")
        print(f"  Difference: {balance:.2f} kg ({balance_pct:.2f}%)")
        print(f"  Water Use Efficiency: {row.get('water_use_efficiency', 0):.2f}%")
        print(f"  Direct Waste Ratio: {row.get('direct_waste_ratio', 0):.2f}%")
        
        # Calculate percentage gain or loss
        ice_gain = components['final_ice'] - components['initial_ice']
        if components['initial_ice'] > 0:
            print(f"  Net Ice Gain: {ice_gain:.2f} kg ({ice_gain/components['initial_ice']*100:.2f}% of initial ice)")
        else:
            print(f"  Net Ice Gain: {ice_gain:.2f} kg (N/A% of initial ice)")
        
        # Display additional metrics if available
        for key in sorted(row.keys()):
            if key not in ['location', 'spray', 'total_water_input', 'initial_ice_mass', 'total_water_frozen', 
                          'total_direct_waste', 'total_meltwater', 'total_sublimation', 'final_ice_mass',
                          'water_balance_diff', 'water_balance_pct', 'water_use_efficiency', 'freezing_efficiency',
                          'stupa_retention_efficiency', 'direct_waste_ratio', 'overall_water_loss_ratio']:
                # This will show any additional metrics like M_ppt
                if not pd.isna(row[key]) and not isinstance(row[key], str):
                    print(f"  {key}: {row[key]:.2f}")
                elif not pd.isna(row[key]):
                    print(f"  {key}: {row[key]}")
        
        # Check for balance issues
        if abs(balance_pct) > 5:
            print("\nNOTE: Water balance discrepancy is greater than 5%. Possible reasons:")
            if balance < 0:
                print("  - Some water inputs may not be fully accounted for")
                print("  - Precipitation or deposition values may be underestimated")
                print("  - There may be water captured through other processes not modeled")
            else:
                print("  - Some water outputs may not be fully accounted for")
                print("  - Sublimation or meltwater may be underestimated")
                print("  - There may be other water loss processes not captured in the model")

def check_missing_data(location="guttannen22"):
    """Check for missing data columns in results, particularly precipitation."""
    # Load summary CSV
    dirname = os.path.dirname(os.path.realpath(__file__))
    comparison_dir = f"{dirname}/results_comparison"
    summary_file = f"{comparison_dir}/{location}_summary.csv"
    
    if not os.path.exists(summary_file):
        print(f"No results found for {location}")
        return
    
    results = pd.read_csv(summary_file)
    print(f"Columns in the summary data:")
    print(results.columns.tolist())
    
    # Check if precipitation data exists in result files
    for spray in results['spray']:
        try:
            # Try to read the simulation result file for this spray method
            result_file = f"{comparison_dir}/{location}_{spray}.csv"
            if os.path.exists(result_file):
                result_data = pd.read_csv(result_file)
                
                print(f"\n=== {spray} ===")
                print(f"Result file columns: {len(result_data.columns)}")
                
                # Check for precipitation data
                precip_cols = [col for col in result_data.columns if 'precip' in col.lower() or 'ppt' in col.lower() or 'snow' in col.lower()]
                if precip_cols:
                    print(f"Found precipitation-related columns: {precip_cols}")
                    for col in precip_cols:
                        if col in result_data.columns:
                            print(f"  {col}: Sum={result_data[col].sum():.2f}, Max={result_data[col].max():.2f}, Last={result_data[col].iloc[-1]:.2f}")
                else:
                    print("No precipitation-related columns found")
                
                # Check for other water components
                water_cols = ['M_ppt', 'M_dep', 'M_F', 'M_ice', 'M_input', 'M_sub', 'M_waste', 'M_water']
                found_cols = [col for col in water_cols if col in result_data.columns]
                if found_cols:
                    print("\nWater component columns:")
                    for col in found_cols:
                        print(f"  {col}: Last value={result_data[col].iloc[-1]:.2f}")
                
                # Check for other key data
                if 'ice' in result_data.columns:
                    print(f"\nIce mass: Initial={result_data['ice'].iloc[0]:.2f}, Final={result_data['ice'].iloc[-1]:.2f}")
                
                # Check precipitation in summary
                m_ppt = results.loc[results['spray'] == spray, 'M_ppt'].values[0] if 'M_ppt' in results.columns else "Not in summary"
                print(f"\nM_ppt in summary: {m_ppt}")
            else:
                print(f"\n{spray}: Result file not found")
        except Exception as e:
            print(f"\n{spray}: Error analyzing results - {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze water balance')
    parser.add_argument('--location', type=str, default="guttannen22",
                        help='Location to analyze')
    parser.add_argument('--check-missing', action='store_true',
                        help='Check for missing data columns')
    args = parser.parse_args()
    
    if args.check_missing:
        check_missing_data(args.location)
    else:
        analyze_water_balance(args.location) 