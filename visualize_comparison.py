"""Script to visualize comparison results across different spray methods
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Directory setup
dirname = os.path.dirname(os.path.realpath(__file__))
comparison_dir = f"{dirname}/results_comparison"

def load_results(csv_path):
    """Load the results summary CSV."""
    df = pd.read_csv(csv_path)
    return df

def plot_efficiency_comparison(df, location, output_dir):
    """Plot efficiency metrics for comparison."""
    # Filter data for the specified location
    location_df = df[df['location'] == location].copy()
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Efficiency Metrics Comparison for {location}', fontsize=16)
    
    # Water Use Efficiency
    ax1 = sns.barplot(x='spray', y='water_use_efficiency', data=location_df, ax=axes[0, 0])
    axes[0, 0].set_title('Water Use Efficiency')
    axes[0, 0].set_ylabel('WUE (%)')
    axes[0, 0].grid(True, alpha=0.3)
    # Add value labels
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.1f', fontsize=9)
    
    # Freezing Efficiency
    ax2 = sns.barplot(x='spray', y='freezing_efficiency', data=location_df, ax=axes[0, 1])
    axes[0, 1].set_title('Freezing Efficiency')
    axes[0, 1].set_ylabel('Freezing Efficiency (%)')
    axes[0, 1].grid(True, alpha=0.3)
    # Add value labels
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.1f', fontsize=9)
    
    # Stupa Retention Efficiency
    ax3 = sns.barplot(x='spray', y='stupa_retention_efficiency', data=location_df, ax=axes[1, 0])
    axes[1, 0].set_title('Stupa Retention Efficiency')
    axes[1, 0].set_ylabel('Retention Efficiency (%)')
    axes[1, 0].grid(True, alpha=0.3)
    # Add value labels
    for container in ax3.containers:
        ax3.bar_label(container, fmt='%.1f', fontsize=9)
    
    # Direct Waste Ratio
    ax4 = sns.barplot(x='spray', y='direct_waste_ratio', data=location_df, ax=axes[1, 1])
    axes[1, 1].set_title('Direct Waste Ratio')
    axes[1, 1].set_ylabel('Direct Waste Ratio (%)')
    axes[1, 1].grid(True, alpha=0.3)
    # Add value labels
    for container in ax4.containers:
        ax4.bar_label(container, fmt='%.1f', fontsize=9)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f'{location.replace(" ", "_").lower()}_efficiency_comparison.png'), dpi=300)
    plt.close()

def plot_water_flow(df, location, output_dir):
    """Plot water flow metrics for comparison with improved scaling."""
    # Filter data for the specified location
    location_df = df[df['location'] == location].copy()
    
    # Create a figure for water flow - main components
    plt.figure(figsize=(14, 10))
    
    # Create a bar chart showing the water components
    water_components = [
        'total_water_input', 
        'initial_ice_mass',
        'total_water_frozen', 
        'total_direct_waste', 
        'final_ice_mass'
    ]
    
    # Melt the dataframe to get it into the right format
    melted_df = pd.melt(
        location_df, 
        id_vars=['spray'], 
        value_vars=water_components,
        var_name='Water Component', 
        value_name='Mass (kg)'
    )
    
    # Map the column names to more readable labels
    component_labels = {
        'total_water_input': 'Total Water Input',
        'initial_ice_mass': 'Initial Ice Mass',
        'total_water_frozen': 'Total Water Frozen',
        'total_direct_waste': 'Direct Waste',
        'final_ice_mass': 'Final Ice Mass'
    }
    
    melted_df['Water Component'] = melted_df['Water Component'].map(component_labels)
    
    # Plot with a linear scale
    ax = sns.barplot(x='spray', y='Mass (kg)', hue='Water Component', data=melted_df)
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', fontsize=9)
    
    plt.title(f'Water Flow Components for {location}', fontsize=16)
    plt.ylabel('Mass (kg)', fontsize=14)
    plt.xlabel('Spray Method', fontsize=14)
    plt.legend(title='Component', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{location.replace(" ", "_").lower()}_water_flow.png'), dpi=300)
    plt.close()
    
    # Create a second plot for water efficiency metrics in percentage terms
    plt.figure(figsize=(16, 8))
    
    # Create 4 subplots for derived metrics
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f'Water Mass Distribution for {location}', fontsize=16)
    
    # 1. Plot the percentage of total water input that became ice vs was wasted
    # Create a stacked percentage bar chart
    pct_data = location_df.copy()
    for spray_method in pct_data['spray'].unique():
        method_data = pct_data[pct_data['spray'] == spray_method]
        total_input = method_data['total_water_input'].iloc[0] + method_data['initial_ice_mass'].iloc[0]
        if total_input > 0:
            pct_data.loc[pct_data['spray'] == spray_method, 'frozen_pct'] = (method_data['total_water_frozen'] / total_input * 100).iloc[0]
            pct_data.loc[pct_data['spray'] == spray_method, 'direct_waste_pct'] = (method_data['total_direct_waste'] / total_input * 100).iloc[0]
            pct_data.loc[pct_data['spray'] == spray_method, 'melt_pct'] = ((method_data['total_water_frozen'] - method_data['final_ice_mass'] + method_data['initial_ice_mass']) / total_input * 100).iloc[0]
            pct_data.loc[pct_data['spray'] == spray_method, 'sublimation_pct'] = (method_data['total_sublimation'] / total_input * 100).iloc[0]
            pct_data.loc[pct_data['spray'] == spray_method, 'final_ice_pct'] = (method_data['final_ice_mass'] / total_input * 100).iloc[0]
    
    # Prepare data for stacked bar chart - input distribution
    input_components = [
        'final_ice_pct',
        'melt_pct', 
        'sublimation_pct',
        'direct_waste_pct'
    ]
    
    input_labels = {
        'final_ice_pct': 'Final Ice Remaining',
        'melt_pct': 'Melted Ice', 
        'sublimation_pct': 'Sublimated Ice',
        'direct_waste_pct': 'Direct Waste'
    }
    
    # Melt the dataframe for stacked bar
    input_df = pd.melt(
        pct_data,
        id_vars=['spray'],
        value_vars=input_components,
        var_name='Distribution',
        value_name='Percentage (%)'
    )
    
    input_df['Distribution'] = input_df['Distribution'].map(input_labels)
    
    # Plot stacked bar chart - where the water goes
    input_df_pivot = input_df.pivot(index='spray', columns='Distribution', values='Percentage (%)')
    
    # Ensure all required columns exist
    for col in input_labels.values():
        if col not in input_df_pivot.columns:
            input_df_pivot[col] = 0
            
    # Sort columns to ensure consistent stacking order
    sorted_cols = ['Final Ice Remaining', 'Melted Ice', 'Sublimated Ice', 'Direct Waste']
    input_df_pivot = input_df_pivot[sorted_cols]
    
    # Plot the stacked bar
    input_df_pivot.plot(kind='bar', stacked=True, ax=axes[0], colormap='viridis')
    axes[0].set_title('Distribution of Input Water (%)')
    axes[0].set_ylabel('Percentage of Total Input (%)')
    axes[0].set_xlabel('Spray Method')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(title='Water Destination', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # For the second subplot, show the efficiency metrics: freezing vs retention
    efficiency_components = [
        'freezing_efficiency',
        'stupa_retention_efficiency'
    ]
    
    efficiency_labels = {
        'freezing_efficiency': 'Freezing Efficiency',
        'stupa_retention_efficiency': 'Retention Efficiency'
    }
    
    # Melt the dataframe for the efficiency comparison
    efficiency_df = pd.melt(
        pct_data,
        id_vars=['spray'],
        value_vars=efficiency_components,
        var_name='Efficiency',
        value_name='Percentage (%)'
    )
    
    efficiency_df['Efficiency'] = efficiency_df['Efficiency'].map(efficiency_labels)
    
    # Plot the efficiency comparison
    sns.barplot(x='spray', y='Percentage (%)', hue='Efficiency', data=efficiency_df, ax=axes[1])
    axes[1].set_title('Efficiency Metrics Comparison')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].set_xlabel('Spray Method')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(title='Efficiency Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add value labels
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt='%.1f', fontsize=9)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{location.replace(" ", "_").lower()}_water_distribution.png'), dpi=300)
    plt.close()

def plot_ice_volume_over_time(location, output_dir):
    """Plot ice volume over time from each method's output CSV."""
    plt.figure(figsize=(14, 8))
    
    spray_methods = ['scheduled_field', 'unscheduled_field', 'scheduled_wue', 'scheduled_icv']
    location_dir = location.replace(" ", "").lower()
    
    for method in spray_methods:
        # Load the CSV for this method
        csv_path = os.path.join(output_dir, location_dir, method, 'output.csv')
        try:
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                # Convert time column to datetime if needed
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    plt.plot(df['timestamp'], df['iceV'], label=method)
                elif 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    plt.plot(df['time'], df['iceV'], label=method)
        except Exception as e:
            print(f"Error loading data for {method}: {e}")
    
    plt.title(f'Ice Volume Over Time for {location}', fontsize=16)
    plt.ylabel('Ice Volume (m³)', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{location.replace(" ", "_").lower()}_ice_volume_time.png'), dpi=300)
    plt.close()

def generate_metrics_table(df, location, output_dir):
    """Generate a formatted table of key metrics."""
    # Filter data for the specified location
    location_df = df[df['location'] == location].copy()
    
    # Select relevant columns
    metrics_df = location_df[['spray', 'ice_volume_max', 'water_use_efficiency', 
                             'freezing_efficiency', 'stupa_retention_efficiency', 
                             'direct_waste_ratio']]
    
    # Format the table
    metrics_table = metrics_df.set_index('spray').round(2)
    
    # Save to CSV
    metrics_table.to_csv(os.path.join(output_dir, f'{location.replace(" ", "_").lower()}_metrics.csv'))
    
    return metrics_table

def plot_water_balance(df, location, output_dir):
    """Plot a water balance sankey diagram."""
    # This is a more complex visualization that requires extra libraries
    # If you want to implement this later, you would need to use sankey diagrams
    # For now, just compute the balance and report it
    
    location_df = df[df['location'] == location].copy()
    
    # Create a dataframe to show water balance results
    balance_data = []
    
    # For each spray method, calculate water balance
    for spray in location_df['spray'].unique():
        spray_data = location_df[location_df['spray'] == spray]
        row = spray_data.iloc[0]
        
        # Input components
        fountain_discharge = row.get('total_water_input', 0)
        initial_ice = row.get('initial_ice_mass', 0)
        precipitation = row.get('M_ppt', 0)
        deposition = row.get('M_dep', 0)
        
        # Output components
        final_ice = row.get('final_ice_mass', 0)
        direct_waste = row.get('total_direct_waste', 0)
        meltwater = row.get('total_meltwater', 0)
        sublimation = row.get('total_sublimation', 0)
        
        # Calculate totals
        total_in = fountain_discharge + initial_ice + precipitation + deposition
        total_out = final_ice + direct_waste + meltwater + sublimation
        
        # Calculate balance
        balance = total_in - total_out
        pct_diff = (balance / total_in) * 100 if total_in > 0 else 0
        
        print(f"\nWater Balance for {spray}:")
        print(f"  Total In: {total_in:.2f} kg")
        print(f"    - Water input: {fountain_discharge:.2f} kg")
        print(f"    - Initial ice: {initial_ice:.2f} kg")
        print(f"    - Precipitation: {precipitation:.2f} kg")
        print(f"    - Deposition: {deposition:.2f} kg")
        print(f"  Total Out: {total_out:.2f} kg")
        print(f"    - Final ice: {final_ice:.2f} kg")
        print(f"    - Direct waste: {direct_waste:.2f} kg")
        print(f"    - Meltwater: {meltwater:.2f} kg")
        print(f"    - Sublimation: {sublimation:.2f} kg")
        print(f"  Balance (In - Out): {balance:.2f} kg ({pct_diff:.2f}% discrepancy)")
        
        balance_data.append({
            'spray': spray,
            'total_in': total_in,
            'water_input': fountain_discharge,
            'initial_ice': initial_ice,
            'precipitation': precipitation,
            'deposition': deposition,
            'total_out': total_out,
            'final_ice': final_ice,
            'direct_waste': direct_waste,
            'meltwater': meltwater,
            'sublimation': sublimation,
            'balance': balance,
            'balance_pct': pct_diff
        })
    
    # Save balance data to CSV
    balance_df = pd.DataFrame(balance_data)
    balance_df.to_csv(os.path.join(output_dir, f'{location.replace(" ", "_").lower()}_water_balance.csv'), index=False)
    
    return balance_df

def main():
    # Directory where results are stored
    base_dir = os.path.join(os.getcwd(), 'results_comparison')
    
    # Location to analyze
    location = "guttannen22"
    location_display = "Guttannen 2022"
    
    # Load the summary results
    summary_path = os.path.join(base_dir, f'{location}_summary.csv')
    
    if os.path.exists(summary_path):
        results_df = load_results(summary_path)
        
        # Update location name for display (code vs display name)
        results_df['location'] = location_display
        
        # Create visualizations
        plot_efficiency_comparison(results_df, location_display, base_dir)
        plot_water_flow(results_df, location_display, base_dir)
        balance_df = plot_water_balance(results_df, location_display, base_dir)
        
        # Try to plot ice volume over time from individual CSVs
        try:
            plot_ice_volume_over_time(location_display, base_dir)
        except Exception as e:
            print(f"Warning: Could not generate ice volume time plot: {e}")
        
        # Generate metrics table
        metrics_table = generate_metrics_table(results_df, location_display, base_dir)
        print("\nMetrics Summary:")
        print(metrics_table)
        
        print(f"\nVisualization complete. Results saved to {base_dir}")
    else:
        print(f"No summary file found at {summary_path}")
        print("Please run the comparison script first to generate results.")

if __name__ == "__main__":
    main() 