"""Script to visualize water efficiency metrics across spray methods
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

def load_results(location):
    """Load results for a specific location"""
    location_dir = location.lower().replace(" ", "")
    summary_file = f"{comparison_dir}/{location_dir}_summary.csv"
    
    if os.path.exists(summary_file):
        return pd.read_csv(summary_file)
    else:
        print(f"No results found for {location}")
        return None

def plot_water_flow_sankey(results, location_dir):
    """Create a Sankey diagram showing water flow through the system"""
    try:
        import plotly.graph_objects as go
        from plotly.offline import plot
    except ImportError:
        print("Plotly is required for Sankey diagrams. Install with: pip install plotly")
        return
    
    # Process each spray method
    for _, row in results.iterrows():
        spray = row['spray']
        
        # Define nodes
        node_labels = [
            "Water Input",           # 0
            "Fountain Discharge",    # 1
            "Precipitation",         # 2
            "Initial Ice",           # 3
            "Deposition",            # 4
            "Direct Waste",          # 5
            "Ice Mass",              # 6
            "Meltwater",             # 7
            "Sublimation",           # 8
            "Final Water Balance"    # 9
        ]
        
        # Get all the water components, ensuring they're all available
        discharge = row.get('total_water_input', 0)
        precipitation = row.get('M_ppt', 0)
        initial_ice = row.get('initial_ice_mass', 0)
        deposition = row.get('M_dep', 0) if 'M_dep' in row else 0
        
        direct_waste = row.get('total_direct_waste', 0)
        final_ice = row.get('final_ice_mass', 0)
        meltwater = row.get('total_meltwater', 0)
        sublimation = row.get('total_sublimation', 0)
        
        # Calculate total inputs and outputs for verification
        total_inputs = discharge + precipitation + initial_ice + deposition
        total_outputs = direct_waste + final_ice + meltwater + sublimation
        
        # Define links (source, target, value)
        links = [
            # Inputs → Input Node
            [1, 0, discharge],      # Discharge → Input
            [2, 0, precipitation],  # Precipitation → Input
            [3, 0, initial_ice],    # Initial Ice → Input
            [4, 0, deposition],     # Deposition → Input
            
            # Input Node → Outputs
            [0, 5, direct_waste],   # Input → Direct Waste
            [0, 6, final_ice],      # Input → Final Ice Mass
            [0, 7, meltwater],      # Input → Meltwater
            [0, 8, sublimation],    # Input → Sublimation
            
            # All outputs → Final Balance
            [5, 9, direct_waste],   # Direct Waste → Final Balance
            [6, 9, final_ice],      # Final Ice Mass → Final Balance
            [7, 9, meltwater],      # Meltwater → Final Balance
            [8, 9, sublimation],    # Sublimation → Final Balance
        ]
        
        # Create color scheme
        node_colors = [
            'gray',        # Water Input (central node)
            'blue',        # Fountain Discharge
            'lightblue',   # Precipitation
            'lightcyan',   # Initial Ice
            'skyblue',     # Deposition
            'red',         # Direct Waste
            'lightgreen',  # Ice Mass
            'orange',      # Meltwater
            'purple',      # Sublimation
            'gray'         # Final Water Balance
        ]
        
        link_colors = [
            'rgba(0, 0, 255, 0.4)',     # Discharge → Input
            'rgba(173, 216, 230, 0.4)',  # Precipitation → Input
            'rgba(224, 255, 255, 0.4)',  # Initial Ice → Input
            'rgba(135, 206, 235, 0.4)',  # Deposition → Input
            
            'rgba(255, 0, 0, 0.4)',     # Input → Direct Waste
            'rgba(144, 238, 144, 0.4)',  # Input → Final Ice Mass
            'rgba(255, 165, 0, 0.4)',   # Input → Meltwater
            'rgba(128, 0, 128, 0.4)',   # Input → Sublimation
            
            'rgba(255, 0, 0, 0.4)',     # Direct Waste → Final Balance
            'rgba(144, 238, 144, 0.4)',  # Final Ice Mass → Final Balance
            'rgba(255, 165, 0, 0.4)',   # Meltwater → Final Balance
            'rgba(128, 0, 128, 0.4)',   # Sublimation → Final Balance
        ]
        
        # Create figure
        fig = go.Figure(data=[go.Sankey(
            arrangement = "snap",
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color=node_colors
            ),
            link=dict(
                source=[link[0] for link in links],
                target=[link[1] for link in links],
                value=[link[2] for link in links],
                color=link_colors
            )
        )])
        
        # Update layout
        title = f"Complete Water Flow in the {spray} Method"
        fig.update_layout(
            title_text=title, 
            font_size=12,
            width=1000,
            height=800,
            annotations=[
                dict(
                    text=f"Total Inputs: {total_inputs:.2f} kg<br>Total Outputs: {total_outputs:.2f} kg<br>Balance: {total_inputs - total_outputs:.2f} kg ({(total_inputs - total_outputs)/total_inputs*100:.2f}%)",
                    showarrow=False,
                    x=0.01,
                    y=0.99,
                    font=dict(size=14)
                )
            ]
        )
        
        # Save figure
        output_file = f"{comparison_dir}/{location_dir}_{spray}_complete_water_flow.html"
        plot(fig, filename=output_file, auto_open=False)
        print(f"Saved complete water flow diagram to {output_file}")

def plot_simple_water_flow_sankey(results, location_dir):
    """Create a simple Sankey diagram showing water flow through the system"""
    try:
        import plotly.graph_objects as go
        from plotly.offline import plot
    except ImportError:
        print("Plotly is required for Sankey diagrams. Install with: pip install plotly")
        return
    
    # Process each spray method
    for _, row in results.iterrows():
        spray = row['spray']
        
        # Define nodes
        node_labels = [
            "Total Inputs",          # 0
            "Fountain Discharge",    # 1
            "Precipitation",         # 2
            "Initial Ice",           # 3
            "Deposition",            # 4
            "Ice Formation",         # 5
            "Direct Waste",          # 6
            "Remaining Ice",         # 7
            "Meltwater",             # 8
            "Sublimation",           # 9
            "Water Balance"          # 10
        ]
        
        # Get water components
        discharge = row.get('total_water_input', 0)
        precipitation = row.get('M_ppt', 0) if 'M_ppt' in row else 0
        initial_ice = row.get('initial_ice_mass', 0)
        deposition = row.get('M_dep', 0) if 'M_dep' in row else 0
        
        total_water_frozen = row.get('total_water_frozen', 0)
        direct_waste = row.get('total_direct_waste', 0)
        final_ice = row.get('final_ice_mass', 0)
        meltwater = row.get('total_meltwater', 0)
        sublimation = row.get('total_sublimation', 0) if 'total_sublimation' in row else 0
        
        # Calculate total inputs for verification
        total_inputs = discharge + precipitation + initial_ice + deposition
        
        # Define links (source, target, value)
        links = [
            # Inputs → Total Inputs node
            [1, 0, discharge],      # Discharge → Total Inputs
            [2, 0, precipitation],  # Precipitation → Total Inputs
            [3, 0, initial_ice],    # Initial Ice → Total Inputs
            [4, 0, deposition],     # Deposition → Total Inputs
            
            # Total Inputs → Ice Formation and Direct Waste
            [0, 5, total_water_frozen if total_water_frozen > 0 else initial_ice + final_ice - initial_ice],  # Total Inputs → Ice Formation
            [0, 6, direct_waste],   # Total Inputs → Direct Waste
            
            # Ice Formation → Outputs
            [5, 7, final_ice],      # Ice Formation → Remaining Ice
            [5, 8, meltwater],      # Ice Formation → Meltwater
            [5, 9, sublimation],    # Ice Formation → Sublimation
            
            # All outputs → Water Balance
            [6, 10, direct_waste],  # Direct Waste → Water Balance
            [7, 10, final_ice],     # Remaining Ice → Water Balance
            [8, 10, meltwater],     # Meltwater → Water Balance
            [9, 10, sublimation],   # Sublimation → Water Balance
        ]
        
        # Create color scheme
        node_colors = [
            'gray',        # Total Inputs (central node)
            'blue',        # Fountain Discharge
            'lightblue',   # Precipitation
            'lightcyan',   # Initial Ice
            'skyblue',     # Deposition
            'turquoise',   # Ice Formation
            'red',         # Direct Waste
            'lightgreen',  # Remaining Ice
            'orange',      # Meltwater
            'purple',      # Sublimation
            'gray'         # Water Balance
        ]
        
        link_colors = [
            'rgba(0, 0, 255, 0.4)',      # Discharge → Total Inputs
            'rgba(173, 216, 230, 0.4)',  # Precipitation → Total Inputs
            'rgba(224, 255, 255, 0.4)',  # Initial Ice → Total Inputs
            'rgba(135, 206, 235, 0.4)',  # Deposition → Total Inputs
            
            'rgba(64, 224, 208, 0.4)',   # Total Inputs → Ice Formation
            'rgba(255, 0, 0, 0.4)',      # Total Inputs → Direct Waste
            
            'rgba(144, 238, 144, 0.4)',  # Ice Formation → Remaining Ice
            'rgba(255, 165, 0, 0.4)',    # Ice Formation → Meltwater
            'rgba(128, 0, 128, 0.4)',    # Ice Formation → Sublimation
            
            'rgba(255, 0, 0, 0.4)',      # Direct Waste → Water Balance
            'rgba(144, 238, 144, 0.4)',  # Remaining Ice → Water Balance
            'rgba(255, 165, 0, 0.4)',    # Meltwater → Water Balance
            'rgba(128, 0, 128, 0.4)',    # Sublimation → Water Balance
        ]
        
        # Create figure
        fig = go.Figure(data=[go.Sankey(
            arrangement = "snap",
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color=node_colors
            ),
            link=dict(
                source=[link[0] for link in links],
                target=[link[1] for link in links],
                value=[link[2] for link in links],
                color=link_colors
            )
        )])
        
        # Update layout
        title = f"Water Flow in the {spray} Method"
        fig.update_layout(
            title_text=title, 
            font_size=12,
            width=1000,
            height=800,
            annotations=[
                dict(
                    text=f"Water Use Efficiency: {row.get('water_use_efficiency', 0):.2f}%<br>Direct Waste Ratio: {row.get('direct_waste_ratio', 0):.2f}%",
                    showarrow=False,
                    x=0.01,
                    y=0.99,
                    font=dict(size=14)
                )
            ]
        )
        
        # Save figure
        output_file = f"{comparison_dir}/{location_dir}_{spray}_water_flow.html"
        plot(fig, filename=output_file, auto_open=False)
        print(f"Saved water flow diagram to {output_file}")

def plot_water_balance_stacked(results):
    """Create a stacked bar chart showing water balance for each spray method"""
    if results is None or len(results) == 0:
        return
    
    # Prepare data
    plot_data = results.copy()
    
    # Calculate water components as percentages if possible
    if all(col in plot_data.columns for col in ['total_water_input', 'final_ice_mass', 'total_direct_waste', 'total_meltwater']):
        for idx, row in plot_data.iterrows():
            total = row['total_water_input']
            if total > 0:
                plot_data.loc[idx, 'final_ice_pct'] = (row['final_ice_mass'] / total) * 100
                plot_data.loc[idx, 'direct_waste_pct'] = (row['total_direct_waste'] / total) * 100
                plot_data.loc[idx, 'meltwater_pct'] = (row['total_meltwater'] / total) * 100
                if 'total_sublimation' in row:
                    plot_data.loc[idx, 'sublimation_pct'] = (row['total_sublimation'] / total) * 100
                else:
                    plot_data.loc[idx, 'sublimation_pct'] = 0

    # Set up the figure
    plt.figure(figsize=(12, 8))
    
    # Create the stacked bar chart
    bar_width = 0.6
    
    # Organize data for stacked bars
    sprays = plot_data['spray'].tolist()
    final_ice = plot_data.get('final_ice_pct', [0] * len(sprays)).tolist()
    direct_waste = plot_data.get('direct_waste_pct', [0] * len(sprays)).tolist()
    meltwater = plot_data.get('meltwater_pct', [0] * len(sprays)).tolist()
    sublimation = plot_data.get('sublimation_pct', [0] * len(sprays)).tolist()
    
    # Create bars
    plt.bar(sprays, final_ice, bar_width, label='Remaining Ice', color='lightgreen')
    plt.bar(sprays, direct_waste, bar_width, bottom=final_ice, label='Direct Waste', color='lightcoral')
    
    # Calculate the position for the meltwater bars
    bottom_values = [a + b for a, b in zip(final_ice, direct_waste)]
    plt.bar(sprays, meltwater, bar_width, bottom=bottom_values, label='Meltwater', color='lightskyblue')
    
    # Add sublimation if available
    if 'sublimation_pct' in plot_data.columns and any(plot_data['sublimation_pct'] > 0):
        bottom_values = [a + b + c for a, b, c in zip(final_ice, direct_waste, meltwater)]
        plt.bar(sprays, sublimation, bar_width, bottom=bottom_values, label='Sublimation', color='lightgray')
    
    # Add a horizontal line at 100%
    plt.axhline(y=100, color='black', linestyle='--', alpha=0.3)
    
    # Add labels and title
    plt.xlabel('Spray Method', fontsize=14)
    plt.ylabel('Percentage of Total Water Input (%)', fontsize=14)
    plt.title(f'Water Balance Comparison for {results.iloc[0]["location"]}', fontsize=16)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=12)
    
    # Add value labels on each segment
    for i, spray in enumerate(sprays):
        # Ice remaining
        if final_ice[i] > 0:
            plt.text(i, final_ice[i]/2, f'{final_ice[i]:.1f}%', 
                    ha='center', va='center', fontsize=11, color='black')
        
        # Direct waste
        if direct_waste[i] > 0:
            plt.text(i, final_ice[i] + direct_waste[i]/2, f'{direct_waste[i]:.1f}%', 
                    ha='center', va='center', fontsize=11, color='black')
        
        # Meltwater
        if meltwater[i] > 0:
            plt.text(i, final_ice[i] + direct_waste[i] + meltwater[i]/2, f'{meltwater[i]:.1f}%', 
                    ha='center', va='center', fontsize=11, color='black')
    
    plt.tight_layout()
    
    # Save the figure
    location_dir = results.iloc[0]["location"].lower().replace(" ", "")
    plt.savefig(f"{comparison_dir}/{location_dir}_water_balance.png", dpi=300)
    print(f"Saved water balance comparison to {comparison_dir}/{location_dir}_water_balance.png")

def plot_efficiency_metrics(results):
    """Create a grouped bar chart comparing efficiency metrics"""
    if results is None or len(results) == 0:
        return
    
    # Check which efficiency metrics are available
    efficiency_metrics = [col for col in ['water_use_efficiency', 'freezing_efficiency', 'melt_ratio', 'direct_waste_ratio'] 
                         if col in results.columns]
    
    if not efficiency_metrics:
        print("No efficiency metrics available to plot")
        return
    
    # Melt the dataframe to have metrics in one column
    plot_data = pd.melt(results, 
                        id_vars=['spray'], 
                        value_vars=efficiency_metrics, 
                        var_name='Metric', 
                        value_name='Value')
    
    # Create nicer labels for the metrics
    metric_labels = {
        'water_use_efficiency': 'Water Use\nEfficiency (%)',
        'freezing_efficiency': 'Freezing\nEfficiency (%)',
        'stupa_retention_efficiency': 'Stupa Retention\nEfficiency (%)', 
        'direct_waste_ratio': 'Direct Waste\nRatio (%)'
    }
    plot_data['Metric'] = plot_data['Metric'].map(lambda x: metric_labels.get(x, x))
    
    # Set up the figure
    plt.figure(figsize=(14, 8))
    
    # Create the grouped bar chart
    ax = sns.barplot(x='Metric', y='Value', hue='spray', data=plot_data)
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', fontsize=9)
    
    # Add labels and title
    plt.xlabel('Efficiency Metric', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.title(f'Water Efficiency Metrics for {results.iloc[0]["location"]}', fontsize=16)
    plt.legend(title='Spray Method')
    
    plt.tight_layout()
    
    # Save the figure
    location_dir = results.iloc[0]["location"].lower().replace(" ", "")
    plt.savefig(f"{comparison_dir}/{location_dir}_efficiency_metrics.png", dpi=300)
    print(f"Saved efficiency metrics to {comparison_dir}/{location_dir}_efficiency_metrics.png")

def plot_water_volumes(results):
    """Create a grouped bar chart showing absolute water volumes"""
    if results is None or len(results) == 0:
        return
    
    # Check which volume metrics are available
    volume_metrics = [col for col in ['total_water_input', 'total_water_frozen', 'final_ice_mass', 
                                    'total_direct_waste', 'total_meltwater'] 
                     if col in results.columns]
    
    if not volume_metrics:
        print("No volume metrics available to plot")
        return
    
    # Convert to m³ for better readability (assuming kg, with density ~1000 kg/m³)
    plot_data = results.copy()
    for col in volume_metrics:
        plot_data[f'{col}_m3'] = plot_data[col] / 1000
    
    # Adjust column names for plot
    volume_metrics_m3 = [f'{col}_m3' for col in volume_metrics]
    
    # Melt the dataframe to have metrics in one column
    plot_data = pd.melt(plot_data, 
                        id_vars=['spray'], 
                        value_vars=volume_metrics_m3, 
                        var_name='Metric', 
                        value_name='Volume (m³)')
    
    # Create nicer labels for the metrics
    metric_labels = {
        'total_water_input_m3': 'Total Water\nInput',
        'total_water_frozen_m3': 'Total Water\nFrozen',
        'final_ice_mass_m3': 'Final Ice\nMass',
        'total_direct_waste_m3': 'Total Direct\nWaste',
        'total_meltwater_m3': 'Total\nMeltwater'
    }
    plot_data['Metric'] = plot_data['Metric'].map(lambda x: metric_labels.get(x, x))
    
    # Set up the figure
    plt.figure(figsize=(14, 8))
    
    # Create the grouped bar chart
    ax = sns.barplot(x='Metric', y='Volume (m³)', hue='spray', data=plot_data)
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', fontsize=9)
    
    # Add labels and title
    plt.xlabel('Water Volume Category', fontsize=14)
    plt.ylabel('Volume (m³)', fontsize=14)
    plt.title(f'Water Volumes for {results.iloc[0]["location"]}', fontsize=16)
    plt.legend(title='Spray Method')
    
    plt.tight_layout()
    
    # Save the figure
    location_dir = results.iloc[0]["location"].lower().replace(" ", "")
    plt.savefig(f"{comparison_dir}/{location_dir}_water_volumes.png", dpi=300)
    print(f"Saved water volumes chart to {comparison_dir}/{location_dir}_water_volumes.png")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize water efficiency metrics')
    parser.add_argument('--location', type=str, default="Guttannen 2022",
                        help='Location to analyze')
    parser.add_argument('--sankey', action='store_true',
                        help='Generate simple Sankey flow diagrams (requires plotly)')
    parser.add_argument('--complete-sankey', action='store_true',
                        help='Generate comprehensive Sankey diagrams with all water components (requires plotly)')
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.location)
    
    if results is not None:
        # Generate standard charts
        plot_water_balance_stacked(results)
        plot_efficiency_metrics(results)
        plot_water_volumes(results)
        
        # Optional Sankey diagrams if requested
        location_dir = args.location.lower().replace(" ", "")
        if args.sankey:
            plot_simple_water_flow_sankey(results, location_dir)
            
        if args.complete_sankey:
            # Preserve the original function and create a new one for the complete diagram
            from types import FunctionType
            original_sankey_fn = plot_water_flow_sankey
            
            # If both are requested, temporarily rename the enhanced function to avoid conflict
            if args.sankey and args.complete_sankey:
                # Store and generate the complete diagrams
                globals()['plot_complete_water_flow_sankey'] = globals()['plot_water_flow_sankey']
                globals()['plot_water_flow_sankey'] = original_sankey_fn
            
            # Generate the complete Sankey diagrams
            if 'plot_complete_water_flow_sankey' in globals():
                globals()['plot_complete_water_flow_sankey'](results, location_dir)
            else:
                plot_water_flow_sankey(results, location_dir)
        
        print(f"\nVisualization complete for {args.location}")
    else:
        print(f"No data available for {args.location}") 