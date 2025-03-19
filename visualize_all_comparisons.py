"""Script to visualize comparisons for all locations
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import glob
import argparse

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Directory setup
dirname = os.path.dirname(os.path.realpath(__file__))
comparison_dir = f"{dirname}/results_comparison"

def load_all_results(base_dir):
    """Load the combined results summary CSV."""
    summary_path = os.path.join(base_dir, "all_results_summary.csv")
    if os.path.exists(summary_path):
        return pd.read_csv(summary_path)
    else:
        print(f"No summary file found at {summary_path}")
        return None

def load_results(location):
    """Load results for a specific location"""
    location_dir = location.lower().replace(" ", "")
    summary_file = f"{comparison_dir}/{location_dir}_summary.csv"
    
    if os.path.exists(summary_file):
        return pd.read_csv(summary_file)
    else:
        print(f"No results found for {location}")
        return None

def plot_individual_location(results, metric, ylabel, title_suffix):
    """Create a bar chart for a single location"""
    if results is None or len(results) == 0:
        return
    
    location = results.iloc[0]["location"]
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='spray', y=metric, data=results)
    
    # Add value labels on top of bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.1f}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'bottom',
                   xytext = (0, 5), textcoords = 'offset points')
    
    plt.title(f'{title_suffix} for {location}', fontsize=16)
    plt.xlabel('Spray Method', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the figure
    location_dir = location.lower().replace(" ", "")
    plt.savefig(f"{comparison_dir}/{location_dir}_{metric}_comparison.png", dpi=300)
    print(f"Saved {metric} comparison to {comparison_dir}/{location_dir}_{metric}_comparison.png")

def plot_combined_locations(all_results, metric, ylabel, title):
    """Create a grouped bar chart comparing all locations"""
    if all_results is None or len(all_results) == 0:
        return
    
    plt.figure(figsize=(12, 8))
    ax = sns.catplot(
        data=all_results, 
        kind="bar",
        x="spray", y=metric, hue="location",
        height=6, aspect=1.5
    )
    
    plt.title(title, fontsize=16)
    plt.xlabel('Spray Method', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"{comparison_dir}/combined_{metric}_comparison.png", dpi=300)
    print(f"Saved combined {metric} comparison to {comparison_dir}/combined_{metric}_comparison.png")

def visualize_location(location):
    """Create visualizations for a specific location"""
    results = load_results(location)
    
    if results is not None:
        # Calculate efficiency
        results = results.copy()
        if 'total_meltwater' in results.columns and 'ice_volume_max' in results.columns:
            results['efficiency'] = results['ice_volume_max'] / (results['total_meltwater'] / 1000)  # m³ per m³
            results['total_meltwater_m3'] = results['total_meltwater'] / 1000  # Convert to m³
        
        # Create individual visualizations
        plot_individual_location(
            results, 
            'ice_volume_max', 
            'Maximum Ice Volume (m³)',
            'Maximum Ice Volume Comparison'
        )
        
        if 'total_meltwater_m3' in results.columns:
            plot_individual_location(
                results, 
                'total_meltwater_m3', 
                'Total Meltwater (m³)',
                'Total Meltwater Comparison'
            )
            
        if 'efficiency' in results.columns:
            plot_individual_location(
                results, 
                'efficiency', 
                'Efficiency (Max Ice Volume / Total Meltwater)',
                'Ice Storage Efficiency Comparison'
            )
        
        print(f"\nSummary of results for {location}:")
        print(results)

def visualize_all_locations():
    """Create visualizations combining all locations"""
    all_results = load_all_results(comparison_dir)
    
    if all_results is None or all_results.empty:
        print("No results to visualize.")
        return
    
    # Calculate efficiency
    all_results = all_results.copy()
    if 'total_meltwater' in all_results.columns and 'ice_volume_max' in all_results.columns:
        all_results['efficiency'] = all_results['ice_volume_max'] / (all_results['total_meltwater'] / 1000)  # m³ per m³
        all_results['total_meltwater_m3'] = all_results['total_meltwater'] / 1000  # Convert to m³
    
    # Create combined visualizations
    plot_combined_locations(
        all_results, 
        'ice_volume_max', 
        'Maximum Ice Volume (m³)',
        'Maximum Ice Volume Comparison Across Locations'
    )
    
    if 'total_meltwater_m3' in all_results.columns:
        plot_combined_locations(
            all_results, 
            'total_meltwater_m3', 
            'Total Meltwater (m³)',
            'Total Meltwater Comparison Across Locations'
        )
        
    if 'efficiency' in all_results.columns:
        plot_combined_locations(
            all_results, 
            'efficiency', 
            'Efficiency (Max Ice Volume / Total Meltwater)',
            'Ice Storage Efficiency Comparison Across Locations'
        )
    
    print("\nSummary of all results:")
    print(all_results)

def plot_efficiency_comparison_by_location(df, output_dir):
    """Plot efficiency metrics grouped by location."""
    # Create figure for each efficiency metric
    metrics = [
        ('water_use_efficiency', 'Water Use Efficiency (%)'),
        ('freezing_efficiency', 'Freezing Efficiency (%)'),
        ('stupa_retention_efficiency', 'Stupa Retention Efficiency (%)'),
        ('direct_waste_ratio', 'Direct Waste Ratio (%)')
    ]
    
    for metric, title in metrics:
        plt.figure(figsize=(14, 8))
        
        # Filter rows with valid values
        plot_df = df[df[metric].notna() & ~df[metric].isin([float('inf'), float('-inf')])]
        
        if plot_df.empty:
            print(f"No valid data for {metric}")
            continue
            
        # Create grouped bar plot
        ax = sns.barplot(x='location', y=metric, hue='spray', data=plot_df)
        
        plt.title(f'{title} Comparison Across Locations', fontsize=16)
        plt.ylabel(title, fontsize=14)
        plt.xlabel('Location', fontsize=14)
        plt.xticks(rotation=45)
        plt.legend(title='Spray Method')
        plt.grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'all_locations_{metric}.png'))
        plt.close()

def plot_ice_volume_comparison(df, output_dir):
    """Plot ice volume max comparison across all locations."""
    plt.figure(figsize=(14, 8))
    
    # Create grouped bar plot
    ax = sns.barplot(x='location', y='ice_volume_max', hue='spray', data=df)
    
    plt.title('Maximum Ice Volume Comparison Across Locations', fontsize=16)
    plt.ylabel('Maximum Ice Volume (m³)', fontsize=14)
    plt.xlabel('Location', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='Spray Method')
    plt.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_locations_ice_volume.png'))
    plt.close()

def plot_water_components_by_spray(df, output_dir):
    """Plot water components grouped by spray method."""
    # Create a figure showing water components for each spray method
    spray_methods = df['spray'].unique()
    
    # Components to visualize
    components = [
        'total_water_input',
        'total_water_frozen',
        'total_direct_waste',
        'total_meltwater',
        'total_sublimation',
        'final_ice_mass'
    ]
    
    component_labels = {
        'total_water_input': 'Total Water Input',
        'total_water_frozen': 'Total Water Frozen',
        'total_direct_waste': 'Direct Waste',
        'total_meltwater': 'Meltwater',
        'total_sublimation': 'Sublimation',
        'final_ice_mass': 'Final Ice Mass'
    }
    
    for spray in spray_methods:
        plt.figure(figsize=(14, 8))
        
        # Filter data for this spray method
        spray_df = df[df['spray'] == spray].copy()
        
        # Melt dataframe for plotting
        melted_df = pd.melt(
            spray_df,
            id_vars=['location'],
            value_vars=components,
            var_name='Component',
            value_name='Volume (kg)'
        )
        
        # Map component names to readable labels
        melted_df['Component'] = melted_df['Component'].map(component_labels)
        
        # Create plot
        sns.barplot(x='location', y='Volume (kg)', hue='Component', data=melted_df)
        
        plt.title(f'Water Components for {spray}', fontsize=16)
        plt.ylabel('Volume (kg)', fontsize=14)
        plt.xlabel('Location', fontsize=14)
        plt.xticks(rotation=45)
        plt.legend(title='Component')
        plt.yscale('log')  # Log scale for better visibility of all components
        plt.grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{spray}_water_components.png'))
        plt.close()

def create_efficiency_heatmap(df, output_dir):
    """Create a heatmap of efficiency metrics."""
    # Efficiency metrics to visualize
    metrics = [
        'water_use_efficiency',
        'freezing_efficiency',
        'stupa_retention_efficiency'
    ]
    
    metric_labels = {
        'water_use_efficiency': 'Water Use Efficiency',
        'freezing_efficiency': 'Freezing Efficiency',
        'stupa_retention_efficiency': 'Stupa Retention Efficiency'
    }
    
    # Create a pivot table for the heatmap (location x spray with values as efficiency)
    for metric in metrics:
        try:
            pivot_df = df.pivot(index='location', columns='spray', values=metric)
            
            plt.figure(figsize=(12, 8))
            
            # Create heatmap
            sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.2f', linewidths=0.5)
            
            plt.title(f'{metric_labels[metric]} Across Locations and Spray Methods', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{metric}_heatmap.png'))
            plt.close()
        except Exception as e:
            print(f"Error creating heatmap for {metric}: {e}")

def generate_ranked_methods_table(df, output_dir):
    """Generate a table showing the ranked methods for each location based on different metrics."""
    metrics = [
        'ice_volume_max',
        'water_use_efficiency',
        'freezing_efficiency'
    ]
    
    # Create a summary dataframe for rankings
    rankings = pd.DataFrame(columns=['location', 'best_ice_volume', 'best_water_efficiency', 'best_freezing_efficiency'])
    
    # Process each location
    for location in df['location'].unique():
        location_df = df[df['location'] == location]
        
        # Find best method for each metric
        best_ice_volume = location_df.loc[location_df['ice_volume_max'].idxmax()]['spray'] if not location_df['ice_volume_max'].isna().all() else 'N/A'
        best_wue = location_df.loc[location_df['water_use_efficiency'].idxmax()]['spray'] if not location_df['water_use_efficiency'].isna().all() else 'N/A'
        best_freeze = location_df.loc[location_df['freezing_efficiency'].idxmax()]['spray'] if not location_df['freezing_efficiency'].isna().all() else 'N/A'
        
        # Add to rankings dataframe
        rankings = rankings.append({
            'location': location,
            'best_ice_volume': best_ice_volume,
            'best_water_efficiency': best_wue,
            'best_freezing_efficiency': best_freeze
        }, ignore_index=True)
    
    # Save to CSV
    rankings.to_csv(os.path.join(output_dir, 'method_rankings.csv'), index=False)
    return rankings

def main():
    # Directory where results are stored
    base_dir = comparison_dir
    
    # Load the summary results
    results_df = load_all_results(base_dir)
    
    if results_df is None or results_df.empty:
        print("No results to visualize.")
        return
    
    # Create visualizations
    plot_efficiency_comparison_by_location(results_df, base_dir)
    plot_ice_volume_comparison(results_df, base_dir)
    plot_water_components_by_spray(results_df, base_dir)
    create_efficiency_heatmap(results_df, base_dir)
    
    # Generate rankings table
    rankings = generate_ranked_methods_table(results_df, base_dir)
    
    print("\nBest Methods by Location:")
    print(rankings)
    
    print(f"\nVisualization complete. Results saved to {base_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize comparisons for different locations')
    parser.add_argument('--locations', nargs='+', 
                        default=["Guttannen 2020", "Guttannen 2021", "Guttannen 2022", "Gangles 2021"],
                        help='List of locations to visualize')
    parser.add_argument('--all', action='store_true', 
                        help='Visualize all locations in combined plots')
    args = parser.parse_args()
    
    if args.all:
        visualize_all_locations()
    else:
        for location in args.locations:
            visualize_location(location)
        main() 