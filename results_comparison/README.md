# Ice Stupa Model Comparison Tools

This directory contains tools and results for comparing different spray methods across various locations in the Ice Stupa model.

## Directory Structure

```
results_comparison/
├── README.md
├── all_results_summary.csv        # Combined results from all locations
├── {location}_summary.csv         # Results for specific locations
├── method_rankings.csv            # Best methods by metric for each location
├── *.png                          # Various visualization plots
├── {location}/                    # Folders for each location
│   ├── scheduled_field/           # Results for scheduled_field method
│   │   ├── output.csv             # Simulation output data
│   │   ├── output.png             # Main visualization
│   │   ├── Vol_Validation.png     # Volume validation plot
│   │   └── Discharge.png          # Discharge visualization
│   ├── unscheduled_field/         # Results for unscheduled_field method
│   ├── scheduled_wue/             # Results for scheduled_wue method
│   └── scheduled_icv/             # Results for scheduled_icv method
```

## Available Locations

The comparison tools are set up for the following locations:

- Guttannen 2020 (code: guttannen20)
- Guttannen 2021 (code: guttannen21)
- Guttannen 2022 (code: guttannen22)
- Gangles 2021 (code: gangles21)

## Spray Methods

The following spray methods are compared:

- `scheduled_field`: Scheduled field spraying
- `unscheduled_field`: Unscheduled field spraying
- `scheduled_wue`: Scheduled Water Use Efficiency optimization
- `scheduled_icv`: Scheduled Ice Volume optimization

## Metrics

The comparison tools track and visualize the following metrics:

- `ice_volume_max`: Maximum ice volume achieved (m³)
- `total_water_input`: Total water sprayed (kg)
- `total_water_frozen`: Total water that froze (kg)
- `total_direct_waste`: Total water wasted directly (not frozen) (kg)
- `total_meltwater`: Total water lost to melting (kg)
- `total_sublimation`: Total water lost to sublimation (kg)
- `final_ice_mass`: Final ice mass at the end of simulation (kg)
- `water_use_efficiency`: Percentage of input water that ends up as ice (%)
- `freezing_efficiency`: Percentage of input water that gets frozen (%)
- `stupa_retention_efficiency`: Percentage of frozen water that remains as ice (%)
- `direct_waste_ratio`: Percentage of input water directly wasted (%)
- `overall_water_loss_ratio`: Percentage of input water lost (direct waste + meltwater) (%)

## Usage

### Running Comparisons

To run simulations for all locations and methods:

```bash
python run_all_comparisons.py
```

To run simulations for a specific location (e.g., Guttannen 2022):

```bash
python run_comparison.py
```

### Visualizing Results

To create visualizations for all locations:

```bash
python visualize_all_comparisons.py
```

To create visualizations for a specific location (e.g., Guttannen 2022):

```bash
python visualize_comparison.py
```

## Interpretation

### Water Use Efficiency (WUE)

Water Use Efficiency (WUE) is defined as the percentage of input water that ends up as ice at the end of the simulation. Higher values indicate more efficient use of water.

### Freezing Efficiency

Freezing Efficiency is the percentage of input water that gets frozen during spraying. Higher values indicate better initial freezing performance.

### Stupa Retention Efficiency

Stupa Retention Efficiency is the percentage of frozen water that remains as ice at the end of the simulation. Higher values indicate better resistance to melting.

### Direct Waste Ratio

Direct Waste Ratio is the percentage of input water that is directly wasted during spraying (not frozen). Lower values indicate less waste during spraying.

### Overall Water Loss Ratio

Overall Water Loss Ratio is the percentage of input water that is lost either through direct waste or melting. Lower values indicate better overall water conservation. 