# Ice Stupa Model Comparison Tools

This document provides an overview of the tools created for comparing different spray methods across various locations in the Ice Stupa Model.

## Available Tools

1. **run_comparison.py** - Run simulations for a single location using all spray methods
2. **run_all_comparisons.py** - Run simulations for multiple locations using all spray methods
3. **visualize_comparison.py** - Create visualizations for a single location
4. **visualize_all_comparisons.py** - Create visualizations for multiple locations, including combined plots

## Spray Methods

Four different spray methods can be compared:

1. **scheduled_field** - Scheduled spraying with field-based measurements
2. **unscheduled_field** - Unscheduled (continuous) spraying with field-based measurements
3. **scheduled_wue** - Scheduled spraying optimized for water use efficiency
4. **scheduled_icv** - Scheduled spraying optimized for ice volume

## How to Use

### Running a Single Location

To run simulations for a single location (e.g., Guttannen 2022):

```bash
conda activate air_model
python run_comparison.py
```

This will run simulations for all four spray methods and save the results to the `results_comparison/guttannen2022/` directory.

### Running Multiple Locations

To run simulations for all available locations:

```bash
conda activate air_model
python run_all_comparisons.py
```

You can also specify specific locations:

```bash
conda activate air_model
python run_all_comparisons.py --locations "Guttannen 2020" "Gangles 2021"
```

### Visualizing Results for a Single Location

To create visualizations for a single location:

```bash
conda activate air_model
python visualize_comparison.py
```

This will create three comparison charts:
- Maximum ice volume
- Total meltwater
- Ice storage efficiency

### Visualizing Results for Multiple Locations

To create visualizations for multiple locations, including combined charts:

```bash
conda activate air_model
python visualize_all_comparisons.py --all
```

To visualize specific locations only:

```bash
conda activate air_model
python visualize_all_comparisons.py --locations "Guttannen 2020" "Gangles 2021"
```

## Output Directory Structure

Results are saved in the `results_comparison` directory with the following structure:

```
results_comparison/
├── README.md                          # Overview of comparison results
├── guttannen2020/                     # Results for Guttannen 2020
│   ├── scheduled_field/               # Results for scheduled_field method
│   │   ├── output.csv                 # Simulation output data
│   │   ├── output.png                 # Main output visualization
│   │   ├── Vol_Validation.png         # Volume validation plot
│   │   └── Discharge.png              # Discharge plot
│   ├── unscheduled_field/             # Results for unscheduled_field method
│   ├── scheduled_wue/                 # Results for scheduled_wue method
│   └── scheduled_icv/                 # Results for scheduled_icv method
├── guttannen2021/                     # Results for Guttannen 2021
├── guttannen2022/                     # Results for Guttannen 2022
├── gangles2021/                       # Results for Gangles 2021
├── guttannen2020_summary.csv          # Summary data for Guttannen 2020
├── guttannen2021_summary.csv          # Summary data for Guttannen 2021
├── guttannen2022_summary.csv          # Summary data for Guttannen 2022
├── gangles2021_summary.csv            # Summary data for Gangles 2021
├── guttannen2020_ice_volume_comparison.png   # Ice volume chart for Guttannen 2020
├── guttannen2020_meltwater_comparison.png    # Meltwater chart for Guttannen 2020
├── guttannen2020_efficiency_comparison.png   # Efficiency chart for Guttannen 2020
└── combined_ice_volume_comparison.png        # Combined ice volume chart for all locations
```

## Key Metrics

The following metrics are used for comparison:

1. **Maximum Ice Volume (m³)** - The largest volume of ice achieved during the simulation period
2. **Total Meltwater (m³)** - The total amount of meltwater produced during the melt season
3. **Efficiency** - The ratio of maximum ice volume to total meltwater (higher is better)

## Contributing

To add a new spray method for comparison:

1. Implement the method in the appropriate location in the codebase
2. Update the spray method lists in the comparison scripts
3. Run the simulations and visualizations to include the new method

## Troubleshooting

- If you encounter any errors during simulation, check the error messages and ensure the required data is available for each location
- If visualizations fail, ensure the simulations have completed successfully and the CSV files are properly formatted
- If you see warnings about deprecated pandas dtype operations, these are harmless and will be addressed in future updates 