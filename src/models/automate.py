#!/usr/bin/env python3
"""
Automation script to run make_dataset.py and userApp.py for all countries in the data directory
"""

import os
import subprocess
import time
import sys

def main():
    # Base data directory
    base_dir = "data"
    
    # Get all country folders in the base directory
    country_folders = [f.path for f in os.scandir(base_dir) if f.is_dir()]
    
    if not country_folders:
        print(f"No country folders found in {base_dir}")
        return
    
    print(f"Found {len(country_folders)} country folders to process")
    
    # Track successful and failed folders
    successful = []
    failed = []
    
    # Process each country folder
    for country_folder in country_folders:
        print(f"\n{'='*50}")
        print(f"Processing: {country_folder}")
        print(f"{'='*50}")
        
        try:
            # Verify the folder exists
            if not os.path.exists(country_folder):
                print(f"Warning: Folder {country_folder} does not exist. Skipping.")
                failed.append((country_folder, "Folder does not exist"))
                continue
            
            # Create era5 subfolder if it doesn't exist
            era5_folder = os.path.join(country_folder, "era5")
            if not os.path.exists(era5_folder):
                print(f"Creating missing subfolder: {era5_folder}")
                os.makedirs(era5_folder, exist_ok=True)
                
            # Step 1: Run make_dataset.py with the country folder
            print("\nRunning make_dataset.py...")
            make_dataset_cmd = ["python", "src/data/make_dataset.py", "--datadir", country_folder]
            make_result = subprocess.run(make_dataset_cmd, check=False, capture_output=True, text=True)
            
            if make_result.returncode != 0:
                print(f"Error in make_dataset.py: {make_result.stderr}")
                failed.append((country_folder, "make_dataset.py failed"))
                continue
            else:
                print(make_result.stdout)  # Print stdout for successful run
            
            # Step 2: Run userApp.py with the country folder
            print("\nRunning userApp.py...")
            user_app_cmd = ["python", "src/models/userApp.py", "--datadir", country_folder]
            app_result = subprocess.run(user_app_cmd, check=False, capture_output=True, text=True)
            
            if app_result.returncode != 0:
                print(f"Error in userApp.py: {app_result.stderr}")
                failed.append((country_folder, "userApp.py failed"))
                continue
            else:
                print(app_result.stdout)  # Print stdout for successful run
            
            print(f"\nCompleted processing: {country_folder}")
            successful.append(country_folder)
            
        except Exception as e:
            print(f"Error processing {country_folder}: {str(e)}")
            failed.append((country_folder, str(e)))
    
    # Print summary
    print("\n\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Successfully processed: {len(successful)}/{len(country_folders)} folders")
    
    if failed:
        print("\nFailed folders:")
        for folder, reason in failed:
            print(f"  - {folder}: {reason}")
    
    if successful:
        print("\nSuccessful folders:")
        for folder in successful:
            print(f"  - {folder}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = (time.time() - start_time) / 60
    print(f"\nTotal execution time: {round(elapsed_time, 2)} minutes")
