#!/bin/bash
# Automate running make_dataset.py and userApp.py for all country folders

# Base directory
BASE_DIR="data"

# Log file
LOG_FILE="icestupa_automation.log"

# Function to log messages
log_message() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Clear or create log file
>"$LOG_FILE"

# Arrays to store results
SUCCESSFUL=()
FAILED=()

# Count folders
FOLDERS=($(ls -d "$BASE_DIR"/*/ 2>/dev/null))
TOTAL=${#FOLDERS[@]}

if [ $TOTAL -eq 0 ]; then
  log_message "No country folders found in $BASE_DIR"
  exit 1
fi

log_message "Found $TOTAL country folders to process"

# Process each folder
for ((i = 0; i < $TOTAL; i++)); do
  FOLDER=${FOLDERS[$i]}
  COUNTRY=$(basename "$FOLDER")

  log_message "==================================================="
  log_message "Processing ($((i + 1))/$TOTAL): $FOLDER"
  log_message "==================================================="

  # Create era5 subfolder if it doesn't exist
  ERA5_FOLDER="$FOLDER/era5"
  if [ ! -d "$ERA5_FOLDER" ]; then
    log_message "Creating missing subfolder: $ERA5_FOLDER"
    mkdir -p "$ERA5_FOLDER"
  fi

  # Step 1: Run make_dataset.py
  log_message "Running make_dataset.py..."
  python src/data/make_dataset.py --datadir "$FOLDER" 2>&1 | tee -a "$LOG_FILE"

  if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log_message "Error in make_dataset.py for $FOLDER"
    FAILED+=("$FOLDER (make_dataset.py failed)")
    continue
  fi

  # Wait for filesystem operations to complete
  log_message "Waiting for file operations to complete..."
  sleep 5

  # Step 2: Run userApp.py
  log_message "Running userApp.py..."
  python src/models/userApp.py --datadir "$FOLDER" 2>&1 | tee -a "$LOG_FILE"

  if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log_message "Error in userApp.py for $FOLDER"
    FAILED+=("$FOLDER (userApp.py failed)")
    continue
  fi

  log_message "Successfully processed $FOLDER"
  SUCCESSFUL+=("$FOLDER")
done

# Print summary
log_message "==================================================="
log_message "PROCESSING SUMMARY"
log_message "==================================================="
log_message "Successfully processed: ${#SUCCESSFUL[@]}/$TOTAL folders"

if [ ${#FAILED[@]} -gt 0 ]; then
  log_message "Failed folders:"
  for FOLDER in "${FAILED[@]}"; do
    log_message "  - $FOLDER"
  done
fi

if [ ${#SUCCESSFUL[@]} -gt 0 ]; then
  log_message "Successful folders:"
  for FOLDER in "${SUCCESSFUL[@]}"; do
    log_message "  - $FOLDER"
  done
fi

log_message "Automation completed!"
