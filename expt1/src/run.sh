#!/bin/bash

# Set environment name and script path
ENV_NAME="/home/rjaditya/miniconda3/envs/MSP_KAN"
SCRIPT_PATH="/home/rjaditya/KAN-1/expt1/src/main.py"

# Initialize conda in the script environment
eval "$(~/miniconda3/bin/conda shell.bash hook)"

# Activate environment
echo "Activating conda environment: $ENV_NAME"
conda activate $ENV_NAME

# Check if activation was successful
if [ $? -eq 0 ]; then
    echo "Environment activated successfully"
    
    # Run your script
    echo "Running script: $SCRIPT_PATH"
    python $SCRIPT_PATH
    
    # Check if script ran successfully
    if [ $? -eq 0 ]; then
        echo "Script completed successfully"
    else
        echo "Script failed with exit code $?"
    fi
else
    echo "Failed to activate environment"
fi

exit