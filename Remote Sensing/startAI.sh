#!/bin/bash

# Check if the current directory is named "Remote Sensing"
if [ "$(basename "$PWD")" = "Remote Sensing" ]; then
    echo "Current directory is 'Remote Sensing'. Proceeding with execution."

    #echo "Generating Folds"
    #python "$PWD/AI System/generateDatasets.py"

    while true; do
        read -p "Do you want to proceed to XGBoost? (y/n) " yn
        case $yn in
            [Yy]* ) echo "Proceeding to processing..."; break;;
            [Nn]* ) echo "Exiting..."; exit;;
            * ) echo "Please answer yes or no.";;
        esac
    done

    echo 'Starting venv'
    
    # Activate the virtual environment
    source "$PWD/Venv/xgboost/Scripts/activate"  # Use 'source' to activate

    echo 'Venv activated'

    echo "Running XGBoost"
    echo 'Topography'
    
    # Run your XGBoost Python script
    python "$PWD/AI System/Topography/AI_Norm_XG.py"

    echo 'Program Completed'
    
    # Deactivate the virtual environment
    deactivate

else
    echo "Error: Current directory is not 'Remote Sensing'."
    echo "Current directory is: $PWD"
    exit 1
fi
