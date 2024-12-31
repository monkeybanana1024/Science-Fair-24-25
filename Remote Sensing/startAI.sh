#!/bin/bash

# Check if the current directory is named "Remote Sensing"
if [ "$(basename "$PWD")" = "Remote Sensing" ]; then
    echo "Current directory is 'Remote Sensing'. Proceeding with execution."

    while true; do
        echo "Select an option:"
        echo "1. Run Topography"
        echo "2. Run Aspect"
        echo "3. Run NDVI"
        echo "4. Run Rain"
        read -p "Enter your choice (1-4): " choice
        case $choice in
            1 ) 
                echo "Starting venv"
                source "$PWD/Venv/xgboost/Scripts/activate"
                echo "Venv activated"
                echo "Running XGBoost"
                echo "Topography"
                python "$PWD/AI System/Topography/AI_Norm_XG.py"
                echo "Program Completed"
                deactivate
                break;;
            2 ) 
                echo "Starting venv"
                source "$PWD/Venv/xgboost/Scripts/activate"
                echo "Venv activated"
                echo "Running XGBoost"
                echo "Aspect"
                python "$PWD/AI System/Slope/AI_ASP_XG.py"
                echo "Program Completed"
                deactivate
                break;;
            3 ) 
                echo "Option 3 selected (placeholder)"
                break;;
            4 ) 
                echo "Option 4 selected (placeholder)"
                break;;
            * ) 
                echo "Invalid option. Please enter a number between 1 and 4.";;
        esac
    done
else
    echo "Error: Current directory is not 'Remote Sensing'."
    echo "Current directory is: $PWD"
    exit 1
fi
