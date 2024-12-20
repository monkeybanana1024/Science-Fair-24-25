#!/bin/bash

# Check if the current directory is named "Data"
if [ "$(basename "$PWD")" = "Data" ]; then
    echo "Current directory is 'Data'. Proceeding with execution."
    
    echo "Normalizing Coordinates"
    python "$PWD/Processing/fixCoordinates.py"

    # Your commands go here
    echo "Collecting Topography"
    python "$PWD/Collection/GetTopography.py"

    echo "Collecting Rain Data"
    python "$PWD/Collection/GetRain.py"
    
    echo "Collecting Susceptibility Data"
    python "$PWD/Collection/GetSusceptibility.py"
    
    echo "Collecting NDVI Data"
    python "$PWD/Collection/getNDVI.py"

    echo "Collecting Imagery Data"
    python "$PWD/Collection/GetSat.py"

    echo "Data collection completed."

    # Ask if user wants to proceed to processing
    while true; do
        read -p "Do you want to proceed to processing? (y/n) " yn
        case $yn in
            [Yy]* ) echo "Proceeding to processing..."; break;;
            [Nn]* ) echo "Exiting..."; exit;;
            * ) echo "Please answer yes or no.";;
        esac
    done

    echo "Proccessing Slope Data"
    python "$PWD/Processing/calculateSlope.py"

    echo "Proccessing Normalization Data"
    python "$PWD/Processing/calculateNormalized.py"

    echo "Processing Completed."
    echo "All tasks completed successfully."
else
    echo "Error: Current directory is not 'Data'."
    echo "Current directory is: $PWD"
    exit 1
fi
