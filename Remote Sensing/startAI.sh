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
                source "$PWD/Venv/xgboost/Scripts/activate" # Windows
                echo "Venv activated"
                echo "Running XGBoost"
                echo "Topography"
                python "$PWD/AI System/Topography/AI_Norm_XG.py"
                echo "Program Completed"
                deactivate
                break;;
            2 ) 
                echo "Starting venv"
                source "$PWD/Venv/xgboost/Scripts/activate" # Windows
                echo "Venv activated"
                echo "Running XGBoost"
                echo "Aspect"
                python "$PWD/AI System/Slope/AI_ASP_XG.py"
                echo "Program Completed"
                deactivate
                break;;
            3 ) 
                echo "Installing Dependencies"
                source "$PWD/Venv/xgboost/Scripts/activate" # Windows
                pip install affine==2.4.0 \
                          alembic==1.14.0 \
                          attrs==24.3.0 \
                          certifi==2024.12.14 \
                          click==8.1.8 \
                          click-plugins==1.1.1 \
                          cligj==0.7.2 \
                          colorama==0.4.6 \
                          colorlog==6.9.0 \
                          contourpy==1.3.1 \
                          cycler==0.12.1 \
                          fonttools==4.55.3 \
                          greenlet==3.1.1 \
                          joblib==1.4.2 \
                          kiwisolver==1.4.8 \
                          lightgbm==4.5.0 \
                          Mako==1.3.8 \
                          markdown-it-py==3.0.0 \
                          MarkupSafe==3.0.2 \
                          matplotlib==3.10.0 \
                          mdurl==0.1.2 \
                          numpy==2.2.1 \
                          optuna==4.1.0 \
                          packaging==24.2 \
                          pandas==2.2.3 \
                          pillow==11.0.0 \
                          pyaml==24.12.1 \
                          Pygments==2.18.0 \
                          pyparsing==3.2.0 \
                          python-dateutil==2.9.0.post0 \
                          pytz==2024.2 \
                          PyYAML==6.0.2 \
                          rasterio==1.4.3 \
                          rich==13.9.4 \
                          scikit-learn==1.5.2 \
                          scikit-optimize==0.10.2 \
                          scipy==1.14.1 \
                          six==1.17.0 \
                          SQLAlchemy==2.0.36 \
                          threadpoolctl==3.5.0 \
                          tqdm==4.67.1 \
                          typing_extensions==4.12.2 \
                          tzdata==2024.2 \
                          xgboost==2.1.3
                echo "Dependencies installed"
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
