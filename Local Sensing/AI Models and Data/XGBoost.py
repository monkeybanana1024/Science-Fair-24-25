import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import shap
import os
import matplotlib.pyplot as plt
import joblib  # for saving the scaler

# Function to load data from a directory
def load_data(directory):
    data = []
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(directory, file))
            data.append(df)
    return pd.concat(data, ignore_index=True)

# Function to export models
def export_models(models, scaler, export_dir='exported_models'):
    os.makedirs(export_dir, exist_ok=True)
    for event, model in models.items():
        model_path = os.path.join(export_dir, f'{event}_model.json')
        model.save_model(model_path)
    
    scaler_path = os.path.join(export_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    
    print(f"Models and scaler exported to {export_dir}")

# Function to import models
def import_models(import_dir='exported_models'):
    models = {}
    for file in os.listdir(import_dir):
        if file.endswith('_model.json'):
            event = file.split('_')[0]
            model_path = os.path.join(import_dir, file)
            model = XGBClassifier()
            model.load_model(model_path)
            models[event] = model
    
    scaler_path = os.path.join(import_dir, 'scaler.joblib')
    scaler = joblib.load(scaler_path)
    
    print(f"Models and scaler imported from {import_dir}")
    return models, scaler

# Load data
train_data = load_data('Data/Train')
valid_data = load_data('Data/Valid')
test_data = load_data('Data/Test')

# Preprocess the data
for df in [train_data, valid_data, test_data]:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Create features and targets
features = ['Water Flow Rate', 'Moisture', 'Movement X', 'Movement Y', 'Movement Z', 'Incline']
target_events = ['Landslide', 'Debris', 'Slump']

X_train = train_data[features]
X_valid = valid_data[features]
X_test = test_data[features]

y_train = train_data[target_events]
y_valid = valid_data[target_events]
y_test = test_data[target_events]

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# Train models for each event type
models = {}
for event in target_events:
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train_scaled, y_train[event], 
              eval_set=[(X_valid_scaled, y_valid[event])],
              early_stopping_rounds=10,
              verbose=False)
    models[event] = model

# Function to make predictions and save results with modified filenames
def predict_and_save(X, y, data, scaled_data, folder_name, suffix):
    results_df = data.copy()
    
    for event in target_events:
        y_pred_proba = models[event].predict_proba(scaled_data)[:, 1]
        results_df[f'{event}_Probability'] = y_pred_proba
        results_df[f'Next_100ms_{event}_Probability'] = results_df[f'{event}_Probability'].rolling(window=2, min_periods=1).max()

    # Ensure the directory exists
    os.makedirs(folder_name, exist_ok=True)

    # Save the results with appended suffix
    results_file_name = os.path.join(folder_name, f'results_with_predictions_{suffix}.csv')
    results_df.to_csv(results_file_name, index=False)
    
    return results_df

# Make predictions and save results for validation and test sets with suffixes
os.chdir('XGBoost')
valid_results = predict_and_save(X_valid, y_valid, valid_data, X_valid_scaled, 'ValidResults', 'Valid')
test_results = predict_and_save(X_test, y_test, test_data, X_test_scaled, 'TestResults', 'Test')

# Feature importance analysis and saving plots with modified filenames
for event in target_events:
    explainer = shap.TreeExplainer(models[event])
    shap_values = explainer.shap_values(X_test_scaled)
    
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    
    # Save plot with appended suffix
    plt.title(f'Feature Importance for {event} Prediction')
    plt.tight_layout()
    
    plot_file_name = f'TestResults/{event}_feature_importance_Test.png'
    plt.savefig(plot_file_name)
    
    plt.close()

print("Analysis complete. Results saved in 'ValidResults' and 'TestResults' folders.")
print("Feature importance plots saved in 'TestResults' folder.")

# Export the trained models and scaler
export_models(models, scaler)

# To use the imported models later, you can uncomment the following lines:
imported_models, imported_scaler = import_models()
# Then use imported_models and imported_scaler instead of models and scaler