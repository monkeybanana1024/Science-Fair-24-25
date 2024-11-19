import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import shap
import os
import matplotlib.pyplot as plt

# Function to load data from a directory
def load_data(directory):
    data = []
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(directory, file))
            data.append(df)
    return pd.concat(data, ignore_index=True)

# Load data
train_data = load_data('Data/train')
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

# Function to make predictions and save results
def predict_and_save(X, y, data, scaled_data, folder_name):
    results_df = data.copy()
    
    for event in target_events:
        y_pred_proba = models[event].predict_proba(scaled_data)[:, 1]
        results_df[f'{event}_Probability'] = y_pred_proba
        results_df[f'Next_100ms_{event}_Probability'] = results_df[f'{event}_Probability'].rolling(window=2, min_periods=1).max()

    # Ensure the directory exists
    os.makedirs(folder_name, exist_ok=True)
    
    # Save the results
    results_df.to_csv(os.path.join(folder_name, 'results_with_predictions.csv'), index=False)
    
    return results_df

# Make predictions and save results for validation and test sets
valid_results = predict_and_save(X_valid, y_valid, valid_data, X_valid_scaled, 'ValidResults')
test_results = predict_and_save(X_test, y_test, test_data, X_test_scaled, 'TestResults')

# Feature importance analysis
for event in target_events:
    explainer = shap.TreeExplainer(models[event])
    shap_values = explainer.shap_values(X_test_scaled)
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title(f'Feature Importance for {event} Prediction')
    plt.tight_layout()
    plt.savefig(f'TestResults/{event}_feature_importance.png')
    plt.close()

print("Analysis complete. Results saved in 'ValidResults' and 'TestResults' folders.")
print("Feature importance plots saved in 'TestResults' folder.")