import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
import rasterio
import random
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from scipy.stats import skew, kurtosis

# Function to load GeoTIFF as a 2D array
def load_geotiff(file_path):
    with rasterio.open(file_path) as src:
        return src.read(1)  # Read the first band, preserving 2D structure

# Set the path to your root directory containing subdirectories
root_dir = "Data/Datasets"  # Adjust this path

# Get all subdirectories
all_subdirs = [str(i) for i in range(1, 501)]  # Assuming 500 subdirectories
random.shuffle(all_subdirs)  # Shuffle the list of subdirectories

# Select 400 for training and 100 for testing
train_subdirs = all_subdirs[:400]
test_subdirs = all_subdirs[400:500]

# Function to load data from specified subdirectories
def load_data(subdirs):
    X, y = [], []
    for subdir in subdirs:
        tiff_path = os.path.join(root_dir, subdir, "normalized_topography.tif")
        susc_path = os.path.join(root_dir, subdir, "susc.txt")
        
        if os.path.exists(tiff_path) and os.path.exists(susc_path):
            X.append(load_geotiff(tiff_path))  # Keep values in range [0, 255]
            y.append(float(open(susc_path).read().strip()))  # Susceptibility values [0, 100]
        else:
            print(f"Missing files in {subdir}: TIFF: {tiff_path}, Susc: {susc_path}")
    
    print(f"Loaded {len(X)} samples from {len(subdirs)} directories.")
    return X, np.array(y)

# Load training and testing data
X_train, y_train = load_data(train_subdirs)
X_test, y_test = load_data(test_subdirs)

# Check if y_test is empty and handle it appropriately
if len(y_test) == 0:
    print("Error: No samples loaded for testing. Please check your data paths.")
    exit(1)

# Normalize susceptibility values to be between 0 and 1 for model training
y_scaler = MinMaxScaler(feature_range=(0, 1))
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

# Function to extract advanced features from 2D arrays
def extract_advanced_features(X):
    features = []
    for x in X:
        feat = [
            np.mean(x),
            np.std(x),
            np.min(x),
            np.max(x),
            np.median(x),
            np.percentile(x, 25),
            np.percentile(x, 75),
            np.var(x),
            np.sum(x > np.mean(x)),
            np.sum(x < np.mean(x)),
            np.ptp(x),  # Peak-to-peak (max - min)
            np.mean(np.gradient(x)),  # Mean of gradient
            np.std(np.gradient(x)),   # Standard deviation of gradient
            np.percentile(x, 10),
            np.percentile(x, 90),
            skew(x.flatten()),  # Skewness
            kurtosis(x.flatten()),  # Kurtosis
            np.sum(np.abs(np.diff(x))),  # Total variation
        ]
        features.append(feat)
    return np.array(features)

# Preprocess the data with advanced features
X_train_features = extract_advanced_features(X_train)
X_test_features = extract_advanced_features(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_features)
X_test_scaled = scaler.transform(X_test_features)

# Define the XGBoost objective function for Bayesian optimization
def xgb_objective(params):
    max_depth, learning_rate, n_estimators, min_child_weight, subsample, colsample_bytree, gamma = params
    model = XGBRegressor(
        max_depth=int(max_depth),
        learning_rate=learning_rate,
        n_estimators=int(n_estimators),
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        objective='reg:squarederror'
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_index, val_index in kf.split(X_train_scaled):
        X_train_fold, X_val_fold = X_train_scaled[train_index], X_train_scaled[val_index]
        y_train_fold, y_val_fold = y_train_scaled[train_index], y_train_scaled[val_index]
        model.fit(X_train_fold, y_train_fold)
        predictions = model.predict(X_val_fold)
        scores.append(mean_squared_error(y_val_fold, predictions))
    return np.mean(scores)

# Define the Random Forest objective function for Bayesian optimization
def rf_objective(params):
    n_estimators, max_depth, min_samples_split, min_samples_leaf = params
    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        random_state=42
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_index, val_index in kf.split(X_train_scaled):
        X_train_fold, X_val_fold = X_train_scaled[train_index], X_train_scaled[val_index]
        y_train_fold, y_val_fold = y_train_scaled[train_index], y_train_scaled[val_index]
        model.fit(X_train_fold, y_train_fold)
        predictions = model.predict(X_val_fold)
        scores.append(mean_squared_error(y_val_fold, predictions))
    return np.mean(scores)

# Define the search space for XGBoost
xgb_space = [
    Integer(3, 10, name='max_depth'),
    Real(0.01, 0.3, name='learning_rate'),
    Integer(100, 1000, name='n_estimators'),
    Real(1, 10, name='min_child_weight'),
    Real(0.5, 1.0, name='subsample'),
    Real(0.5, 1.0, name='colsample_bytree'),
    Real(0, 5, name='gamma')
]

# Perform Bayesian optimization for XGBoost
xgb_result = gp_minimize(xgb_objective, xgb_space, n_calls=50, random_state=42, verbose=True, n_jobs=-1)

# Get the best XGBoost parameters
best_xgb_params = {
    'max_depth': int(xgb_result.x[0]),
    'learning_rate': xgb_result.x[1],
    'n_estimators': int(xgb_result.x[2]),
    'min_child_weight': xgb_result.x[3],
    'subsample': xgb_result.x[4],
    'colsample_bytree': xgb_result.x[5],
    'gamma': xgb_result.x[6]
}

# Define the search space for Random Forest
rf_space = [
    Integer(100, 500, name='n_estimators'),
    Integer(5, 30, name='max_depth'),
    Integer(2, 20, name='min_samples_split'),
    Integer(1, 10, name='min_samples_leaf')
]

# Perform Bayesian optimization for Random Forest
rf_result = gp_minimize(rf_objective, rf_space, n_calls=50, random_state=42, verbose=True, n_jobs=-1)

# Get the best Random Forest parameters
best_rf_params = {
    'n_estimators': int(rf_result.x[0]),
    'max_depth': int(rf_result.x[1]),
    'min_samples_split': int(rf_result.x[2]),
    'min_samples_leaf': int(rf_result.x[3])
}

# Train the best models
best_xgb = XGBRegressor(**best_xgb_params, objective='reg:squarederror')
best_xgb.fit(X_train_scaled, y_train_scaled)

best_rf = RandomForestRegressor(**best_rf_params, random_state=42)
best_rf.fit(X_train_scaled, y_train_scaled)

# Make predictions
xgb_pred = y_scaler.inverse_transform(best_xgb.predict(X_test_scaled).reshape(-1, 1)).ravel()
rf_pred = y_scaler.inverse_transform(best_rf.predict(X_test_scaled).reshape(-1, 1)).ravel()

# Calculate metrics
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_mse = mean_squared_error(y_test, xgb_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)

print("\nXGBoost Results:")
print(f"Mean Absolute Error: {xgb_mae:.4f}")
print(f"Mean Squared Error: {xgb_mse:.4f}")

print("\nRandom Forest Results:")
print(f"Mean Absolute Error: {rf_mae:.4f}")
print(f"Mean Squared Error: {rf_mse:.4f}")

# Ensemble prediction (simple average)
ensemble_pred = (xgb_pred + rf_pred) / 2
ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
ensemble_mse = mean_squared_error(y_test, ensemble_pred)

print("\nEnsemble Results:")
print(f"Mean Absolute Error: {ensemble_mae:.4f}")
print(f"Mean Squared Error: {ensemble_mse:.4f}")

# Print best parameters
print("\nBest XGBoost Parameters:")
print(best_xgb_params)
print("\nBest Random Forest Parameters:")
print(best_rf_params)
