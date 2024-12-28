import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression  # Meta-model for stacking
from scipy.stats import skew, kurtosis
from skopt import gp_minimize
from skopt.space import Real, Integer
import rasterio
import random

# Function to load GeoTIFF as a 2D array
def load_geotiff(file_path):
    with rasterio.open(file_path) as src:
        return src.read(1)

# Set the path to your root directory containing subdirectories
root_dir = "Data/Datasets"

# Get all subdirectories and shuffle them
all_subdirs = [str(i) for i in range(1, 501)]
random.shuffle(all_subdirs)

# Select 400 for training and 100 for testing
train_subdirs = all_subdirs[:400]
test_subdirs = all_subdirs[400:]

# Function to load data from specified subdirectories
def load_data(subdirs):
    X, y = [], []
    for subdir in subdirs:
        tiff_path = os.path.join(root_dir, subdir, "normalized_topography.tif")
        susc_path = os.path.join(root_dir, subdir, "susc.txt")
        if os.path.exists(tiff_path) and os.path.exists(susc_path):
            X.append(load_geotiff(tiff_path))
            y.append(float(open(susc_path).read().strip()))
        else:
            print(f"Missing files in {subdir}: TIFF: {tiff_path}, Susc: {susc_path}")
    print(f"Loaded {len(X)} samples from {len(subdirs)} directories.")
    return X, np.array(y)

# Load training and testing data
X_train, y_train = load_data(train_subdirs)
X_test, y_test = load_data(test_subdirs)

# Normalize susceptibility values to be between 0 and 1 for model training
y_scaler = MinMaxScaler(feature_range=(0, 1))
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

# Function to extract advanced features from 2D arrays
def extract_advanced_features(X):
    features = []
    for x in X:
        feat = [
            np.mean(x), np.std(x), np.min(x), np.max(x), np.median(x),
            np.percentile(x, 25), np.percentile(x, 75), np.var(x),
            np.sum(x > np.mean(x)), np.sum(x < np.mean(x)),
            np.ptp(x), np.mean(np.gradient(x)), np.std(np.gradient(x)),
            skew(x.flatten()), kurtosis(x.flatten()),
            np.sum(np.abs(np.diff(x)))
        ]
        features.append(feat)
    return np.array(features)

# Preprocess the data with advanced features
X_train_features = extract_advanced_features(X_train)
X_test_features = extract_advanced_features(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_features)
X_test_scaled = scaler.transform(X_test_features)

# Handle NaN values using SimpleImputer before fitting models
imputer = SimpleImputer(strategy='mean')
X_train_scaled = imputer.fit_transform(X_train_scaled)
X_test_scaled = imputer.transform(X_test_scaled)

# Define the objective function for XGBoost optimization
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
        objective='reg:squarederror',
        n_jobs=-1,
        early_stopping_rounds=10
        )
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_index, val_index in kf.split(X_train_scaled):
        X_train_fold, X_val_fold = X_train_scaled[train_index], X_train_scaled[val_index]
        y_train_fold, y_val_fold = y_train_scaled[train_index], y_train_scaled[val_index]
        
        model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=False  # Early stopping to prevent overfitting
        )
        
        preds = model.predict(X_val_fold)
        scores.append(mean_absolute_error(y_val_fold, preds))
    
    return np.mean(scores)

# Define the objective function for LightGBM optimization
def lgb_objective(params):
    n_estimators, max_depth, learning_rate = params
    
    model = lgb.LGBMRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=learning_rate,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_index, val_index in kf.split(X_train_scaled):
        X_train_fold, X_val_fold = X_train_scaled[train_index], X_train_scaled[val_index]
        y_train_fold, y_val_fold = y_train_scaled[train_index], y_train_scaled[val_index]
        
        model.fit(X_train_fold, y_train_fold)
        
        preds = model.predict(X_val_fold)
        
        scores.append(mean_absolute_error(y_val_fold, preds))
    
    return np.mean(scores)


# Perform Bayesian optimization for each model.
xgb_result = gp_minimize(
    xgb_objective,
    [
       Integer(3 ,50),              # max_depth 
       Real(0.001 ,0.5),             # learning_rate 
       Integer(100 ,50000),          # n_estimators 
       Real(1e-5 ,10),              # min_child_weight 
       Real(0.5 ,1.0),              # subsample 
       Real(0.05 ,1.0),              # colsample_bytree 
       Real(0 ,20)                  # gamma 
   ],
   n_calls=100,
   random_state=42,
   verbose=True,
)

lgb_result = gp_minimize(
   lgb_objective,
   [
       Integer(100 ,50000),          # n_estimators 
       Integer(3 ,50),             # max_depth 
       Real(0.001 ,0.5)            # learning_rate 
   ],
   n_calls=100,
   random_state=42,
   verbose=True,
)

# Get best parameters from optimization results.
best_xgb_params = {
   'max_depth': int(xgb_result.x[0]),
   'learning_rate': xgb_result.x[1],
   'n_estimators': int(xgb_result.x[2]),
   'min_child_weight': xgb_result.x[3],
   'subsample': xgb_result.x[4],
   'colsample_bytree': xgb_result.x[5],
   'gamma': xgb_result.x[6]
}

best_lgb_params = {
   'n_estimators': int(lgb_result.x[0]),
   'max_depth': int(lgb_result.x[1]),
   'learning_rate': lgb_result.x[2]
}

# Train the best models.
best_xgb_model = XGBRegressor(**best_xgb_params)
best_xgb_model.fit(X_train_scaled,y_train_scaled)

best_lgb_model = lgb.LGBMRegressor(**best_lgb_params)
best_lgb_model.fit(X_train_scaled,y_train_scaled)

# Make predictions.
xbg_pred=y_scaler.inverse_transform(best_xgb_model.predict(X_test_scaled).reshape(-1 ,1)).ravel()
lbg_pred=y_scaler.inverse_transform(best_lgb_model.predict(X_test_scaled).reshape(-1 ,1)).ravel()

# Calculate metrics.
xbg_mae=mean_absolute_error(y_test,xbg_pred)  
xbg_mse=mean_squared_error(y_test,xbg_pred)  
lbg_mae=mean_absolute_error(y_test ,lbg_pred)  
lbg_mse=mean_squared_error(y_test ,lbg_pred)  

print("\nXGBoost Results:")
print(f"Mean Absolute Error: {xbg_mae:.4f}")
print(f"Mean Squared Error: {xbg_mse:.4f}")

print("\nLightGBM Results:")
print(f"Mean Absolute Error: {lbg_mae:.4f}")
print(f"Mean Squared Error: {lbg_mse:.4f}")

# Ensemble prediction (weighted average).
ensemble_pred=(xbg_pred + (lbg_pred * 1.5)) / 2.5  # Giving more weight to LightGBM
ensemble_mae=mean_absolute_error(y_test ,ensemble_pred) 
ensemble_mse=mean_squared_error(y_test ,ensemble_pred) 

print("\nEnsemble Results:")
print(f"Mean Absolute Error: {ensemble_mae:.4f}")
print(f"Mean Squared Error: {ensemble_mse:.4f}")

# Implementing a meta-model using Linear Regression on predictions of both models.
meta_X_train = np.column_stack((best_xgb_model.predict(X_train_scaled), best_lgb_model.predict(X_train_scaled)))
meta_X_test = np.column_stack((best_xgb_model.predict(X_test_scaled), best_lgb_model.predict(X_test_scaled)))

meta_model = LinearRegression()
meta_model.fit(meta_X_train, y_train_scaled)  # Train on scaled target

meta_preds = meta_model.predict(meta_X_test)

# Inverse transform the meta predictions back to original scale.
meta_preds_inverse = y_scaler.inverse_transform(meta_preds.reshape(-1, 1)).ravel()

# Calculate metrics for meta-model.
meta_mae = mean_absolute_error(y_test, meta_preds_inverse)
meta_mse = mean_squared_error(y_test, meta_preds_inverse)

print("\nMeta-Model Results:")
print(f"Mean Absolute Error: {meta_mae:.4f}")
print(f"Mean Squared Error: {meta_mse:.4f}")

# Print best parameters.
print("\nBest XGBoost Parameters:")
print(best_xgb_params)

print("\nBest LightGBM Parameters:")
print(best_lgb_params)
