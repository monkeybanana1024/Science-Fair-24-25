import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from scipy.stats import skew, kurtosis
import optuna
import rasterio
import random
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

def load_geotiff(file_path):
    with rasterio.open(file_path) as src:
        return src.read(1)

root_dir = "Data/Datasets"
all_subdirs = [str(i) for i in range(1, 501)]
random.shuffle(all_subdirs)
train_subdirs = all_subdirs[:400]
test_subdirs = all_subdirs[400:]

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

X_train, y_train = load_data(train_subdirs)
X_test, y_test = load_data(test_subdirs)

y_scaler = MinMaxScaler(feature_range=(0, 1))
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

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

X_train_features = extract_advanced_features(X_train)
X_test_features = extract_advanced_features(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_features)
X_test_scaled = scaler.transform(X_test_features)

imputer = SimpleImputer(strategy='mean')
X_train_scaled = imputer.fit_transform(X_train_scaled)
X_test_scaled = imputer.transform(X_test_scaled)

def select_features(X, y, threshold=0.01):
    model = XGBRegressor(n_estimators=100)
    model.fit(X, y)
    importance = model.feature_importances_
    selected_indices = importance > threshold
    return X[:, selected_indices], selected_indices

X_train_selected, selected_indices = select_features(X_train_scaled, y_train_scaled)
X_test_selected = X_test_scaled[:, selected_indices]

def xgb_objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 25),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_float('min_child_weight', 1, 10, log=True),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
        'early_stopping_rounds': 20
    }
    
    model = XGBRegressor(**params, objective='reg:squarederror', n_jobs=-1)
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_index, val_index in kf.split(X_train_selected, pd.cut(y_train_scaled, bins=5, labels=False)):
        X_train_fold, X_val_fold = X_train_selected[train_index], X_train_selected[val_index]
        y_train_fold, y_val_fold = y_train_scaled[train_index], y_train_scaled[val_index]
        
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=False
        )
        
        preds = model.predict(X_val_fold)
        scores.append(mean_absolute_error(y_val_fold, preds))
    
    return np.mean(scores)

def lgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 25),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 3000),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'early_stopping_rounds':20, 
        'verbosity': -1
    }
    
    model = lgb.LGBMRegressor(**params, n_jobs=-1)
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_index, val_index in kf.split(X_train_selected, pd.cut(y_train_scaled, bins=5, labels=False)):
        X_train_fold, X_val_fold = X_train_selected[train_index], X_train_selected[val_index]
        y_train_fold, y_val_fold = y_train_scaled[train_index], y_train_scaled[val_index]
        
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
        )
        
        preds = model.predict(X_val_fold)
        scores.append(mean_absolute_error(y_val_fold, preds))
    
    return np.mean(scores)

study_xgb = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
study_xgb.optimize(xgb_objective, n_trials=100, n_jobs=-1)

study_lgb = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
study_lgb.optimize(lgb_objective, n_trials=100, n_jobs=-1)

best_xgb = XGBRegressor(**study_xgb.best_params, objective='reg:squarederror', n_jobs=-1)
best_xgb.fit(X_train_selected, y_train_scaled)

best_lgb = lgb.LGBMRegressor(**study_lgb.best_params, n_jobs=-1)
best_lgb.fit(X_train_selected, y_train_scaled)

xgb_pred = y_scaler.inverse_transform(best_xgb.predict(X_test_selected).reshape(-1, 1)).ravel()
lgb_pred = y_scaler.inverse_transform(best_lgb.predict(X_test_selected).reshape(-1, 1)).ravel()

meta_X_train = np.column_stack((
    best_xgb.predict(X_train_selected),
    best_lgb.predict(X_train_selected),
    X_train_selected
))
meta_X_test = np.column_stack((
    best_xgb.predict(X_test_selected),
    best_lgb.predict(X_test_selected),
    X_test_selected
))

meta_model = XGBRegressor(n_estimators=100, max_depth=3)
meta_model.fit(meta_X_train, y_train_scaled)

meta_pred = y_scaler.inverse_transform(meta_model.predict(meta_X_test).reshape(-1, 1)).ravel()

for name, pred in [('XGBoost', xgb_pred), ('LightGBM', lgb_pred), ('Stacked', meta_pred)]:
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"\n{name} Results:")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared: {r2:.4f}")

print("\nBest XGBoost Parameters:")
print(study_xgb.best_params)

print("\nBest LightGBM Parameters:")
print(study_lgb.best_params)
