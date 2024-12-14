import os
import rasterio
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import csv

def read_dataset_file(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f]

def load_tif_data(directory):
    filepath = os.path.join(directory, 'normalized_topography.tif')  # Changed to normalized_topography.tif
    with rasterio.open(filepath) as src:
        return src.read(1).flatten(), src.transform, src.crs

def read_susc_file(directory):
    with open(os.path.join(directory, 'susc.txt'), 'r') as f:
        return float(f.read().strip())

def process_datasets(root_dir, dataset_files):
    X, y, dirs = [], [], []
    for filename in dataset_files:
        subdirs = read_dataset_file(filename)
        for subdir in subdirs:
            data, _, _ = load_tif_data(os.path.join(root_dir, subdir))
            actual_susc = read_susc_file(os.path.join(root_dir, subdir))
            X.append(data)
            y.append(actual_susc)
            dirs.append(subdir)
    return np.array(X), np.array(y), dirs

def save_results(directory, actual, predicted, mae, mse, rmse):
    filepath = os.path.join(directory, 'analytics.csv')
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['AI_Norm_XG', mae, mse, rmse, ((predicted-actual)/((predicted+actual)/2)), actual, predicted])

# Main execution
root_directory = 'Data/Datasets/'
train_dirs = read_dataset_file('AI System/train.txt')
test_dirs = read_dataset_file('AI System/test.txt')
valid_dirs = read_dataset_file('AI System/valid.txt')

X_train, y_train, _ = process_datasets(root_directory, ['AI System/train.txt'])
X_test, y_test, test_subdirs = process_datasets(root_directory, ['AI System/test.txt'])
X_valid, y_valid, valid_subdirs = process_datasets(root_directory, ['AI System/valid.txt'])

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

# Set XGBoost parameters
params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

# Train XGBoost model
num_rounds = 100
model = xgb.train(params, dtrain, num_rounds, evals=[(dvalid, 'validation')],
                  early_stopping_rounds=10, verbose_eval=10)

# Make predictions
test_predictions = model.predict(dtest)
valid_predictions = model.predict(dvalid)

# Save results for test set
for i, subdir in enumerate(test_subdirs):
    mae = mean_absolute_error([y_test[i]], [test_predictions[i]])
    mse = mean_squared_error([y_test[i]], [test_predictions[i]])
    rmse = np.sqrt(mse)
    save_results(os.path.join(root_directory, subdir), y_test[i], test_predictions[i], mae, mse, rmse)

# Save results for validation set
for i, subdir in enumerate(valid_subdirs):
    mae = mean_absolute_error([y_valid[i]], [valid_predictions[i]])
    mse = mean_squared_error([y_valid[i]], [valid_predictions[i]])
    rmse = np.sqrt(mse)
    save_results(os.path.join(root_directory, subdir), y_valid[i], valid_predictions[i], mae, mse, rmse)

print("Results have been saved in analytics.csv files in each subdirectory.")
