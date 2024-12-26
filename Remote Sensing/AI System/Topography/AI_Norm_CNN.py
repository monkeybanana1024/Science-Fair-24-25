import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import csv
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # type: ignore

def read_dataset_file(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f]

def load_png_data(directory):
    filepath = os.path.join(directory, 'normalized_topography.png')
    # Load the image in RGB mode
    image = tf.keras.preprocessing.image.load_img(filepath, target_size=(128, 128))  # Resize as needed
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    return image_array / 255.0  # Normalize pixel values

def read_susc_file(directory):
    with open(os.path.join(directory, 'susc.txt'), 'r') as f:
        return float(f.read().strip())

def process_datasets(root_dir, dataset_files):
    X, y, dirs = [], [], []
    for filename in dataset_files:
        subdirs = read_dataset_file(filename)
        for subdir in subdirs:
            data = load_png_data(os.path.join(root_dir, subdir))
            actual_susc = read_susc_file(os.path.join(root_dir, subdir))
            X.append(data)
            y.append(actual_susc)
            dirs.append(subdir)
    return np.array(X), np.array(y), dirs

def save_results(directory, actual, predicted, mae, mse, rmse):
    filepath = os.path.join(directory, 'analytics.csv')
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['AI_Norm_CNN', mae, mse, rmse, ((predicted - actual) / ((predicted + actual) / 2)), actual, predicted])

# Main execution
root_directory = 'Data/Datasets/'
train_dirs = read_dataset_file('AI System/train.txt')
test_dirs = read_dataset_file('AI System/test.txt')
valid_dirs = read_dataset_file('AI System/valid.txt')

X_train, y_train, _ = process_datasets(root_directory, ['AI System/train.txt'])
X_test, y_test, test_subdirs = process_datasets(root_directory, ['AI System/test.txt'])
X_valid, y_valid, valid_subdirs = process_datasets(root_directory, ['AI System/valid.txt'])

# Reshape data for CNN (samples, height, width, channels)
X_train = X_train.reshape(-1, 128, 128, 3)  # Adjust dimensions based on your image size and color channels
X_test = X_test.reshape(-1, 128, 128, 3)
X_valid = X_valid.reshape(-1, 128, 128, 3)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),  # Input shape adjusted for RGB
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1)  # Output layer for regression
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train CNN model
model.fit(X_train, y_train, epochs=50, batch_size=16)

# Make predictions
test_predictions = model.predict(X_test).flatten()
valid_predictions = model.predict(X_valid).flatten()

# Save results for test set
for i, subdir in enumerate(test_subdirs):
    mae = mean_absolute_error([y_test[i]], [test_predictions[i]])
    mse = mean_squared_error([y_test[i]], [test_predictions[i]])
    rmse = np.sqrt(mse)
    save_results(os.path.join(root_directory, subdir), y_test[i], test_predictions[i], mae, mse, rmse)

# Save results for validation set
for i, subdir in enumerate(valid_dirs):
    mae = mean_absolute_error([y_valid[i]], [valid_predictions[i]])
    mse = mean_squared_error([y_valid[i]], [valid_predictions[i]])
    rmse = np.sqrt(mse)
    save_results(os.path.join(root_directory, subdir), y_valid[i], valid_predictions[i], mae, mse, rmse)

print("Results have been saved in analytics.csv files in each subdirectory.")
