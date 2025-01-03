import os
import numpy as np
import rasterio
import tensorflow as tf
import math
from rasterio.enums import Resampling
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# Load and resize the normalized topography
def load_geotiff(file_path, target_shape=(256, 256)):
    try:
        with rasterio.open(file_path) as src:
            img = src.read(1, out_shape=(1, target_shape[0], target_shape[1]), resampling=Resampling.bilinear).squeeze()
            return img
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

# Main function to load and process topography data
def main():
    root_dir = "Data/Datasets"
    subdirs = [str(i) for i in range(1, 61)]  # Assume 60 datasets for simplicity

    images = []
    outputs = []

    for subdir in subdirs:
        tiff_path = os.path.join(root_dir, subdir, "normalized_topography.tif")
        susc_path = os.path.join(root_dir, subdir, "susc.txt")

        if os.path.exists(tiff_path) and os.path.exists(susc_path):
            img = load_geotiff(tiff_path)
            if img is not None:
                try:
                    susc = float(open(susc_path).read().strip())
                    images.append(img)
                    outputs.append(susc)
                except ValueError:
                    print(f"Invalid value in {susc_path}")

    # Convert lists to NumPy arrays
    images = np.array(images)
    outputs = np.array(outputs)

    print(f"Loaded {len(images)} images and {len(outputs)} output values")

    # Apply imputer to handle missing values
    imputer = SimpleImputer(strategy='mean')  # Use 'mean' strategy to replace missing values

    # Reshape images to 2D for the imputer
    images = images.reshape(images.shape[0], -1)  # Flatten each image to a 1D array

    # Apply imputer to images
    images = imputer.fit_transform(images)

    # Apply MinMaxScaler to normalize the images to the range [0, 1]
    scaler_images = MinMaxScaler()
    images = scaler_images.fit_transform(images)

    # Reshape back to original shape
    images = images.reshape(-1, 256, 256, 1)  # Add the channel dimension here (1 channel for grayscale)

    # Apply imputer to outputs (in case there are missing labels)
    outputs = imputer.fit_transform(outputs.reshape(-1, 1)).reshape(-1)

    # Apply MinMaxScaler to normalize the outputs to the range [0, 1]
    scaler_outputs = MinMaxScaler()
    outputs = scaler_outputs.fit_transform(outputs.reshape(-1, 1)).reshape(-1)

    # Specify training and testing splits
    train_indices = slice(0, 50)  # First 50 for training
    test_indices = slice(50, 60)  # Next 10 for testing

    X_train, X_test = images[train_indices], images[test_indices]
    y_train, y_test = outputs[train_indices], outputs[test_indices]

    # Build a model with Conv2D layers
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(256, 256, 1)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])


    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=75, validation_data=(X_test, y_test))

    # Evaluate the model
    test_loss, test_mae, test_mse = model.evaluate(X_test, y_test)
    print(f"Test MAE: {test_mae*100}, Test RMSE: {math.sqrt(test_mse)*100}")

if __name__ == "__main__":
    main()
