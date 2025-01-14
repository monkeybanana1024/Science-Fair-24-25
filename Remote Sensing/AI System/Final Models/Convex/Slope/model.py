import os
import numpy as np
import tensorflow as tf
from collections import Counter
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer
from skimage.transform import resize
from PIL import Image
import pandas as pd
import rasterio

# Function to load and resize 8-band GeoTIFF image
def load_geotiff(file_path, target_shape=(128, 128)):
    try:
        # Read the GeoTIFF file
        img = rasterio.open(file_path).read()  # Read all bands (assuming 8 bands)
        img_resized = resize(img, (img.shape[0], target_shape[0], target_shape[1]), mode='reflect', anti_aliasing=True)
        return img_resized.transpose(1, 2, 0)  # Reorder to (height, width, bands)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

# Function to check for missing values
def check_missing_values(data):
    if np.any(np.isnan(data)):
        print("Missing values detected in the dataset!")
    else:
        print("No missing values found in the dataset.")

# Add the imputer to handle missing values in the data
def apply_imputer(data, strategy='mean'):
    n_samples, height, width, channels = data.shape
    data_reshaped = data.reshape(n_samples, -1)
    imputer = SimpleImputer(strategy=strategy)
    data_imputed_reshaped = imputer.fit_transform(data_reshaped)
    data_imputed = data_imputed_reshaped.reshape(n_samples, height, width, channels)
    return data_imputed

# Function to make predictions and save results
def make_predictions_and_save(model, X_test, output_file="predictions.csv"):
    predictions = model.predict(X_test)
    predicted_buckets = np.argmax(predictions, axis=1)
    confidence_scores = np.max(predictions, axis=1)

    # Combine bucket and confidence into the desired format
    combined_predictions = [f"{bucket}.{int(confidence * 1000):03d}" for bucket, confidence in zip(predicted_buckets, confidence_scores)]

    # Create a DataFrame with the predictions
    results = pd.DataFrame({
        "Predicted": combined_predictions
    })

    # Save to CSV
    results.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

# Modify the main function to load geotiff files correctly
def main():
    # Path to the pre-trained model
    model_path = "AI System/Results/Slope/model.keras"

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    # Load the pre-trained model
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded successfully from {model_path}")

    root_dir = "convex/Datasets"  # The root directory where subfolders with image files are stored
    subdirs = [str(i) for i in range(1, 201)]  # Assuming 47 datasets

    images = []

    for subdir in subdirs:
        aspect_path = os.path.join(root_dir, subdir, "slope/slope.tif")

        if os.path.exists(aspect_path):
            img = load_geotiff(aspect_path, target_shape=(128, 128))
            if img is not None:
                images.append(img)

    if len(images) == 0:
        print("No valid data found. Please check the file paths and data format.")
        return

    images = np.array(images)

    print(f"Loaded {len(images)} images with shape {images.shape}")

    check_missing_values(images)

    if np.any(np.isnan(images)):
        print("Imputing missing values in image data...")
        images = apply_imputer(images, strategy='mean')

    check_missing_values(images)

    images = images.astype(np.float32)
    images = (images - images.min()) / (images.max() - images.min())
    print(f"Images normalized. Example pixel values: {images[0, 0, 0, :]}")

    # Split into testing set (use entire dataset as test data for demonstration)
    X_test = images

    print(f"Making predictions...")

    # Make predictions and save to CSV
    make_predictions_and_save(model, X_test, output_file="AI System/Final Models/Convex/Slope/results.csv")

if __name__ == "__main__":
    main()
