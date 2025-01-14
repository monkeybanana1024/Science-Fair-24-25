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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Function to load and resize RGB PNG image
def load_rgb_png(file_path, target_shape=(128, 128)):
    try:
        img = Image.open(file_path)
        img = img.convert("RGB")
        img = np.array(img)
        img_resized = resize(img, target_shape, mode='reflect', anti_aliasing=True)
        return img_resized
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

# Function to convert ground truth to bucketed form
def bucketize_ground_truth(susc_value):
    # Define the bucket boundaries
    return int(susc_value // 5)

# Function to make predictions and save results
def make_predictions_and_save(model, X_test, ground_truth, output_file="predictions.csv"):
    predictions = model.predict(X_test)
    predicted_buckets = np.argmax(predictions, axis=1)
    confidence_scores = np.max(predictions, axis=1)

    # Combine bucket and confidence into the desired format
    combined_predictions = [f"{bucket}.{int(confidence * 1000):03d}" for bucket, confidence in zip(predicted_buckets, confidence_scores)]
    
    # Convert ground truth values to their bucketed form
    bucketed_ground_truth = [bucketize_ground_truth(value) for value in ground_truth]

    # Create a DataFrame with the predictions and ground truth
    results = pd.DataFrame({
        "Predicted": combined_predictions,
        "Ground Truth": bucketed_ground_truth
    })

    # Save to CSV
    results.to_csv(output_file, index=False)
    print(f"Predictions and ground truth saved to {output_file}")
    
    # Return predicted labels and ground truth for further evaluation
    return predicted_buckets, bucketed_ground_truth

# Function to calculate and print metrics
def print_metrics(y_true, y_pred):
    print("\nEvaluation Metrics:")
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Weighted): {precision:.4f}")
    print(f"Recall (Weighted): {recall:.4f}")
    print(f"F1 Score (Weighted): {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

# Main function
def main():
    # Path to the pre-trained model
    model_path = "AI System/Results/Topography/model.keras"

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    # Load the pre-trained model
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded successfully from {model_path}")
    root_dir = "data/Datasets"  # The root directory where subfolders with image files are stored
    subdirs = [str(i) for i in range(1, 501)]  # Assuming 47 datasets

    images = []
    ground_truth = []

    for subdir in subdirs:
        aspect_path = os.path.join(root_dir, subdir, "normalized_topography.png")
        susc_path = os.path.join(root_dir, subdir, "susc.txt")

        if os.path.exists(aspect_path) and os.path.exists(susc_path):
            # Load image
            img = load_rgb_png(aspect_path, target_shape=(128, 128))
            if img is not None:
                images.append(img)

            # Load ground truth from susc.txt
            try:
                with open(susc_path, 'r') as f:
                    susc_value = float(f.readline().strip())
                    ground_truth.append(susc_value)
            except Exception as e:
                print(f"Error reading file {susc_path}: {e}")

    if len(images) == 0 or len(ground_truth) == 0:
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

    # Make predictions and save to CSV, including ground truth
    predicted_buckets, bucketed_ground_truth = make_predictions_and_save(model, X_test, ground_truth, output_file="AI System/Final Models/Convex/Topography/results.csv")

    # Print metrics
    print_metrics(bucketed_ground_truth, predicted_buckets)

if __name__ == "__main__":
    main()
