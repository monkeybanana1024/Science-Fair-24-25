import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from skimage.transform import resize
from sklearn.impute import SimpleImputer
import tensorflow as tf
import cv2

def load_single_band_geotiff(file_path, target_shape=(128, 128)):
    try:
        with rasterio.open(file_path) as src:
            img = src.read(1)
            img_resized = resize(img, target_shape, mode='reflect', anti_aliasing=True)
            img_resized = np.expand_dims(img_resized, axis=-1)
            return img_resized
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def load_multiband_geotiff(file_path, target_shape=(128, 128)):
    try:
        with rasterio.open(file_path) as src:
            img = src.read(
                out_shape=(src.count, target_shape[0], target_shape[1]),
                resampling=Resampling.bilinear
            )
            img = np.transpose(img, (1, 2, 0))
            return img
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def load_png_image(file_path, target_shape=(128, 128)):
    try:
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"File not found or unable to read: {file_path}")
        img_resized = cv2.resize(img, target_shape, interpolation=cv2.INTER_AREA)
        img_resized = img_resized.astype(np.float32) / 255.0
        return img_resized
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def classify_susc(prediction):
    # Assuming prediction is a class index, return the corresponding susceptibility range
    return prediction * 5  # For example, class 0 is 0-4%, class 1 is 5-9%, etc.

def calculate_confidence(predictions):
    return np.max(predictions)  # Assuming predictions are probabilities, return the max confidence

def apply_imputer(data, strategy='mean'):
    n_samples, height, width, channels = data.shape
    data_reshaped = data.reshape(n_samples, -1)
    imputer = SimpleImputer(strategy=strategy)
    data_imputed_reshaped = imputer.fit_transform(data_reshaped)
    data_imputed = data_imputed_reshaped.reshape(n_samples, height, width, channels)
    return data_imputed

def load_model_predictions(model, image):
    # Ensure the image has the batch dimension
    image = np.expand_dims(image, axis=0)  # Adding batch dimension
    predictions = model.predict(image)
    return predictions

def save_predictions_to_csv(predictions_dict, output_csv_path):
    df = pd.DataFrame(predictions_dict)
    df.to_csv(output_csv_path, index=False)

def process_images(input_image_dir, models_slope, models_satellite, models_topography, models_ndvi, output_csv_path):
    predictions_slope = []
    predictions_satellite = []
    predictions_topography = []
    predictions_ndvi = []

    for filename in os.listdir(input_image_dir):
        if filename.endswith('.tif') or filename.endswith('.png'):
            file_path = os.path.join(input_image_dir, filename)
            if "slope" in filename:
                image = load_multiband_geotiff(file_path)
                if image is not None:
                    predictions = load_model_predictions(models_slope, image)
                    predictions_slope.append(predictions)
            elif "satellite" in filename:
                image = load_png_image(file_path, target_shape=(512, 512))
                if image is not None:
                    predictions = load_model_predictions(models_satellite, image)
                    predictions_satellite.append(predictions)
            elif "topography" in filename:
                image = load_png_image(file_path, target_shape=(128, 128))
                if image is not None:
                    predictions = load_model_predictions(models_topography, image)
                    predictions_topography.append(predictions)
            elif "ndvi" in filename or "aspect" in filename:
                image = load_single_band_geotiff(file_path)
                if image is not None:
                    predictions = load_model_predictions(models_ndvi, image)
                    predictions_ndvi.append(predictions)

    # Process the predictions and create the CSV
    predictions_dict = {
        'Slope Predictions': [f"{classify_susc(np.argmax(pred))}.{calculate_confidence(pred)}" for pred in predictions_slope],
        'Satellite Predictions': [f"{classify_susc(np.argmax(pred))}.{calculate_confidence(pred)}" for pred in predictions_satellite],
        'Topography Predictions': [f"{classify_susc(np.argmax(pred))}.{calculate_confidence(pred)}" for pred in predictions_topography],
        'NDVI Predictions': [f"{classify_susc(np.argmax(pred))}.{calculate_confidence(pred)}" for pred in predictions_ndvi]
    }

    save_predictions_to_csv(predictions_dict, output_csv_path)

    # Ensemble predictions (average of all models)
    ensemble_predictions = {
        'Ensemble Predictions': []
    }

    num_images = len(predictions_slope)
    for i in range(num_images):
        slope_pred = predictions_slope[i]
        satellite_pred = predictions_satellite[i]
        topography_pred = predictions_topography[i]
        ndvi_pred = predictions_ndvi[i]

        # Average the model predictions for the ensemble prediction
        ensemble_pred = np.mean([np.max(slope_pred), np.max(satellite_pred), np.max(topography_pred), np.max(ndvi_pred)])
        ensemble_predictions['Ensemble Predictions'].append(f"{classify_susc(ensemble_pred)}.{ensemble_pred}")

    save_predictions_to_csv(ensemble_predictions, 'AI System/Final Models/ensemble_predictions.csv')

def main():
    input_image_dir = 'Implementation/Datasets'
    output_csv_path = 'AI System/Final Models/predictions.csv'

    # Load models
    models_slope = tf.keras.models.load_model("AI System/Results/Slope/model.keras")
    models_satellite = tf.keras.models.load_model("AI System/Results/Satellite/model.keras")
    models_topography = tf.keras.models.load_model("AI System/Results/Topography/model.keras")
    models_ndvi = tf.keras.models.load_model("AI System/Results/NDVI/model.keras")

    # Process images and make predictions
    process_images(input_image_dir, models_slope, models_satellite, models_topography, models_ndvi, output_csv_path)

if __name__ == "__main__":
    main()
