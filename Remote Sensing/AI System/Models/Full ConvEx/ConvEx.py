import os
import numpy as np
import rasterio
import tensorflow as tf
import xgboost as xgb
from rasterio.enums import Resampling
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from collections import Counter
from datetime import datetime
from skimage.transform import resize
from sklearn.impute import SimpleImputer
import tensorflow.compat.v2 as tfv2
from tensorboard import program

# Function to load datasets
def load_single_band_geotiff(file_path, target_shape=(128, 128)):
    try:
        with rasterio.open(file_path) as src:
            # Read only the first band (single-channel)
            img = src.read(1)  # Read the band into a 2D array
            print(f"Original image shape: {img.shape}")
            
            # Resize image to target shape without preserving aspect ratio
            img_resized = resize(img, target_shape, mode='reflect', anti_aliasing=True)
            
            # Expand dimensions to include channel dimension (height, width, 1)
            img_resized = np.expand_dims(img_resized, axis=-1)
            
            print(f"Resized image shape: {img_resized.shape}")
            return img_resized
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None