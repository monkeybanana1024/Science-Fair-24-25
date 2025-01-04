import os
import numpy as np
import rasterio
import tensorflow as tf
from rasterio.enums import Resampling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from collections import Counter
from datetime import datetime
import subprocess

# Function to load and resize multi-band GeoTIFF
def load_multiband_geotiff(file_path, target_shape=(128, 128)):
    try:
        with rasterio.open(file_path) as src:
            img = src.read(
                out_shape=(src.count, target_shape[0], target_shape[1]),
                resampling=Resampling.bilinear
            )
            img = np.transpose(img, (1, 2, 0))  # Shape (height, width, bands)
            return img
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

# Function to classify susceptibility into 25 buckets (0-4, 5-9, ..., 95-99)
def classify_susc(susc_value):
    return susc_value // 5  # This will categorize the values into buckets: 0-4 -> 0, 5-9 -> 1, ..., 95-99 -> 19

# Optimized CNN model with increased complexity
def cnn_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='valid')(inputs)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='valid')(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='valid')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='valid')(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)  # Replace Flatten with Global Average Pooling
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.6)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    output = tf.keras.layers.Dense(25, activation='softmax')(x)  # 25 output classes for 25 buckets

    model = tf.keras.Model(inputs, output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Main function
def main():
    tensorboard_process = None
    root_dir = "Data/Datasets"
    subdirs = [str(i) for i in range(1, 501)]  # Assuming 500 datasets

    images = []
    labels = []

    for subdir in subdirs:
        tiff_path = os.path.join(root_dir, subdir, "slope/slope.tif")
        susc_path = os.path.join(root_dir, subdir, "susc.txt")

        if os.path.exists(tiff_path) and os.path.exists(susc_path):
            img = load_multiband_geotiff(tiff_path, target_shape=(128, 128))
            if img is not None:
                try:
                    with open(susc_path, 'r') as f:
                        susc = float(f.read().strip())
                    # Classify susceptibility into 25 buckets
                    label = classify_susc(susc)
                    images.append(img)
                    labels.append(label)
                except ValueError:
                    print(f"Invalid value in {susc_path}")

    # Validation: Check if images and labels are loaded correctly
    if len(images) == 0 or len(labels) == 0:
        print("No valid data found. Please check the file paths and data format.")
        return

    # Convert lists to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)

    print(f"Loaded {len(images)} images with shape {images.shape}")
    print(f"Loaded {len(labels)} labels with distribution: {Counter(labels)}")

    # Shuffle images and labels together to randomize order
    images, labels = shuffle(images, labels, random_state=42)

    # Normalize images
    scaler = MinMaxScaler()
    for i in range(images.shape[-1]):  # Normalize each band independently
        band = images[:, :, :, i].reshape(-1, 1)
        band = scaler.fit_transform(band)
        images[:, :, :, i] = band.reshape(images.shape[0], images.shape[1], images.shape[2])

    print(f"Images normalized. Example pixel values: {images[0, 0, 0, :]}")

    # Data Augmentation: Flip X, Flip Y, Rotate 90°, 180°, and 270°
    augmented_images = []
    augmented_labels = []

    for img, label in zip(images, labels):
        augmented_images.extend([
            img,  # Original
            np.flipud(img),  # Flip vertically
            np.fliplr(img),  # Flip horizontally
            np.rot90(img, k=1),  # Rotate 90°
            np.rot90(img, k=2),  # Rotate 180°
            np.rot90(img, k=3)  # Rotate 270°
        ])
        augmented_labels.extend([label] * 6)  # Duplicate labels for augmentations

    # Concatenate original and augmented datasets
    images = np.array(augmented_images)
    labels = np.array(augmented_labels)

    # Shuffle the combined dataset
    images, labels = shuffle(images, labels, random_state=42)

    print(f"After augmentation: {len(images)} images and {len(labels)} labels")

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42, shuffle=True)

    print(f"Train label distribution: {Counter(y_train)}")
    print(f"Test label distribution: {Counter(y_test)}")

    # Build and train CNN model
    model = cnn_model(input_shape=(128, 128, images.shape[-1]))

    # TensorBoard log directory
    log_dir = os.path.join("AI System/Slope/logs/", "AI_SLO_TIF-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Launch TensorBoard in a subprocess
    try:
        print(f"Launching TensorBoard at log directory: {log_dir}")
        tensorboard_process = subprocess.Popen(
            ["tensorboard", "--logdir", log_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Add callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)

        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=100,  # Reduced batch size for better generalization
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, lr_scheduler, tensorboard_callback]
        )

        # Evaluate the model
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {test_accuracy:.2f}, Test Loss: {test_loss:.2f}")

    finally:
        # Ensure TensorBoard process is terminated when done
        if tensorboard_process:
            tensorboard_process.terminate()
            print("TensorBoard process terminated.")

if __name__ == "__main__":
    main()
