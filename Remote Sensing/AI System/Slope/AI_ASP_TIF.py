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

# Function to load and resize single-band GeoTIFF
def load_singleband_geotiff(file_path, target_shape=(128, 128)):
    try:
        with rasterio.open(file_path) as src:
            img = src.read(1, out_shape=(target_shape[0], target_shape[1]), resampling=Resampling.bilinear)  # Read the first (and only) band
            return img
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

# Optimized CNN model for single-band input
def cnn_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, (7, 7), activation='relu', padding='same', strides=(2, 2))(inputs)
    x = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='valid', strides=(2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (5, 5), activation='relu', padding='valid', strides=(2, 2))(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='valid', strides=(2, 2))(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='valid', strides=(2, 2))(x)

    y = tf.keras.layers.Conv2D(512, (7, 7), activation='softmax', padding='same', strides=(2, 2))(inputs)
    y = tf.keras.layers.Conv2D(256, (5, 5), activation='softmax', padding='same', strides=(2, 2))(y)
    y = tf.keras.layers.Conv2D(128, (5, 5), activation='softmax', padding='same', strides=(2, 2))(y)
    y = tf.keras.layers.Conv2D(64, (3, 3), activation='softmax', padding='same', strides=(2, 2))(y)
    y = tf.keras.layers.Conv2D(32, (3, 3), activation='softmax', padding='same', strides=(2, 2))(y)
    
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(y)
    y = tf.keras.layers.Flatten()(y)  # Pool spatial dimensions into one feature per channel
    
    x = tf.keras.layers.Concatenate()([x, y])

    # Fully Connected Layers
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.00)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)  
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(4, activation='softmax')(x)  # 4 classes for slope categories

    # Compile the model
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
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
        aspect_path = os.path.join(root_dir, subdir, "Slope/aspect.tif")
        susc_path = os.path.join(root_dir, subdir, "susc.txt")

        if os.path.exists(aspect_path) and os.path.exists(susc_path):
            img = load_singleband_geotiff(aspect_path, target_shape=(128, 128))
            if img is not None:
                try:
                    with open(susc_path, 'r') as f:
                        susc = float(f.read().strip())
                    # Classify into 4 buckets based on the value ranges
                    if 0 <= susc <= 25:
                        label = 0
                    elif 26 <= susc <= 50:
                        label = 1
                    elif 51 <= susc <= 75:
                        label = 2
                    elif 76 <= susc <= 100:
                        label = 3
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

    # Add channel dimension to images for single-band data
    images = np.expand_dims(images, axis=-1)  # Shape: (samples, height, width, 1)

    print(f"Loaded {len(images)} images with shape {images.shape}")
    print(f"Loaded {len(labels)} labels with distribution: {Counter(labels)}")

    # Shuffle images and labels together to randomize order
    images, labels = shuffle(images, labels, random_state=42)

    # Normalize images
    scaler = MinMaxScaler()
    images = images.reshape(-1, 1)
    images = scaler.fit_transform(images)
    images = images.reshape(-1, 128, 128, 1)

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
    model = cnn_model(input_shape=(128, 128, 1))

    # TensorBoard log directory
    log_dir = os.path.join("AI System/Slope/logs/", "AI_ASPECT_TIF-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
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
            batch_size=8,  # Reduced batch size for better generalization
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
