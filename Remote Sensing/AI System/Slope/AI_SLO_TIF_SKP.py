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

def cnn_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # First Conv Block
    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    sk1 = tf.keras.layers.Conv2D(16, (1, 1), padding='same')(x)

    # Second Conv Block
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    sk2 = tf.keras.layers.Conv2D(32, (1, 1), padding='same')(x)

    # Third Conv Block
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    sk3 = tf.keras.layers.Conv2D(64, (1, 1), strides=(2, 2), padding='same')(sk2)

    # Fourth Conv Block
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    sk4 = tf.keras.layers.Conv2D(64, (1, 1), padding='same')(x)

    # Fifth Conv Block
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    sk5 = tf.keras.layers.Conv2D(128, (1, 1), padding='same')(x)

    # Further Conv Block + Skip Connections
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    # Adjust skip connections
    sk1 = tf.keras.layers.Conv2D(256, (1, 1), strides=(4, 4), padding='same')(sk1)
    sk2 = tf.keras.layers.Conv2D(256, (1, 1), strides=(4, 4), padding='same')(sk2)
    sk3 = tf.keras.layers.Conv2D(256, (1, 1), strides=(2, 2), padding='same')(sk3)
    sk4 = tf.keras.layers.Conv2D(256, (1, 1), strides=(2, 2), padding='same')(sk4)
    sk5 = tf.keras.layers.Conv2D(256, (1, 1), strides=(2, 2), padding='same')(sk5)

    # Add skip connections
    x = tf.keras.layers.Add()([x, sk1, sk2, sk3, sk4, sk5])

    # Global Average Pooling
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Dense layers
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.7)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)

    # Output layer
    output = tf.keras.layers.Dense(4, activation='softmax')(x)

    model = tf.keras.Model(inputs, output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    tensorboard_process = None
    root_dir = "Data/Datasets"
    subdirs = [str(i) for i in range(1, 501)]
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

    if len(images) == 0 or len(labels) == 0:
        print("No valid data found. Please check the file paths and data format.")
        return

    images = np.array(images)
    labels = np.array(labels)
    print(f"Loaded {len(images)} images with shape {images.shape}")
    print(f"Loaded {len(labels)} labels with distribution: {Counter(labels)}")

    images, labels = shuffle(images, labels, random_state=42)

    scaler = MinMaxScaler()
    for i in range(images.shape[-1]):
        band = images[:, :, :, i].reshape(-1, 1)
        band = scaler.fit_transform(band)
        images[:, :, :, i] = band.reshape(images.shape[0], images.shape[1], images.shape[2])
    print(f"Images normalized. Example pixel values: {images[0, 0, 0, :]}")

    augmented_images = []
    augmented_labels = []
    for img, label in zip(images, labels):
        augmented_images.extend([
            img,
            np.flipud(img),
            np.fliplr(img),
            np.rot90(img, k=1),
            np.rot90(img, k=2),
            np.rot90(img, k=3)
        ])
        augmented_labels.extend([label] * 6)

    images = np.array(augmented_images)
    labels = np.array(augmented_labels)
    images, labels = shuffle(images, labels, random_state=42)
    print(f"After augmentation: {len(images)} images and {len(labels)} labels")

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42, shuffle=True)
    print(f"Train label distribution: {Counter(y_train)}")
    print(f"Test label distribution: {Counter(y_test)}")

    model = cnn_model(input_shape=(128, 128, images.shape[-1]))

    log_dir = os.path.join("AI System/Slope/logs/", "AI_SLO_TIF_SKP-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    try:
        print(f"Launching TensorBoard at log directory: {log_dir}")
        tensorboard_process = subprocess.Popen(
            ["tensorboard", "--logdir", log_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=25,
            restore_best_weights=True
        )
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)

        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=64,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, lr_scheduler, tensorboard_callback]
        )

        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {test_accuracy:.2f}, Test Loss: {test_loss:.2f}")

    finally:
        if tensorboard_process:
            tensorboard_process.terminate()
            print("TensorBoard process terminated.")

if __name__ == "__main__":
    main()
