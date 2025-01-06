import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from collections import Counter
from datetime import datetime
from skimage.transform import resize
from sklearn.impute import SimpleImputer
from PIL import Image

# Function to load and resize RGB PNG image
def load_rgb_png(file_path, target_shape=(128, 128)):
    try:
        # Open the image using PIL
        img = Image.open(file_path)
        
        # Convert to RGB if not already
        img = img.convert("RGB")
        
        # Convert image to numpy array
        img = np.array(img)
        print(f"Original image shape: {img.shape}")
        
        # Resize image to target shape without preserving aspect ratio
        img_resized = resize(img, target_shape, mode='reflect', anti_aliasing=True)
        
        print(f"Resized image shape: {img_resized.shape}")
        return img_resized
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

# Function to classify susceptibility into 25 buckets (0-4, 5-9, ..., 95-99)
def classify_susc(susc_value):
    return susc_value // 5  # This will categorize the values into buckets: 0-4 -> 0, 5-9 -> 1, ..., 95-99 -> 19

# Function to check for missing values
def check_missing_values(data):
    if np.any(np.isnan(data)):
        print("Missing values detected in the dataset!")
    else:
        print("No missing values found in the dataset.")

# Add the imputer to handle missing values in the data
def apply_imputer(data, strategy='mean'):
    # Reshape the data from (n_samples, height, width, channels) to (n_samples, height * width * channels)
    n_samples, height, width, channels = data.shape
    data_reshaped = data.reshape(n_samples, -1)  # Flatten the height, width, and channels into a single dimension
    
    # Apply the imputer
    imputer = SimpleImputer(strategy=strategy)
    data_imputed_reshaped = imputer.fit_transform(data_reshaped)
    
    # Reshape the data back to its original dimensions
    data_imputed = data_imputed_reshaped.reshape(n_samples, height, width, channels)
    return data_imputed

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
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
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
        aspect_path = os.path.join(root_dir, subdir, "sat.png")  # Changed to png for RGB images
        susc_path = os.path.join(root_dir, subdir, "susc.txt")

        if os.path.exists(aspect_path) and os.path.exists(susc_path):
            img = load_rgb_png(aspect_path, target_shape=(128, 128))
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

    # Check for missing values in images
    check_missing_values(images)

    # Apply imputer if necessary
    if np.any(np.isnan(images)):
        print("Imputing missing values in image data...")
        images = apply_imputer(images, strategy='mean')  # You can choose other strategies like 'median' or 'most_frequent'

    # Check for missing values again after imputation
    check_missing_values(images)

    # Normalize images across all pixels
    images = images.astype(np.float32)
    images = (images - images.min()) / (images.max() - images.min())  # Normalize to [0, 1]
    print(f"Images normalized. Example pixel values: {images[0, 0, 0, :]}")

    # Shuffle images and labels together to randomize order
    images, labels = shuffle(images, labels, random_state=42)

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
    model = cnn_model(input_shape=(128, 128, 3))  # 3-channel (RGB) input shape

    # TensorBoard log directory
    log_dir = os.path.join("AI System/Topography/logs/", "AI_NORM_RGB-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Add callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)

    # Print the magic command for TensorBoard in Colab
    print(f"To start TensorBoard, run the following command in a new cell:")
    print(f"%tensorboard --logdir {log_dir}")

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=250,
        batch_size=100,  # Reduced batch size for better generalization
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, lr_scheduler, tensorboard_callback]
    )

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.2f}, Test Loss: {test_loss:.2f}")

if __name__ == "__main__":
    main()
