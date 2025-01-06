import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from collections import Counter
from datetime import datetime
from sklearn.impute import SimpleImputer

# Function to load rain data from a txt file
def load_rain_data(file_path):
    try:
        with open(file_path, 'r') as f:
            rain_value = float(f.read().strip())  # Read the float value from the file
        return rain_value
    except ValueError:
        print(f"Invalid value in {file_path}")
        return None

# Function to classify susceptibility into 25 buckets (0-4, 5-9, ..., 95-99)
def classify_susc(susc_value):
    return int(susc_value // 5)  # This will categorize the values into buckets: 0-4 -> 0, 5-9 -> 1, ..., 95-99 -> 19

# Optimized model with dense layers for handling the float input
def dense_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.6)(x)  # Add dropout for regularization
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
    root_dir = "Data/Datasets"
    subdirs = [str(i) for i in range(1, 501)]  # Assuming 500 datasets

    rain_data = []
    labels = []

    for subdir in subdirs:
        rain_path = os.path.join(root_dir, subdir, "rain.txt")
        susc_path = os.path.join(root_dir, subdir, "susc.txt")

        if os.path.exists(rain_path) and os.path.exists(susc_path):
            rain_value = load_rain_data(rain_path)
            if rain_value is not None:
                try:
                    with open(susc_path, 'r') as f:
                        susc = float(f.read().strip())
                    # Classify susceptibility into 25 buckets
                    label = classify_susc(susc)
                    rain_data.append([rain_value])  # Keep rain value in a 2D list
                    labels.append(label)
                except ValueError:
                    print(f"Invalid value in {susc_path}")

    # Validation: Check if rain_data and labels are loaded correctly
    if len(rain_data) == 0 or len(labels) == 0:
        print("No valid data found. Please check the file paths and data format.")
        return

    # Convert lists to NumPy arrays
    rain_data = np.array(rain_data)
    labels = np.array(labels)

    print(f"Loaded {len(rain_data)} rain data with shape {rain_data.shape}")
    print(f"Loaded {len(labels)} labels with distribution: {Counter(labels)}")

    # Check for missing values in rain_data
    if np.any(np.isnan(rain_data)):
        print("Missing values detected in the rain data!")

    # Shuffle data
    rain_data, labels = shuffle(rain_data, labels, random_state=42)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(rain_data, labels, test_size=0.2, random_state=42, shuffle=True)

    print(f"Train label distribution: {Counter(y_train)}")
    print(f"Test label distribution: {Counter(y_test)}")

    # Build and train Dense model
    model = dense_model(input_shape=(1,))  # Input is a single float value

    # TensorBoard log directory
    log_dir = os.path.join("AI System/Rain/logs/", "AI_RAIN-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Add callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)

    # Print the magic command for TensorBoard in Colab
    print(f"To start TensorBoard, run the following command in a new cell: %tensorboard --logdir {log_dir}")

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=100,  # Reduced batch size for better generalization
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, lr_scheduler, tensorboard_callback]
    )

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.2f}, Test Loss: {test_loss:.2f}")

if __name__ == "__main__":
    main()
