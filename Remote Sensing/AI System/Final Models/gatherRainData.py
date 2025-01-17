import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Function to load rain values from a file in each subdirectory
def load_rain_values_from_subdirs(root_dir):
    rain_values = []
    subdirs = [str(i) for i in range(1,501)]  # Assuming subdirs are labeled 1, 2, ..., 47

    for subdir in subdirs:
        rain_file_path = os.path.join(root_dir, subdir, "rain.txt")
        
        if os.path.exists(rain_file_path):
            try:
                with open(rain_file_path, 'r') as f:
                    rain_value = float(f.readline().strip())
                    print(rain_value)  # Read the single rain value
                    rain_values.append(rain_value)
            except Exception as e:
                print(f"Error reading {rain_file_path}: {e}")

    return rain_values

# Main function to load rain values, augment, shuffle, split, and print test dataset values
def main():
    # Set random seed for reproducibility
    random_seed = 42
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    # Root directory where subdirectories are stored
    root_dir = "Data/Datasets"  # Change this to the correct path

    # Load the rain values from each subdirectory
    rain_values = load_rain_values_from_subdirs(root_dir)

    if not rain_values:
        print("No rain values found or there was an error loading the files.")
        return

    # Convert to TensorFlow tensor for augmentation and ensure consistent behavior
    rain_tensor = tf.constant(rain_values, dtype=tf.float32)

    # Augment the rain values to match 600 datasets (6 times)
    augmented_rain_values = tf.tile(rain_tensor, [6])  # Tile the tensor 6 times

    # Convert back to numpy array for shuffling and splitting
    augmented_rain_values_np = augmented_rain_values.numpy()

    # Shuffle the augmented rainfall values with the same random state
    shuffled_rain_values = np.random.RandomState(random_seed).permutation(augmented_rain_values_np)

    # Split the data into training and testing (80% train, 20% test for example)
    X_train, X_test = train_test_split(shuffled_rain_values, test_size=0.2, random_state=random_seed)

    # Print the rainfall values in the test dataset
    print("Rainfall Values in Test Dataset:")
    for value in X_test:
        print(value)

if __name__ == "__main__":
    main()
