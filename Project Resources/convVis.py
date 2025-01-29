import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import random
import os

# Simple convolution filter
def simple_convolution(input_image):
    # Define a simple 3x3 filter (convolution kernel)
    kernel = np.array([[0, -5, 0],
                       [-5, 20, -5],
                       [0, -5, 0]], dtype=np.float32)

    # Add batch and channel dimensions
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
    input_image = np.expand_dims(input_image, axis=-1)  # Add channel dimension

    # Convert input to float32 for compatibility with tf.nn.conv2d
    input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)
    
    # Define kernel (convolution filter)
    kernel = np.expand_dims(kernel, axis=-1)  # Add channel dimension to kernel
    kernel = np.expand_dims(kernel, axis=-1)  # Add filter dimension to kernel
    kernel_tensor = tf.convert_to_tensor(kernel, dtype=tf.float32)
    
    # Perform the convolution using tf.nn.conv2d with VALID padding
    conv_result = tf.nn.conv2d(input_image, kernel_tensor, strides=[1, 1, 1, 1], padding="VALID")
    
    # Apply activation (ReLU)
    activated_image = tf.nn.relu(conv_result)

    # Remove batch and channel dimensions from intermediate outputs
    conv_result = conv_result.numpy().squeeze()
    activated_image = activated_image.numpy().squeeze()

    return conv_result, activated_image

# Load the normalized topography image
def load_image(file_path):
    try:
        img = Image.open(file_path).convert('L')  # Convert to grayscale (L)
        return np.array(img)  # Convert image to numpy array
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

# Visualize the original image, convolution result, and activation result
def visualize_convolution(input_image):
    plt.figure(figsize=(12, 12))

    # Plot the original image
    plt.subplot(1, 3, 1)
    plt.imshow(input_image, cmap='magma')
    plt.title("Original Image")
    plt.axis('off')

    # Apply convolution and get the result
    conv_result, activated_image = simple_convolution(input_image)
    
    # Plot the convolution result
    plt.subplot(1, 3, 2)
    plt.imshow(conv_result, cmap='magma')
    plt.title("After Convolution")
    plt.axis('off')

    # Plot the activation result (ReLU)
    plt.subplot(1, 3, 3)
    plt.imshow(activated_image, cmap='magma')
    plt.title("After Activation (ReLU)")
    plt.axis('off')

    plt.show()

# Main function to load and process topography data
def main():
    root_dir = "Data/Datasets"
    subdirs = [str(i) for i in range(1, 11)]  # Use a smaller set of data for testing
    random.shuffle(subdirs)

    for subdir in subdirs:
        tiff_path = os.path.join(root_dir, subdir, "normalized_topography.png")
        if os.path.exists(tiff_path):
            img = load_image(tiff_path)
            if img is not None:
                visualize_convolution(img)  # Visualize the convolution, activation, and pooling for each image

if __name__ == "__main__":
    main()
