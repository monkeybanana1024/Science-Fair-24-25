import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import random
import os

# Simple convolution filter
def simple_convolution(input_image):
    # Define a simple 3x3 filter (convolution kernel)
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float32)

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

    # Apply max pooling (2x2) until 1x1
    pooled_image = activated_image
    while pooled_image.shape[1] > 1 and pooled_image.shape[2] > 1:
        pooled_image = tf.nn.max_pool2d(pooled_image, ksize=2, strides=2, padding="VALID")
    
    # Get the final single value
    single_value = pooled_image.numpy().squeeze()

    # Remove batch and channel dimensions from intermediate outputs
    conv_result = conv_result.numpy().squeeze()
    activated_image = activated_image.numpy().squeeze()
    pooled_image = pooled_image.numpy().squeeze()

    return conv_result, activated_image, pooled_image, single_value

# Load the normalized topography image
def load_image(file_path):
    try:
        img = Image.open(file_path).convert('L')  # Convert to grayscale (L)
        return np.array(img)  # Convert image to numpy array
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

# Visualize the image, convolution result, activation, and pooling result
def visualize_convolution(input_image):
    plt.figure(figsize=(15, 15))

    # Plot the original image
    plt.subplot(3, 2, 1)
    plt.imshow(input_image)
    plt.title("Original Image")
    plt.axis('off')

    # Apply convolution and plot the result
    conv_result, activated_image, pooled_image, single_value = simple_convolution(input_image)
    
    # Plot the convolution result
    plt.subplot(3, 2, 2)
    plt.imshow(conv_result)
    plt.title("After Convolution")
    plt.axis('off')

    # Plot the activation result (ReLU)
    plt.subplot(3, 2, 3)
    plt.imshow(activated_image)
    plt.title("After Activation (ReLU)")
    plt.axis('off')

    # Plot the pooled result
    # Ensure pooled_image has the correct shape for imshow
    if pooled_image.ndim == 0:  # Scalar
        pooled_image = np.reshape(pooled_image, (1, 1))
    
    plt.subplot(3, 2, 4)
    plt.imshow(pooled_image)
    plt.title("After Pooling")
    plt.axis('off')

    # Plot the final single value
    plt.subplot(3, 2, 5)
    plt.imshow(np.full_like(pooled_image, single_value), cmap='jet')
    plt.title(f"Final Single Value: {single_value:.2f}")
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
