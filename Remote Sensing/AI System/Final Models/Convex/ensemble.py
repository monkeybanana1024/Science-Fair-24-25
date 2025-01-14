import tensorflow as tf

def summarize_model(model):
    """
    Print a detailed summary of the model's components, including:
    - Architecture
    - Optimizer settings
    - Loss function
    - Metrics
    - Weights shapes and counts
    """
    print("Model Summary and Settings\n")
    
    # 1. Architecture
    print("=== Architecture ===")
    model.summary()
    print("\n")

    # 2. Optimizer Settings
    print("=== Optimizer Settings ===")
    if model.optimizer:
        optimizer_config = model.optimizer.get_config()
        for key, value in optimizer_config.items():
            print(f"{key}: {value}")
    else:
        print("No optimizer found.")
    print("\n")

    # 3. Loss Function
    print("=== Loss Function ===")
    if model.loss:
        print(f"Loss: {model.loss}")
    else:
        print("No loss function found.")
    print("\n")

    # 4. Metrics
    print("=== Metrics ===")
    if model.metrics:
        for metric in model.metrics:
            print(f"Metric: {metric.name}")
    else:
        print("No metrics found.")
    print("\n")

    # 5. Weights and Layers
    print("=== Weights and Layers ===")
    for layer in model.layers:
        print(f"Layer: {layer.name}")
        print(f"  Trainable: {layer.trainable}")
        print(f"  Weights: {[w.shape for w in layer.get_weights()]}")
        print("\n")

    # 6. Total Weights
    total_params = model.count_params()
    print(f"=== Total Parameters: {total_params:,} ===")

    print("\nSummary Complete.")

# Example Usage
if __name__ == "__main__":
    # Load the saved model
    model_path = "AI System/Results/Topography/model.keras"  # Replace with your model's path
    model = tf.keras.models.load_model(model_path)
    
    # Summarize the model
    summarize_model(model)
