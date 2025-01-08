from tensorflow.keras.models import load_model

# Path to the model file
model_path = "model.keras"

# Attempt to load the model
try:
    model = load_model(model_path)
    model.summary()  # Print model structure
    print("Model loaded successfully and appears to be complete.")
except Exception as e:
    print(f"Error loading model: {e}")
