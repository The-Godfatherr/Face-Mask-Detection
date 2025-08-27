from tensorflow.keras.models import load_model

# Load the existing model
model = load_model('mask_detector.h5')

# Save the model in the new format
model.save('mask_detector.keras')  # Updated to use .keras extension
