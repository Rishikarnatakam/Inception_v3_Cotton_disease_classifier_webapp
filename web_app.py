import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load the pre-trained Keras model (.h5 file)
@st.cache_resource
def load_keras_model():
    model = load_model("finetune3.h5")  # Load model from current directory
    return model

model = load_keras_model()

# Preprocess the image to match the model's input shape
def preprocess_image(image, target_size):
    image = image.resize(target_size)  # Resize the image to match the input size required by the model
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Class mapping (replace this with your actual mappings)
class_mapping = {
    0: 'APHIDS',
    1: 'ARMY WORM',
    2: 'BACTERIAL BLIGHT',
    3: 'HEALTHY',
    4: 'POWDERY MILDEW',
    5: 'TARGET SPOT'
    # Add more mappings as needed
}

# Predict the disease from the image
def predict_disease(image):
    processed_image = preprocess_image(image, target_size=(224, 224))  # Adjust target size based on your model
    prediction = model.predict(processed_image)

    # Get the predicted class index (highest probability)
    predicted_class_index = np.argmax(prediction[0])

    # Map the predicted class index to a disease label
    predicted_disease = class_mapping.get(predicted_class_index, "Unknown Disease")

    return predicted_disease

# Streamlit app UI
st.title("Cotton Disease Prediction")
st.markdown("""
This application uses a deep learning model to predict cotton plant diseases from images.
Upload an image to get a prediction.
""")

# Handle Image Upload
image = None  # Initialize image variable
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

# Predict when the image is ready
if st.button("Predict"):
    if image is not None:
        with st.spinner('Analyzing image...'):
            predicted_disease = predict_disease(image)
        
        if predicted_disease == "HEALTHY":
            st.success(f"The plant appears to be HEALTHY")
        else:
            st.warning(f"The predicted disease is: {predicted_disease}")
    else:
        st.error("Please upload an image.")
