import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf

# Page Configuration
st.set_page_config(
    page_title="Aerial Cactus Classifier", page_icon="🌵", layout="centered"
)

st.title("🌵 Aerial Cactus Identification")
st.write(
    "Upload an aerial image to detect the presence of columnar cacti (*Neobuxbaumia tetetzo*)."
)


# ---------------------------------------------------------
# LOAD MODEL (Cached to optimize performance)
# ---------------------------------------------------------
@st.cache_resource
def load_cactus_model():
    model_path = "cactus_model.h5"  # Replace with your trained model path (.h5 or .keras)
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        return None


model = load_cactus_model()

# ---------------------------------------------------------
# UPLOAD IMAGE & PREDICTION
# ---------------------------------------------------------
st.subheader("📸 Upload Image & Predict")

uploaded_file = st.file_uploader(
    "Choose or drag and drop an image (PNG, JPG, JPEG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    # 1. Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # 2. Image Preprocessing
    # Aerial Cactus dataset images are standard 32x32 pixels
    img_resized = image.resize((32, 32))
    img_array = np.array(img_resized)

    # Convert RGBA to RGB if necessary
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]

    # Normalize pixel values to [0, 1]
    img_array = img_array.astype("float32") / 255.0

    # Expand dimensions for model input batch (1, 32, 32, 3)
    img_batch = np.expand_dims(img_array, axis=0)

    # 3. Predict Button
    if st.button("Predict Cactus Presence 🚀"):
        if model is not None:
            with st.spinner("Analyzing image..."):
                prediction = model.predict(img_batch)[0][0]

                # Evaluate prediction confidence
                if prediction >= 0.5:
                    confidence = prediction * 100
                    st.success(
                        f"🌵 **Result: CACTUS DETECTED!** (Confidence: {confidence:.2f}%)"
                    )
                else:
                    confidence = (1 - prediction) * 100
                    st.error(
                        f"❌ **Result: NO CACTUS DETECTED** (Confidence: {confidence:.2f}%)"
                    )

                # Prediction probability bar
                st.progress(float(prediction))
        else:
            st.warning(
                "⚠ **Model File Missing!** Please place your 'cactus_model.h5' file in the root directory."
            )
