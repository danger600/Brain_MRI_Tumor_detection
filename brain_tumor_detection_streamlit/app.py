# ==========================================
# 22. Streamlit Web App for Brain Tumor Prediction
# ==========================================

# Import Required Libraries and Modules
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image


# Configure the Streamlit page title and browser icon
st.set_page_config(page_title="Brain Tumor Detection")

# Display header and introductory instructions on the web interface
st.title("Brain Tumor Detection")
st.write("Upload an MRI image to check for a brain tumor.")

MODEL_PATH = Path(__file__).resolve().parent / "Model" / "brain_tumor_model.h5"


@st.cache_resource
def load_model(model_path: str):
    import tensorflow as tf

    return tf.keras.models.load_model(model_path)


model = None
if MODEL_PATH.is_file():
    try:
        model = load_model(str(MODEL_PATH))
    except Exception:
        st.error("The trained model could not be loaded. Check that the model file is valid and compatible.")
else:
    st.warning(
        "The trained model is not included in this repository. Uploads can be previewed, "
        "but predictions are unavailable until `Model/brain_tumor_model.h5` is added "
        "to this app folder."
    )


# Create a file uploader widget accepting common image formats
uploaded_file = st.file_uploader(
    "Upload MRI image",
    type=["jpg", "jpeg", "png"]
)

# Check if an image has been uploaded by the user
if uploaded_file is not None:

    # Open the uploaded image using PIL
    image = Image.open(uploaded_file)

    # Display the uploaded image on the web app interface
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if model is None:
        st.info("A prediction is unavailable until a valid trained model is added to this app folder.")
    else:
        # Convert the image to RGB, resize to the model input dimensions, and add a batch axis.
        image = image.convert("RGB")
        x = np.asarray(image.resize((128, 128))).reshape(1, 128, 128, 3)

        # Run prediction on the preprocessed image array without verbose logging.
        res = model.predict(x, verbose=0)
        st.write("Model output:", res)

        classification = int(np.argmax(res[0]))
        confidence = float(res[0][classification]) * 100

        if classification == 0:
            st.error(f"Brain Tumor Detected! ({confidence:.2f}% confidence)")
        else:
            st.success(f"No Tumor Detected ({confidence:.2f}% confidence)")

        st.caption("Research prototype only; this output is not a medical diagnosis.")
