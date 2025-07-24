import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# CIFAR-10 class labels
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Load your trained model
@st.cache_resource
def load_cnn_model():
    return load_model("cifar10_model.h5")  # Change path if needed

model = load_cnn_model()

st.title("CIFAR-10 Image Classification 🎯")
st.write("Upload a 32x32 image or any image to classify it using your trained CNN model.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocessing
    img_resized = image.resize((32, 32))
    img_array = img_to_array(img_resized)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # (1, 32, 32, 3)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.markdown(f"### Prediction: **{class_names[predicted_class]}**")
    st.markdown(f"**Confidence:** {confidence:.2f}%")
