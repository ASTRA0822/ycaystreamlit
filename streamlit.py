import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

model = load_model('best_cifar10_model.h5')

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

st.title("CIFAR-10 Image Classifier 🚀")
st.write("Upload a CIFAR-10 image (32x32 or resized) to predict its class.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=False)

    img = img.resize((32, 32))
    img_array = keras_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    class_label = class_names[class_idx]
    confidence = np.max(prediction)

    st.success(f"Prediction: **{class_label}** ({confidence*100:.2f}%)")
