import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the pre-trained model
model = keras.models.load_model('your_model_path.h5')

# List of class names for the model output
class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']

# Dictionary of precautions for each class
precautions = {
    'No_DR': 'No precautions required.',
    'Mild': 'Visit an eye specialist regularly.',
    'Moderate': 'Monitor your blood sugar levels. Follow the advice of your doctor.',
    'Severe': 'Consult an eye specialist immediately. Follow their instructions.',
    'Proliferate_DR': 'Seek immediate medical attention. Follow the advice of an eye specialist.'
}

def classify_image(image):
    # Preprocess the image
    image = image.resize((224, 224))  # Resize the image
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Perform classification
    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    class_label = class_names[class_index]

    return class_label

# Set Streamlit app title
st.title("Diabetic Retinopathy Detection")

# Add image below the title
title_image = Image.open("art.jpeg")
st.image(title_image, use_column_width=True)

custom_style = """
<style>
body {
    background-image: url("bg1.webp");
    background-size: cover;
}

.side-bar {
    background-color: #f0f0f0;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.2);
}
</style>
"""
st.markdown(custom_style, unsafe_allow_html=True)

# Add GIF in the sidebar
st.sidebar.image("retina.gif", use_column_width=True)


# Display side decorations
st.sidebar.markdown("INSTRUCTIONS")
st.sidebar.markdown(
    '''
    1. *Upload Image:*
       - Look for the "Upload an image" section.
       - Click on the "Browse Files" button.
       - Choose an image file (JPG, JPEG, or PNG) from your computer.

    2. *View Uploaded Image:*
       - Once the image is selected, it will be displayed on the page.
       - Take a moment to review the uploaded image.

    3. *Classify Image:*
       - Locate the "Classify" button.
       - Click on the "Classify" button to start the classification process.

    4. *View Classification Result:*
       - The system will process the uploaded image.
       - The classification result will be displayed on the page.
       - Take note of the predicted class label (e.g., "Mild", "Severe", etc.).

    5. *Follow Precautions (if provided):*
       - If any precautions or recommendations are mentioned based on the classified label, they will be displayed.
       - Read and follow the mentioned precautions to ensure appropriate care.
    ''')

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

pre_upload_image = Image.open("Sample.jpeg")
st.image(pre_upload_image, caption="SAMPLE IMAGE", use_column_width=True,width=300)

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    if st.button("Preview"):
        thumbnail_size = (150, 150)
        image.thumbnail(thumbnail_size)
        st.image(image, caption="Preview)")

    # Classify when the classify button is clicked
    if st.button("Classify"):
        # Perform classification
        class_label = classify_image(image)
        st.subheader("Classification Result")
        st.write("Predicted Class: ", class_label)

        # Display precautions based on the class label
        if class_label in precautions:
            st.subheader("Precautions")
            st.write(precautions[class_label])
        else:
            st.write("No precautions available for the classified label.")