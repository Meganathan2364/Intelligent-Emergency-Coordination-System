import streamlit as st
import easyocr
import numpy as np
from PIL import Image

st.set_page_config(page_title="Vehicle Number Plate Reader", layout="centered")
st.title("🚘 Vehicle Number Plate Reader (India Format)")

# Upload section
uploaded_file = st.file_uploader("Upload Number Plate Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image for easyocr
    img_array = np.array(image)

    # OCR Processing
    with st.spinner("🔍 Extracting text from number plate..."):
        reader = easyocr.Reader(['en'])  # English only
        result = reader.readtext(img_array)

        # Show extracted text
        st.subheader("📌 Extracted Text:")
        for (bbox, text, confidence) in result:
            st.success(f"{text}  (Confidence: {round(confidence * 100, 2)}%)")
