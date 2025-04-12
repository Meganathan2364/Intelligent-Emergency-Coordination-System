# main_script.py
import os
import cv2
import torch
import numpy as np
import pandas as pd
import streamlit as st
import random
from twilio import send_message  # Import the send_message function from twilio.py

# Your existing code for loading data and model
# ...

# Function to split the message into smaller chunks
def split_message(message, max_length=160):
    return [message[i:i+max_length] for i in range(0, len(message), max_length)]

# Split the message for accident alert
def send_accident_alerts():
    # Message 1: Accident Location
    message_1 = "Accident happened in Sathyabama University, OMR, Chennai, 600119. Please respond urgently."
    message_1_chunks = split_message(message_1)

    # Message 2: Person Information & Urgency
    message_2 = "The person in the accident is severely injured with more blood loss. The required blood group is AB+. Please wait for the bed and doctor availability."
    message_2_chunks = split_message(message_2)

    # Message 3: Nearest Hospitals
    message_3 = """
    Nearest hospitals:
    1. Sathyabama Medical Center
    2. Global Care Hospital
    3. Chennai General Hospital
    """
    message_3_chunks = split_message(message_3)

    # Return all the chunks
    return message_1_chunks + message_2_chunks + message_3_chunks

# Streamlit Interface
st.title("Fingerprint Aadhaar ID Prediction")

# Upload Image
uploaded_file = st.file_uploader("Choose a fingerprint image...", type=["bmp"])

# Send the accident messages
if uploaded_file is not None:
    # Process the fingerprint image and get the ID as before
    # ...

    # Now send the messages using the Twilio function
    all_messages = send_accident_alerts()
    
    # Send each chunk of messages
    for msg in all_messages:
        response = send_message.send_message("+918072593510", msg)  # Replace with the recipient's phone number
        st.write(response)  # Display the response from Twilio

    st.write(f"✅ Accident alerts sent successfully.")
