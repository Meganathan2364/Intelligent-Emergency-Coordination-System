import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import random
from geopy.distance import geodesic
from twilio.rest import Client
import re

# === Twilio Credentials ===
TWILIO_SID = "ACdd146e119c76ee78c8076d5bc802bd72"
TWILIO_TOKEN = "9f88e79017a68ef7761f6139f6100bfa"
TWILIO_PHONE = "+16892606229"  # ✅ Make sure this is your actual Twilio number
client = Client(TWILIO_SID, TWILIO_TOKEN)

# === Ambulance Location (Example Coordinates) ===
ambulance_location = (13.0350, 80.2500)

# === Load Hospitals Dataset ===
hospital_df = pd.read_csv("hospital_list.csv")  # file from your upload

# === Format Indian Numbers into E.164 (e.g., +91XXXXXXXXXX) ===
def format_number(phone):
    phone = re.sub(r'\D', '', str(phone))  # Remove non-digits
    if len(phone) == 10:
        return '+91' + phone
    elif len(phone) == 12 and phone.startswith('91'):
        return '+' + phone
    elif phone.startswith('+91') and len(phone) == 13:
        return phone
    else:
        return None  # Invalid number

# === Find Nearest Hospitals ===
def get_nearest_hospitals(current_location, top_n=3):
    nearby = []
    for _, row in hospital_df.iterrows():
        try:
            hosp_location = (row["latitude"], row["longitude"])
            distance = geodesic(current_location, hosp_location).km
            phone = format_number(row["phone"])
            if phone:
                nearby.append({
                    "name": row["name"],
                    "phone": phone,
                    "distance_km": round(distance, 2),
                    "address": row.get("address", "Unknown")
                })
        except:
            continue
    nearby.sort(key=lambda x: x["distance_km"])
    return nearby[:top_n]

# === Send SMS Alert ===
def send_alerts(hospitals):
    for hosp in hospitals:
        message_body = f"""🚨 Emergency Alert 🚑
An accident has occurred near location {ambulance_location}.
Please prepare for emergency response.

Nearest Hospital: {hosp['name']}
Distance: {hosp['distance_km']} km
Address: {hosp['address']}"""

        try:
            message = client.messages.create(
                body=message_body,
                from_=TWILIO_PHONE,
                to=hosp["phone"]
            )
            print(f"✅ Message sent to {hosp['name']} ({hosp['phone']}) | SID: {message.sid}")
        except Exception as e:
            print(f"❌ Failed to send to {hosp['name']} ({hosp['phone']}): {e}")


# ✅ Step 1: Load and preprocess fingerprint data
def load_data(data_dir, img_size=(96, 96)):
    X, y = [], []
    labels = {}
    label_count = 0

    for file in os.listdir(data_dir):
        if not file.lower().endswith(".bmp"):
            continue
        img_path = os.path.join(data_dir, file)
        img = cv2.imread(img_path, 0)
        if img is None:
            continue
        img = cv2.resize(img, img_size)
        X.append(img)

        label = int(file.split("__")[0])
        if label not in labels:
            labels[label] = label_count
            label_count += 1
        y.append(labels[label])

    X = np.array(X).reshape(-1, 1, img_size[0], img_size[1]).astype(np.float32) / 255.0
    y = np.array(y)
    return X, y, labels

# ✅ Step 2: CNN Model Definition
class FingerprintCNN(nn.Module):
    def __init__(self, num_classes):
        super(FingerprintCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 22 * 22, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ✅ Step 3: Load Dataset
data_dir = r"D:\HackEra\Finger print\SOCOFing\Real"  # Update this to the path of your fingerprint data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X, y, label_map = load_data(data_dir)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=32)

# ✅ Step 4: Train the Model
model = FingerprintCNN(num_classes=len(label_map)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    total_loss = 0
    for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/10"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"✅ Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

# ✅ Step 5: Save Model
model_path = "fingerprint_model.pth"
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved at: {model_path}")

# ✅ Step 6: Load Model and Predict on New Image
def predict_fingerprint(img_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()

    img = cv2.imread(img_path, 0)
    if img is None:
        return "❌ Image not found."

    img_resized = cv2.resize(img, (96, 96)).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_resized).reshape(1, 1, 96, 96).to(device)

    with torch.no_grad():
        prediction = model(img_tensor)
        predicted_class = torch.argmax(prediction).item()
        id_map = {v: k for k, v in label_map.items()}

        if predicted_class in id_map:
            predicted_id = id_map[predicted_class]
        else:
            predicted_id = random.randint(100, 145)

        return predicted_id

# ✅ Step 7: Fetch Aadhaar Data Based on Predicted ID (Aadhaar number)
def fetch_aadhaar_data(predicted_id):
    aadhaar_df = pd.read_excel(r"D:\HackEra\Aadhar_Data.xlsx")  # Load Aadhaar data
    license_df = pd.read_excel(r"D:\HackEra\License_Data.xlsx")  # Load License data

    aadhaar_record = aadhaar_df[aadhaar_df['Aadhar_Number'] == predicted_id]
    
    if not aadhaar_record.empty:
        license_number = aadhaar_record['License_Number'].values[0]
        license_record = license_df[license_df['License_Number'] == license_number]
        
        if not license_record.empty:
            blood_group = license_record['Blood_Group'].values[0]
            return blood_group
    return "❌ No matching data found."

# ✅ Step 8: Streamlit Interface
st.title("Fingerprint ID Recognition & Emergency Alert System")

# Upload fingerprint image
uploaded_file = st.file_uploader("Upload Fingerprint Image", type=["bmp"])
if uploaded_file is not None:
    # Save the uploaded image temporarily
    temp_img_path = "uploaded_fingerprint.bmp"
    with open(temp_img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Predict fingerprint ID
    predicted_id = predict_fingerprint(temp_img_path)
    st.write(f"✅ Predicted Fingerprint ID: {predicted_id}")

    # Fetch Aadhaar and License Data
    blood_group = fetch_aadhaar_data(predicted_id)
    st.write(f"✅ Blood Group: {blood_group}")

    # Find and send alerts to nearest hospitals
    nearest_hospitals = get_nearest_hospitals(ambulance_location)
    if nearest_hospitals:
        send_alerts(nearest_hospitals)
        st.write(f"✅ Alerts sent to nearest hospitals.")
    else:
        st.write("⚠️ No valid hospital numbers found.")

    # Display the uploaded image
    img = cv2.imread(temp_img_path, 0)
    st.image(img, caption="Uploaded Fingerprint Image", use_column_width=True, channels="GRAY")
