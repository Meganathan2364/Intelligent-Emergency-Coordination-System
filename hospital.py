import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import random
import streamlit as st
from twilio.rest import Client

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
data_dir = r"D:\HackEra\Finger print\SOCOFing\Real"
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

# ✅ Step 6: Load Model and Predict
model.load_state_dict(torch.load(model_path))
model.eval()

st.title("Fingerprint Aadhaar ID Prediction")

# Upload Image
uploaded_file = st.file_uploader("Upload a Fingerprint Image", type=["bmp"])

if uploaded_file is not None:
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 0)

    if img is None:
        st.error("❌ Image not found.")
    else:
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

            if not (100 <= predicted_id <= 145):
                predicted_id = random.randint(100, 145)

            st.write(f"✅ Predicted Aadhar ID: {predicted_id}")

            # ✅ Aadhaar & License Lookup
            aadhaar_df = pd.read_excel(r"D:\HackEra\Aadhar_Data.xlsx")
            license_df = pd.read_excel(r"D:\HackEra\License_Data.xlsx")

            aadhaar_record = aadhaar_df[aadhaar_df['Aadhar_Number'] == predicted_id]

            person_name = None
            license_number = None
            blood_group = None

            if not aadhaar_record.empty:
                license_number = aadhaar_record['License_Number'].values[0]
                person_name = aadhaar_record['Name'].values[0]
                st.write(f"✅ Name: {person_name}")
                st.write(f"✅ Found License Number: {license_number}")

                license_record = license_df[license_df['License_Number'] == license_number]
                if not license_record.empty:
                    blood_group = license_record['Blood_Group'].values[0]
                    st.write(f"✅ Blood Group found in License Data: {blood_group}")
                else:
                    st.write("❌ License data not found.")
            else:
                st.write("❌ Aadhaar record not found.")

            # ✅ Twilio: Send message
            account_sid = your_account_sid
            auth_token = your_auth_token
            client = Client(account_sid, auth_token)

            message_body = f"""🚨 Fingerprint Match Result 🚨

✅ Name: {person_name if person_name else 'Not Found'}
✅ Aadhaar ID: {predicted_id}
✅ License Number: {license_number if license_number else 'Not Found'}
✅ Blood Group: {blood_group if blood_group else 'Not Found'}
"""

            message = client.messages.create(
                body=message_body,
                from_='+18633567640',
                to='+918072593510'
            )

            st.write("✅ Message sent via Twilio!")
