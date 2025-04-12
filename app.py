import streamlit as st
import os
from streamlit_js_eval import streamlit_js_eval
from twilio.rest import Client
from geopy.geocoders import Nominatim
import threading

# ----------------- Configuration -----------------
UPLOAD_FOLDER = 'uploads'
account_sid = your_account_sid
auth_token = your_auth_token
twilio_number = '+16892606229'  # Replace with your Twilio number
target_numbers = ['+918754245519', '+919003223048', '+919047937507']  

# ----------------- Streamlit UI Setup -----------------
st.set_page_config(page_title="🚨 Emergency Alert System", layout="centered")
st.title("📍 Emergency File Upload with Location-Based Alert")

# ----------------- Get Location Using Browser -----------------
location = streamlit_js_eval(
    js_expressions="""
        new Promise((resolve, reject) => {
            navigator.geolocation.getCurrentPosition(
                (pos) => {
                    resolve({latitude: pos.coords.latitude, longitude: pos.coords.longitude});
                },
                (err) => {
                    resolve(null);
                }
            );
        })
    """,
    key="get_location"
)

# ----------------- Reverse Geocoding -----------------
def get_location_name(lat, lon):
    try:
        geolocator = Nominatim(user_agent="geoapi")
        location = geolocator.reverse((lat, lon), timeout=15, language='en')
        if location and location.address:
            return location.address
        return "Unknown location"
    except Exception as e:
        return f"Error: {e}"

# ----------------- Twilio Messaging -----------------
def send_sms(body):
    client = Client(account_sid, auth_token)
    for number in target_numbers:
        client.messages.create(
            body=body,
            from_=twilio_number,
            to=number
        )

def make_calls(twiml_message):
    client = Client(account_sid, auth_token)
    for number in target_numbers:
        client.calls.create(
            twiml=twiml_message,
            from_=twilio_number,
            to=number
        )

# ----------------- Alert Function -----------------
def alert_all(message_body, twiml_message):
    sms_thread = threading.Thread(target=send_sms, args=(message_body,))
    call_thread = threading.Thread(target=make_calls, args=(twiml_message,))
    sms_thread.start()
    call_thread.start()

# ----------------- UI and Trigger -----------------
if location and 'latitude' in location and 'longitude' in location:
    lat = location['latitude']
    lon = location['longitude']
    maps_link = f"https://www.google.com/maps?q={lat},{lon}"
    location_name = get_location_name(lat, lon)

    st.success("📍 Location Detected")
    st.write(f"**Latitude:** {lat}")
    st.write(f"**Longitude:** {lon}")
    st.write(f"**Exact Location:** {location_name}")
    st.markdown(f"[🌍 Open in Google Maps]({maps_link})", unsafe_allow_html=True)
else:
    st.warning("⚠ Unable to detect location. Please allow location permission.")

uploaded_file = st.file_uploader("📁 Upload an image or file related to the emergency")

if st.button("Upload and Send Alerts"):
    if uploaded_file:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("✅ File uploaded successfully!")

        if location_name and lat and lon:
            message_body = (
                f"🚨 Emergency Alert:\n"
                f"An accident has occurred at: {location_name}\n"
                f"📍 Coords: {lat}, {lon}\n"
                f"🗺View on Google Maps: https://www.google.com/maps?q={lat},{lon}\n"
                f"Please respond immediately."
            )

            twiml_message = f"""<Response>
  <Say language="en-IN" voice="Polly.Aditi">
    Emergency Alert! An accident has occurred at: {location_name}. 
    Location: Latitude {lat}, Longitude {lon}. 
    Please respond immediately.
  </Say>
</Response>"""

            alert_all(message_body, twiml_message)

            st.success("📩 SMS and 📞 Calls sent successfully!")
        else:
            st.error("❌ Location info missing. Cannot proceed.")
    else:
        st.warning("⚠ Please upload a file before sending alerts.")
