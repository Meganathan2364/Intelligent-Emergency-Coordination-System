# twilio.py
from twilio.rest import Client

# Twilio setup
twilio_account_sid = your_account_sid
twilio_auth_token = your_auth_token
twilio_phone_number = your_ph_no

# Create a Twilio client
client = Client(twilio_account_sid, twilio_auth_token)

def send_message(to, body):
    """Send SMS using Twilio"""
    try:
        message = client.messages.create(
            body=body,
            from_=twilio_phone_number,
            to=to
        )
        return f"Message sent successfully: {message.sid}"
    except Exception as e:
        return f"Error sending message: {str(e)}"
