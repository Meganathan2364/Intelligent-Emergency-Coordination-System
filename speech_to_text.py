from flask import Flask, request
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

# Twilio credentials
account_sid = 'your_account_sid'
auth_token = 'your_auth_token'
twilio_number = '+your_twilio_number'
target_number = '+recipient_number'

app = Flask(__name__)

# Send message once when script starts
def send_message():
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        body="Hello from Meganathan's Twilio app!",
        from_=twilio_number,
        to=target_number
    )
    print(f"Message sent! SID: {message.sid}")

# Receive message via webhook
@app.route("/sms", methods=['POST'])
def sms_reply():
    incoming_msg = request.form.get('Body')
    sender = request.form.get('From')
    
    print(f"📩 Received message: '{incoming_msg}' from {sender}")
    
    resp = MessagingResponse()
    resp.message("Thanks for your message! We'll get back to you.")
    
    return str(resp)

if __name__ == "__main__":
    send_message()
    app.run(debug=True)

