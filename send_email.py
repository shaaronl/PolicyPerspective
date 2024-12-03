from dotenv import load_dotenv
import smtplib
import os

# Load environment variables
load_dotenv()
email_pw = os.getenv("EMAIL_PASSWORD")
sender_email = os.getenv("SENDER_EMAIL")
recipient_email = os.getenv("RECIPIENT_EMAIL")

def send_alert(obj, confidence):
    subject = f"Security Alert: {obj} detected!"
    body = f"Access denied for object {obj} with confidence {confidence}"

    message = f"From: {sender_email}\nTo: {recipient_email}\nSubject: {subject}\n\n{body}"
    
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, email_pw)
            server.sendmail(sender_email, recipient_email, message)
            print(f"Alert sent for {obj} with confidence {confidence}")
            print(f"Sender: {sender_email}, Password: {email_pw[:2]}****, Recipient: {recipient_email}")

    except Exception as e:
        print(f"Failed to send email: {e}")

def main():
    send_alert("knife", 0.86)

if __name__ == "__main__":
    main()
