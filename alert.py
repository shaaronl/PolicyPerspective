import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime

def send_alert(obj, confidence, current_time, current_location, snapshot_path, 
               sender_email, recipient_email, email_pw):
    subject = f"Security Alert: {obj} detected!"
    body = (
        f"Access denied for object {obj} with confidence {confidence:.2f}.\n"
        f"Time: {datetime.fromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Location: {current_location[1]}, {current_location[0]}"
    )

    # creating email w/ env vars
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    # attaching snapshot of weapon detected
    try:
        with open(snapshot_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
            # encode file in base64 for email
            encoders.encode_base64(part)  
            part.add_header(
                "Content-Disposition",
                f"attachment; filename={os.path.basename(snapshot_path)}",
            )
            msg.attach(part)
    except Exception as e:
        print(f"Error attaching snapshot: {e}")

    # send email with credentials entered
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, email_pw)
            server.sendmail(sender_email, recipient_email, msg.as_string())
            print(f"Alert sent for {obj} with confidence {confidence:.2f}")
    except Exception as e:
        print(f"email failed to send : {e}")
        
