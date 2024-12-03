import tensorflow as tf
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
email_pw = os.getenv("EMAIL_PASSWORD")
sender_email = os.getenv("SENDER_EMAIL")
recipient_email = os.getenv("RECIPIENT_EMAIL")

def check_gpu_config():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f"gpus detected:  {len(gpus)}")


