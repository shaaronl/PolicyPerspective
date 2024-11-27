import cv2
import mediapipe as mp
import casbin
import requests
import base64
import logging
from datetime import datetime
from inference_sdk import InferenceHTTPClient
from roboflow import Roboflow

# Setup logging
logging.basicConfig(level=logging.INFO)

# Casbin enforcer which loads the model and policy


def load_casbin_enforcer():
    try:
        return casbin.Enforcer("abac_model.conf", "abac_policy.csv")
    except Exception as excp:
        logging.error(f"Error loading Casbin model or policy: {excp}")
        exit()


# OpenCV camera setup
def setup_camera():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        logging.error("Unable to access the camera.")
        exit()
    return cam


# Roboflow API setup
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="pmyGkNHL30HBOxwxxMZp"
)


def detect_knife(frame):
    """Detects knife in the given frame using Roboflow API."""
    try:
        _, encoded_image = cv2.imencode('.jpg', frame)
        image_base64 = base64.b64encode(
            encoded_image.tobytes()).decode('utf-8')
        result = CLIENT.infer(image_base64, model_id="knife-detection-hgvy2/1")

        # Filter out low-confidence predictions
        predictions = result.get("predictions", [])
        high_confidence_predictions = [
            pred for pred in predictions if pred['confidence'] > 0.5]

        logging.info(
            f"Number of knife detections: {len(high_confidence_predictions)}")
        for pred in high_confidence_predictions:
            logging.info(f"Prediction - Confidence: {pred['confidence']}")

        return high_confidence_predictions

    except Exception as e:
        logging.error(f"Error from Roboflow API: {e}")
        return []


def check_policy(obj, act, time_of_day, location):
    """Checks if the action is allowed according to the Casbin policy."""
    enforcer = load_casbin_enforcer()
    if enforcer.enforce(obj, act, time_of_day, location):
        logging.info("Access granted.")
        return True
    else:
        logging.info("Access denied.")
        return False

# Helper functions for Casbin context


def get_time_of_day():
    """Returns the current time of day as a string."""
    current_hour = datetime.now().hour
    if 5 <= current_hour < 12:
        return "morning"
    elif 12 <= current_hour < 18:
        return "afternoon"
    else:
        return "evening"


def get_location_from_ip():
    """Fetches the country and city based on IP."""
    try:
        response = requests.get("http://ip-api.com/json/")
        data = response.json()
        if data["status"] == "success":
            return data["country"], data["city"]
        else:
            logging.warning("Failed to get location")
            return None, None
    except Exception as e:
        logging.error(f"Error getting location: {e}")
        return None, None


# Getting location
country, city = get_location_from_ip()
if not city:
    logging.error("Unable to retrieve location. Exiting.")
    exit()


# Main loop
def main():
    # Set up camera and load Casbin enforcer
    cam = setup_camera()

    while True:
        ret, frame = cam.read()
        if not ret:
            logging.error("Failed to grab frame.")
            break

        # Flip the frame horizontally
        frame_resized = cv2.flip(frame, 1)

        # Detect knives in the frame
        detections = detect_knife(frame_resized)

        knife_detected = len(detections) > 0

        # If knife detected, check policy
        if knife_detected:
            time_of_day = get_time_of_day()

            for detection in detections:
                # Extract coordinates and confidence
                x, y, w, h = int(detection["x"]), int(detection["y"]), int(
                    detection["width"]), int(detection["height"])
                confidence = detection['confidence']

                # Draw bounding box around the knife
                cv2.rectangle(frame_resized, (x - w // 2, y - h // 2),
                              (x + w // 2, y + h // 2), (0, 0, 255), 2)

                # Display the confidence score on the frame
                cv2.putText(frame_resized, f"Knife ({confidence:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Check Casbin policy for this detection
                access_granted = check_policy(
                    'knife', 'access', time_of_day, city)
                if access_granted:
                    status = "Access Allowed"
                    status_color = (0, 255, 0)
                    logging.info("Action allowed for knife detection.")
                else:
                    status = "Access Denied"
                    status_color = (0, 0, 255)
                    logging.warning("Action denied for knife detection.")
                cv2.putText(frame_resized, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        cv2.imshow("Knife Detection", frame_resized)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



import cv2
import torch
import logging
from datetime import datetime
import casbin
import requests
import base64

# Setup logging
logging.basicConfig(level=logging.INFO)

# Casbin enforcer which loads the model and policy
def load_casbin_enforcer():
    try:
        return casbin.Enforcer("abac_model.conf", "abac_policy.csv")
    except Exception as excp:
        logging.error(f"Error loading Casbin model or policy: {excp}")
        exit()

# OpenCV camera setup
def setup_camera():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        logging.error("Unable to access the camera.")
        exit()
    return cam

# Load YOLOv5 model (use the path to your downloaded model)
def load_yolov5_model():
    try:
        model = torch.hub.load('ultralytics/yolov5:v5.0', 'custom', path='path_to_your_model/best.pt')  # Replace with the correct path
        logging.info("YOLOv5 model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading YOLOv5 model: {e}")
        exit()

# Use YOLOv5 to detect knife in the frame
def detect_knife(frame, model):
    """Detects knife in the given frame using YOLOv5."""
    try:
        results = model(frame)  # Perform inference on the frame

        # Parse the results
        predictions = results.pred[0]  # Get predictions from the first image in the batch
        high_confidence_predictions = [
            pred for pred in predictions if pred[4] > 0.5]  # Filter out predictions with low confidence

        logging.info(f"Number of knife detections: {len(high_confidence_predictions)}")
        for pred in high_confidence_predictions:
            logging.info(f"Prediction - Confidence: {pred[4]}")

        return high_confidence_predictions

    except Exception as e:
        logging.error(f"Error detecting knife: {e}")
        return []
    
def check_policy(obj, act, time_of_day, location):
    """Checks if the action is allowed according to the Casbin policy."""
    enforcer = load_casbin_enforcer()
    if enforcer.enforce(obj, act, time_of_day, location):
        logging.info("Access granted.")
        return True
    else:
        logging.info("Access denied.")
        return False
    
# Helper functions for Casbin context
def get_time_of_day():
    """Returns the current time of day as a string."""
    current_hour = datetime.now().hour
    if 5 <= current_hour < 12:
        return "morning"
    elif 12 <= current_hour < 18:
        return "afternoon"
    else:
        return "evening"

def get_location_from_ip():
    """Fetches the country and city based on IP."""
    try:
        response = requests.get("http://ip-api.com/json/")
        data = response.json()
        if data["status"] == "success":
            return data["country"], data["city"]
        else:
            logging.warning("Failed to get location")
            return None, None
    except Exception as e:
        logging.error(f"Error getting location: {e}")
        return None, None

# Getting location
country, city = get_location_from_ip()
if not city:
    logging.error("Unable to retrieve location. Exiting.")
    exit()

# Main loop
def main():
    # Set up camera and load Casbin enforcer
    cam = setup_camera()
    model = load_yolov5_model()  # Load the YOLOv5 model

    while True:
        ret, frame = cam.read()
        if not ret:
            logging.error("Failed to grab frame.")
            break

        # Flip the frame horizontally
        frame_resized = cv2.flip(frame, 1)

        # Detect knives in the frame
        detections = detect_knife(frame_resized, model)

        knife_detected = len(detections) > 0

        # If knife detected, check policy
        if knife_detected:
            time_of_day = get_time_of_day()

            for detection in detections:
                # Extract coordinates and confidence
                x, y, w, h = int(detection[0]), int(detection[1]), int(detection[2] - detection[0]), int(detection[3] - detection[1])
                confidence = detection[4]

                # Draw bounding box around the knife
                cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # Display the confidence score on the frame
                cv2.putText(frame_resized, f"Knife ({confidence:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Check Casbin policy for this detection
                access_granted = check_policy(
                    'knife', 'access', time_of_day, city)
                if access_granted:
                    status = "Access Allowed"
                    status_color = (0, 255, 0)
                    logging.info("Action allowed for knife detection.")
                else:
                    status = "Access Denied"
                    status_color = (0, 0, 255)
                    logging.warning("Action denied for knife detection.")
                cv2.putText(frame_resized, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        cv2.imshow("Knife Detection", frame_resized)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
