import cv2
import supervision as sv
from ultralytics import YOLO
import tensorflow as tf
import casbin
import requests
from datetime import datetime

def check_gpu_config():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f"gpus detected:  {len(gpus)}")

def setup_camera():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Unable to access camera.")
        exit()
    return cam

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
            print("Failed to get location")
            return None, None
    except Exception as e:
        print(f"Error getting location: {e}")
        return None, None
# Getting location
country, city = get_location_from_ip()
if not city:
    print("Unable to retrieve location. Exiting.")
    exit()   

def main():

    # check to see if gpu is configured 
    check_gpu_config()
    
    # initialize webcam
    cap = setup_camera()

    # load YOLO model
    model = YOLO("yolov8l.pt")

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.BOTTOM_CENTER)

    e = casbin.Enforcer("abac_model.conf", "abac_policy.csv")

    current_location = get_location_from_ip()
    time_of_day = get_time_of_day()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("failed to capture frame, exiting")
            break
        
        # run YOLO inference
        result = model(frame)[0]

        # convert result to supervision detection objects
        detections = sv.Detections.from_ultralytics(result)
        frame = box_annotator.annotate(scene=frame, detections=detections)
        labels = []
        for class_id, confidence, in zip(detections.class_id, detections.confidence):
            class_name = model.model.names[int(class_id)]
            action = "access"

            print(f"Checking policy for {class_name}")
            print(f"tod: {time_of_day}, location: {current_location[1]}")
    
            if e.enforce(class_name, action, time_of_day, current_location[1]):
                labels.append(f"{class_name}({confidence:.2f}) - Allowed")
            else:
                labels.append(f"{class_name}({confidence:.2f}) - Denied")
        

        # annotate frame with labels
        annotated_frame = label_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        # display annotated frame (labels + bounding box)
        cv2.imshow("YOLOv8", annotated_frame)
        if cv2.waitKey(30) == 27:
            break

    # release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
