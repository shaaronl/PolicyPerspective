import time
import casbin
import cv2
from ultralytics import YOLO
import supervision as sv
import os
from config import check_gpu_config, email_pw, sender_email, recipient_email
from camera import setup_camera
from context import get_location_from_ip, get_time_of_day
from alert import send_alert
from datetime import datetime


# custom function for casbin policy 
def confidence_greater(input_confidence, policy_confidence):
    return float(input_confidence) > float(policy_confidence)


def main():
    last_alert_time = None
    alert_interval = 3000
    snapshot_dir = "snapshots"
    os.makedirs(snapshot_dir, exist_ok=True)

    # check GPU configuration
    check_gpu_config()
    
    # initialize webcam
    cap = setup_camera()

    # load YOLO model
    model = YOLO("yolov8n.pt")

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.BOTTOM_CENTER)

    e = casbin.Enforcer("abac_model.conf", "abac_policy.csv")
    e.add_function("confidence_greater", confidence_greater)

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

        for class_id, confidence in zip(detections.class_id, detections.confidence):
            class_name = model.model.names[int(class_id)]
            
            # check access
            if e.enforce( confidence, class_name, time_of_day, current_location[1]):
                labels.append(f"{class_name}({confidence:.2f}) - Allowed")
            else:
                labels.append(f"{class_name}({confidence:.2f}) - Denied")
                print("DENIED!")
                current_time = time.time()
                if last_alert_time is None or current_time - last_alert_time >= alert_interval:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    snapshot_path = f"{snapshot_dir}/snapshot_{timestamp}.jpg"
                    cv2.imwrite(snapshot_path, frame)
                    print(f"Snapshot saved: {snapshot_path}")
                    send_alert(class_name, confidence, current_time, current_location, snapshot_path,
                                sender_email, recipient_email, email_pw)
                    last_alert_time = current_time

        # annotate frame with labels
        annotated_frame = label_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        # display annotated frame
        cv2.imshow("YOLOv8", annotated_frame)
        if cv2.waitKey(30) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
