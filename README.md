# PolicyPerspective

`PolicyPerspective` is an object detection-based security system designed to enhance privacy and security through automated alerts. The system uses the YOLO object detection model to identify specific objects in real-time via a camera feed. It integrates with an **Attribute-Based Access Control (ABAC)** system, powered by **Casbin**, to trigger alerts based on detected objects, time, location, and confidence scores.

## Features

- **Real-time Object Detection**: Detects objects in a live camera feed using YOLOv8.
- **Policy-based Decision Making**: Uses **Casbin** to enforce **ABAC** policies, evaluating object type, confidence score, and context (e.g., time, location) for decision making.
- **Automated Alerts**: Sends alerts when certain objects (e.g., weapons) are detected based on predefined policies.
- **Email Notifications**: Notifies specified recipients via email when a relevant object is detected.
