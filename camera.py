import cv2

def setup_camera():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Unable to access camera.")
        exit()
    return cam
