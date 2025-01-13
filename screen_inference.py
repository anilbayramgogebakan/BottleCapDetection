import numpy as np
from PIL import ImageGrab
import cv2
import math
from ultralytics import YOLO
import time

# Define the bounding box for the screen region to capture
bbox = (100, 100, 800, 600)  # Adjust as needed

# Load YOLO model
model = YOLO(r"C:\Users\abayr\OneDrive\Desktop\System&Device\Project\meric_models\yolo11n.pt")

# Object classes
classNames = ['GoodCap', 'LooseCap', 'NoCap']

# Initialize variables for FPS calculation
frame_times = []
fps_display_interval = 10  # Number of frames to average over
start_time = time.time()

while True:
    # Record the start time of the frame
    frame_start_time = time.time()

    # Capture the defined region of the screen
    printscreen_pil = ImageGrab.grab(bbox=bbox)
    # printscreen_pil = ImageGrab.grab()
    img = np.array(printscreen_pil, dtype='uint8')

    # Perform YOLO inference
    results = model(img, stream=True, conf=0.6)

    # Process results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Class name
            cls = int(box.cls[0])
            if cls==3:
                continue

            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw the bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Confidence score
            confidence = math.ceil((box.conf[0] * 100)) / 100

            # Display object details on the image
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

            # Display object details on the image
            org = [x2, y2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 1
            cv2.putText(img, str(confidence), org, font, fontScale, color, thickness)

    # Calculate FPS
    frame_end_time = time.time()
    frame_time = frame_end_time - frame_start_time
    frame_times.append(frame_time)

    # Calculate average FPS over the last few frames
    if len(frame_times) > fps_display_interval:
        frame_times.pop(0)
    avg_fps = 1 / (sum(frame_times) / len(frame_times))

    # Add FPS display to the image
    fps_text = f"FPS: {avg_fps:.2f}"
    cv2.putText(img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the image with detections
    cv2.imshow('Screen Capture', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV

    # Exit on pressing 'q'
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
