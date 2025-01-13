import cv2
import math
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO(r"C:\Users\abayr\OneDrive\Desktop\System&Device\Project\meric_models\yolo11s.pt")

# Input and output video paths
input_video_path = r"test_videos\yellow.mp4"  # Replace with your input video path
output_video_path = "yellow_output.mp4"  # Desired output path

# Object classes
classNames = ['GoodCap', 'LooseCap', 'NoCap']

# Open video capture
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Cannot open video file!")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Create VideoWriter for saving output
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Initialize variables for FPS calculation
frame_times = []
fps_display_interval = 10  # Number of frames to average over

while True:
    # Start time for FPS calculation
    frame_start_time = time.time()

    # Read a frame
    ret, frame = cap.read()
    if not ret:
        break

    # Perform YOLO inference
    results = model(frame, stream=True, conf=0.6)

    # Process results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Class name
            cls = int(box.cls[0])
            if cls == 3:  # Skip class 3
                continue

            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Confidence score
            confidence = math.ceil((box.conf[0] * 100)) / 100

            # Display object details on the image
            cv2.putText(frame, classNames[cls], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, str(confidence), (x2, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

    # Calculate FPS
    frame_end_time = time.time()
    frame_time = frame_end_time - frame_start_time
    frame_times.append(frame_time)

    # Calculate average FPS over the last few frames
    if len(frame_times) > fps_display_interval:
        frame_times.pop(0)
    avg_fps = 1 / (sum(frame_times) / len(frame_times))

    # Add FPS display to the frame
    fps_text = f"FPS: {avg_fps:.2f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save the frame to output video
    out.write(frame)

    # Display the frame (optional)
    cv2.imshow("YOLO Inference", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved as {output_video_path}")
