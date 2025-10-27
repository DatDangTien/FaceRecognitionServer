import cv2
import sys

camera_index = 0
if len(sys.argv) > 1:
    camera_index = int(sys.argv[1])

cap = cv2.VideoCapture(camera_index)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0:
    fps = 30  # Default FPS if not available
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define codec and create VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

print("Recording video... Press Ctrl+C to stop")

frame_count = 0
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Captured {frame_count} frames")
except KeyboardInterrupt:
    print("\nRecording stopped")

cap.release()
out.release()
print(f"Video saved as output.mp4 ({frame_count} frames)")