"""
Real-time Face Detection - Versi Sederhana
Deteksi wajah menggunakan webcam dengan YOLOv8
"""

import cv2
from ultralytics import YOLO

# Load model
print("Loading model...")
model = YOLO('runs/face_result/weights/best.pt')
print("Model loaded!")

# Check available cameras
def list_cameras(max_cameras=5):
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

cameras = list_cameras()

if not cameras:
    print("No cameras detected!")
    exit()

print("\nAvailable cameras:")
for i, cam_id in enumerate(cameras):
    print(f"[{i}] Camera index {cam_id}")

# Let user choose a camera
choice = int(input("Select camera (enter number): "))
if choice < 0 or choice >= len(cameras):
    print("Invalid choice!")
    exit()

camera_index = cameras[choice]

# Buka webcam
cap = cv2.VideoCapture(camera_index)

print("\nReal-time Face Detection dimulai!")
print("Tekan 'q' untuk keluar\n")

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Face detection
    results = model(frame, conf=0.5, verbose=False)
    
    # Depict detection results
    annotated_frame = results[0].plot()
    
    # Show frame
    cv2.imshow('Face Detection', annotated_frame)
    
    # press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Selesai!")