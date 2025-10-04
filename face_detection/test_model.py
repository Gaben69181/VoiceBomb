"""Quick test to verify the trained model works"""
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

# Load the trained model
model_path = Path('training/face_detection/runs/face_detection/weights/best.pt')

if model_path.exists():
    print(f"Loading model from: {model_path}")
    model = YOLO(str(model_path))
    
    # Test on a sample validation image
    val_images = list(Path('dataset/WIDER_val/images').rglob('*.jpg'))
    if val_images:
        test_image = val_images[0]
        print(f"\nTesting on: {test_image}")
        
        # Run inference
        results = model(str(test_image), conf=0.5)
        
        # Print results
        for result in results:
            if result.boxes is not None:
                print(f"Detected {len(result.boxes)} faces")
                for box in result.boxes:
                    conf = box.conf.cpu().numpy()[0]
                    print(f"  - Confidence: {conf:.3f}")
            else:
                print("No faces detected")
        
        print("\n✅ Model loaded and tested successfully!")
    else:
        print("No validation images found")
else:
    print(f"❌ Model not found at: {model_path}")
    print("Please run face.py first to train the model")