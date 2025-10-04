"""Test script to verify notebook will work"""
import os
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

print("Testing Notebook Components")
print("=" * 50)

# 1. Check if model exists
model_path = 'training/face_detection/runs/face_detection/weights/best.pt'
print(f"\n1. Checking model: {model_path}")
if os.path.exists(model_path):
    model = YOLO(model_path)
    print(f"   ✅ Model loaded successfully")
else:
    print(f"   ❌ Model not found")

# 2. Check if results.csv exists
results_path = 'training/face_detection/runs/face_detection/results.csv'
print(f"\n2. Checking results: {results_path}")
if os.path.exists(results_path):
    results_df = pd.read_csv(results_path)
    print(f"   ✅ Results loaded: {results_df.shape[0]} epochs")
    print(f"   Columns: {list(results_df.columns)[:5]}...")
    
    # Print final metrics
    print(f"\n   Final Training Metrics:")
    print(f"   - Best mAP@0.5: {results_df['metrics/mAP50(B)'].max():.4f}")
    print(f"   - Best mAP@0.5:0.95: {results_df['metrics/mAP50-95(B)'].max():.4f}")
    print(f"   - Best Precision: {results_df['metrics/precision(B)'].max():.4f}")
    print(f"   - Best Recall: {results_df['metrics/recall(B)'].max():.4f}")
else:
    print(f"   ❌ Results not found")

# 3. Check validation images
val_images_dir = Path('dataset/WIDER_val/images')
print(f"\n3. Checking validation images: {val_images_dir}")
if val_images_dir.exists():
    val_images = list(val_images_dir.rglob('*.jpg'))
    print(f"   ✅ Found {len(val_images)} validation images")
else:
    print(f"   ❌ Validation images not found")