import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import yaml
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class FaceDetectionTrainer:
    def __init__(self, data_path='dataset/', output_path='runs'):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Initialize model
        self.model = None
        self.trained_model_path = None

    def create_yolo_dataset_config(self):
        """Create YOLO dataset configuration file"""
        # Use absolute paths for YOLO
        train_path = (self.data_path / 'WIDER_train' / 'images').resolve()
        val_path = (self.data_path / 'WIDER_val' / 'images').resolve()
        
        config = {
            'train': str(train_path),
            'val': str(val_path),
            'nc': 1,  # number of classes (face)
            'names': ['face']  # class names
        }

        config_path = self.output_path / 'face_dataset.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"Dataset configuration saved to: {config_path}")
        print(f"Train images: {train_path}")
        print(f"Val images: {val_path}")
        return config_path

    def convert_wider_to_yolo(self, split='train'):
        """Convert WIDER Face annotations to YOLO format"""
        print(f"Converting {split} annotations to YOLO format...")

        if split == 'train':
            images_dir = self.data_path / 'WIDER_train' / 'images'
            bbx_file = self.data_path / 'wider_face_split' / 'wider_face_train_bbx_gt.txt'
        else:
            images_dir = self.data_path / 'WIDER_val' / 'images'
            bbx_file = self.data_path / 'wider_face_split' / 'wider_face_val_bbx_gt.txt'

        # Create labels directory under dataset split to match YOLO's expected structure
        labels_root = self.data_path / ('WIDER_train' if split == 'train' else 'WIDER_val')
        labels_dir = labels_root / 'labels'
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Read bounding box annotations
        with open(bbx_file, 'r') as f:
            lines = f.readlines()

        current_file = None
        bbox_data = []
        expected_boxes = None
        boxes_read = 0
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            i += 1
            
            if not line:
                continue

            if line.endswith('.jpg'):
                # Save previous file's annotations
                if current_file is not None:
                    self._save_yolo_annotation(current_file, bbox_data, images_dir, labels_dir)

                current_file = line
                bbox_data = []
                boxes_read = 0
                
                # Next line should be the number of boxes
                if i < len(lines):
                    num_line = lines[i].strip()
                    i += 1
                    try:
                        expected_boxes = int(num_line)
                        if expected_boxes == 0:
                            # Create an empty label file for images with no faces
                            self._save_yolo_annotation(current_file, [], images_dir, labels_dir)
                            current_file = None
                    except ValueError:
                        expected_boxes = None
                        
            elif current_file is not None and expected_boxes is not None and boxes_read < expected_boxes:
                # Parse bounding box data lines (WIDER Face: x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose)
                parts = line.split()
                if len(parts) >= 10:
                    try:
                        x1, y1, w, h = map(int, parts[:4])
                        invalid = int(parts[7])
                        if invalid != 1:  # skip invalid boxes
                            bbox_data.append((x1, y1, w, h))
                        boxes_read += 1
                    except Exception:
                        boxes_read += 1
                        continue

        # Save last file's annotations
        if current_file is not None:
            self._save_yolo_annotation(current_file, bbox_data, images_dir, labels_dir)

        print(f"Converted {split} annotations to {len(list(labels_dir.glob('**/*.txt')))} label files")

    def _save_yolo_annotation(self, image_file, bbox_data, images_dir, labels_dir):
        """Save bounding box data in YOLO format"""
        image_path = images_dir / image_file
        if not image_path.exists():
            return

        # Get image dimensions
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except:
            return

        # Create label file path mirroring image subdirectories to match YOLO expected structure
        rel_img_path = Path(image_file)
        target_dir = labels_dir / rel_img_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        label_file = target_dir / (rel_img_path.stem + '.txt')

        with open(label_file, 'w') as f:
            for x1, y1, w, h in bbox_data:
                # Convert to YOLO format (class_id, center_x, center_y, width, height - normalized)
                center_x = (x1 + w/2) / img_width
                center_y = (y1 + h/2) / img_height
                norm_width = w / img_width
                norm_height = h / img_height

                # Validate coordinates
                if 0 <= center_x <= 1 and 0 <= center_y <= 1 and 0 < norm_width <= 1 and 0 < norm_height <= 1:
                    f.write(f"0 {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")

    def train_model(self, epochs=10, batch_size=8, img_size=640, pretrained=True):
        """Train YOLOv8 model for face detection"""
        print("Starting YOLOv8 face detection training...")

        # Create dataset configuration
        config_path = self.create_yolo_dataset_config()

        # Initialize YOLOv8 model
        if pretrained:
            # self.model = YOLO('yolov8n.pt')  # nano model for faster training
            self.model = YOLO('yolov8s.pt')
        else:
            self.model = YOLO('yolov8n.yaml')

        # Training arguments
        train_args = {
            'data': str(config_path),
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'project': str(self.output_path),
            'name': 'face_detection',
            'exist_ok': True,
            'patience': 5,  # early stopping
            'save': True,
            'save_period': 5,
            'plots': True,  # Generate plots otomatis
            # 'verbose': True,  # Show detailed progress
            'save_json': True,  # Save metrics ke JSON
            'cos_lr': True,  # Cosine learning rate schedule
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'workers': 1,  # Reduced to avoid memory issues
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 5,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,  # box loss weight
            'cls': 0.5,  # cls loss weight
            'dfl': 1.5,  # dfl loss weight
            'degrees': 0.0,
            'dropout': 0.0,
            'val': True,
        }

        # Train the model
        results = self.model.train(**train_args)

        # Save the best model path (it's saved in the project/name/weights directory)
        self.trained_model_path = self.output_path / 'face_detection' / 'weights' / 'best.pt'

        print(f"Training completed. Best model saved at: {self.trained_model_path}")
        return results

    def detect_faces(self, image_path, conf_threshold=0.5, save_results=True):
        """Detect faces in an image"""
        if self.model is None:
            print("No trained model found. Please train the model first.")
            return None

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not load image: {image_path}")
            return None

        # Perform detection
        results = self.model(image, conf=conf_threshold)

        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())

                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'class': class_id
                    })

                    # Draw bounding box on image
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(image, f"Face: {confidence:.2f}",
                              (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if save_results:
            # Save result image
            result_path = self.output_path / 'detections' / Path(image_path).name
            result_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(result_path), image)
            print(f"Detection result saved to: {result_path}")

        return detections, image

    def evaluate_model(self, conf_threshold=0.5):
        """Evaluate model on validation set"""
        if self.model is None:
            print("No trained model found. Please train the model first.")
            return None

        print("Evaluating model on validation set...")

        # Get validation images
        val_images_dir = self.data_path / 'WIDER_val' / 'images'
        if not val_images_dir.exists():
            print(f"Validation images directory not found: {val_images_dir}")
            return None

        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        val_images = []
        for ext in image_extensions:
            val_images.extend(list(val_images_dir.rglob(ext)))

        print(f"Found {len(val_images)} validation images")

        results = {
            'total_images': len(val_images),
            'total_detections': 0,
            'confidences': [],
            'image_results': []
        }

        # Process each image
        for img_path in val_images[:100]:  # Limit for demo purposes
            detections, _ = self.detect_faces(img_path, conf_threshold, save_results=False)

            image_result = {
                'image_path': str(img_path),
                'num_detections': len(detections) if detections else 0,
                'detections': detections
            }

            results['image_results'].append(image_result)

            if detections:
                results['total_detections'] += len(detections)
                results['confidences'].extend([d['confidence'] for d in detections])

        # Calculate statistics
        if results['confidences']:
            results['avg_confidence'] = np.mean(results['confidences'])
            results['max_confidence'] = np.max(results['confidences'])
            results['min_confidence'] = np.min(results['confidences'])
        else:
            results['avg_confidence'] = 0
            results['max_confidence'] = 0
            results['min_confidence'] = 0

        print(f"Evaluation completed. Total detections: {results['total_detections']}")
        print(f"Average confidence: {results['avg_confidence']:.3f}")

        return results

def main():
    """Main function to run face detection training and evaluation"""
    print("YOLOv8 Face Detection System")
    print("=" * 40)

    # Initialize trainer
    trainer = FaceDetectionTrainer()

    # Convert annotations to YOLO format
    print("\n1. Converting annotations to YOLO format...")
    # trainer.convert_wider_to_yolo('train')
    # trainer.convert_wider_to_yolo('val')

    # Train the model
    print("\n2. Training the model...")
    results = trainer.train_model(epochs=10, batch_size=8)  # Reduced for memory efficiency

    # Evaluate the model
    print("\n3. Evaluating the model...")
    eval_results = trainer.evaluate_model()

    print("\n4. Training and evaluation completed!")
    print(f"Results saved in: {trainer.output_path}")

    return trainer, results, eval_results
    # return trainer, eval_results

if __name__ == "__main__":
    trainer, train_results, eval_results = main()
    # trainer, eval_results = main()