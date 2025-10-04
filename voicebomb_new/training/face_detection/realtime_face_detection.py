"""
Real-time Face Detection using Webcam
Menggunakan model YOLOv8 yang sudah dilatih untuk mendeteksi wajah secara real-time
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import time

class RealtimeFaceDetector:
    def __init__(self, model_path='training/face_detection/runs/face_detection/weights/best.pt', 
                 conf_threshold=0.5):
        """
        Initialize real-time face detector
        
        Args:
            model_path: Path ke model YOLOv8 yang sudah dilatih
            conf_threshold: Confidence threshold untuk deteksi (0-1)
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.model = None
        self.cap = None
        
        # Statistics
        self.fps = 0
        self.frame_count = 0
        self.total_faces = 0
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv8 model"""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model tidak ditemukan di: {self.model_path}\n"
                "Silakan jalankan face.py terlebih dahulu untuk melatih model."
            )
        
        print(f"Loading model dari: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        print("✅ Model berhasil dimuat!")
    
    def _draw_detections(self, frame, results):
        """
        Draw bounding boxes dan informasi pada frame
        
        Args:
            frame: Frame dari webcam
            results: Hasil deteksi dari YOLO
            
        Returns:
            frame: Frame dengan bounding boxes
            num_faces: Jumlah wajah yang terdeteksi
        """
        num_faces = 0
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    # Convert to integers
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label with confidence
                    label = f'Face {confidence:.2f}'
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    
                    # Draw label background
                    cv2.rectangle(frame, 
                                (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), 
                                (0, 255, 0), 
                                -1)
                    
                    # Draw label text
                    cv2.putText(frame, label, 
                              (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.6, 
                              (0, 0, 0), 
                              2)
                    
                    num_faces += 1
        
        return frame, num_faces
    
    def _draw_info(self, frame, num_faces, fps):
        """
        Draw informasi statistik pada frame
        
        Args:
            frame: Frame dari webcam
            num_faces: Jumlah wajah yang terdeteksi
            fps: Frame per second
        """
        # Background untuk info
        cv2.rectangle(frame, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 120), (0, 255, 0), 2)
        
        # Info text
        info_text = [
            f"FPS: {fps:.1f}",
            f"Wajah Terdeteksi: {num_faces}",
            f"Confidence: {self.conf_threshold:.2f}",
            "Tekan 'q' untuk keluar"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(frame, text, 
                       (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, 
                       (0, 255, 0), 
                       1)
            y_offset += 25
    
    def start(self, camera_index=3, width=1280, height=720):
        """
        Mulai deteksi wajah real-time
        
        Args:
            camera_index: Index kamera (0 untuk kamera default)
            width: Lebar frame
            height: Tinggi frame
        """
        print("\n" + "="*50)
        print("REAL-TIME FACE DETECTION")
        print("="*50)
        print(f"Membuka kamera {camera_index}...")
        
        # Open webcam
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Tidak dapat membuka kamera {camera_index}!\n"
                "Pastikan kamera terhubung dan tidak digunakan aplikasi lain."
            )
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        print("✅ Kamera berhasil dibuka!")
        print(f"Resolusi: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print("\nTekan 'q' untuk keluar")
        print("Tekan '+' untuk meningkatkan confidence threshold")
        print("Tekan '-' untuk menurunkan confidence threshold")
        print("="*50 + "\n")
        
        # FPS calculation
        prev_time = time.time()
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Gagal membaca frame dari kamera!")
                    break
                
                # Run detection
                results = self.model(frame, conf=self.conf_threshold, verbose=False)
                
                # Draw detections
                frame, num_faces = self._draw_detections(frame, results)
                
                # Calculate FPS
                current_time = time.time()
                fps = 1 / (current_time - prev_time)
                prev_time = current_time
                
                # Update statistics
                self.frame_count += 1
                self.total_faces += num_faces
                
                # Draw info
                self._draw_info(frame, num_faces, fps)
                
                # Show frame
                cv2.imshow('Real-time Face Detection - YOLOv8', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nMenghentikan deteksi...")
                    break
                elif key == ord('+') or key == ord('='):
                    self.conf_threshold = min(0.95, self.conf_threshold + 0.05)
                    print(f"Confidence threshold: {self.conf_threshold:.2f}")
                elif key == ord('-') or key == ord('_'):
                    self.conf_threshold = max(0.1, self.conf_threshold - 0.05)
                    print(f"Confidence threshold: {self.conf_threshold:.2f}")
        
        except KeyboardInterrupt:
            print("\n\nDeteksi dihentikan oleh user (Ctrl+C)")
        
        finally:
            # Cleanup
            self._cleanup()
            
            # Print statistics
            self._print_statistics()
    
    def _cleanup(self):
        """Release resources"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Resources berhasil dibersihkan")
    
    def _print_statistics(self):
        """Print statistik deteksi"""
        print("\n" + "="*50)
        print("STATISTIK DETEKSI")
        print("="*50)
        print(f"Total frame diproses: {self.frame_count}")
        print(f"Total wajah terdeteksi: {self.total_faces}")
        if self.frame_count > 0:
            print(f"Rata-rata wajah per frame: {self.total_faces/self.frame_count:.2f}")
        print("="*50)


def main():
    """Main function"""
    try:
        # Initialize detector
        detector = RealtimeFaceDetector(
            model_path='training/face_detection/runs/face_detection/weights/best.pt',
            conf_threshold=0.5
        )
        
        # Start detection
        detector.start(camera_index=3, width=1280, height=720)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
    except RuntimeError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Error tidak terduga: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()