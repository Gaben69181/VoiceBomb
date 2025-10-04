"""
Face Recognition System
Sistem lengkap untuk face detection + face recognition menggunakan YOLOv8 + DeepFace
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from ultralytics import YOLO
from deepface import DeepFace
import time
import warnings
warnings.filterwarnings('ignore')

class FaceRecognitionSystem:
    def __init__(self, detection_model_path='training/face_detection/runs/face_detection/weights/best.pt',
                 database_path='face_database.json'):
        """
        Initialize Face Recognition System

        Args:
            detection_model_path: Path ke model YOLOv8 untuk face detection
            database_path: Path untuk menyimpan database embeddings
        """
        self.detection_model_path = Path(detection_model_path)
        self.database_path = Path(database_path)
        self.detection_model = None
        self.database = {}
        self.recognition_model = "Facenet"  # Model DeepFace yang digunakan

        # Load models
        self._load_detection_model()
        self._load_database()

        print("✅ Face Recognition System initialized!")
        print(f"   Detection model: {self.detection_model_path}")
        print(f"   Recognition model: {self.recognition_model}")
        print(f"   Database: {self.database_path} ({len(self.database)} people registered)")

    def _load_detection_model(self):
        """Load YOLOv8 face detection model"""
        if not self.detection_model_path.exists():
            raise FileNotFoundError(
                f"Detection model tidak ditemukan: {self.detection_model_path}\n"
                "Jalankan training_face.py terlebih dahulu."
            )

        print("Loading face detection model...")
        self.detection_model = YOLO(str(self.detection_model_path))
        print("✅ Detection model loaded!")

    def _load_database(self):
        """Load face recognition database"""
        if self.database_path.exists():
            try:
                with open(self.database_path, 'r') as f:
                    self.database = json.load(f)
                print(f"✅ Database loaded: {len(self.database)} people")
            except Exception as e:
                print(f"⚠️ Error loading database: {e}")
                self.database = {}
        else:
            print("📝 Creating new database...")
            self.database = {}

    def _save_database(self):
        """Save database to file"""
        try:
            with open(self.database_path, 'w') as f:
                json.dump(self.database, f, indent=2)
            print("💾 Database saved!")
        except Exception as e:
            print(f"❌ Error saving database: {e}")

    def _extract_face_embedding(self, face_image):
        """
        Extract face embedding dari gambar wajah

        Args:
            face_image: numpy array gambar wajah

        Returns:
            embedding: numpy array embedding vector
        """
        try:
            # Convert BGR ke RGB
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            # Extract embedding menggunakan DeepFace
            embedding = DeepFace.represent(
                img_path=face_rgb,
                model_name=self.recognition_model,
                enforce_detection=False,
                detector_backend='skip'  # Skip detection karena sudah di-crop
            )

            if isinstance(embedding, list) and len(embedding) > 0:
                return np.array(embedding[0]['embedding'])
            else:
                return None

        except Exception as e:
            print(f"❌ Error extracting embedding: {e}")
            return None

    def _cosine_similarity(self, emb1, emb2):
        """Calculate cosine similarity between two embeddings"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def register_person(self, name, face_images):
        """
        Register new person dengan multiple face images

        Args:
            name: Nama orang
            face_images: List of face images (numpy arrays)
        """
        if not face_images:
            print("❌ No face images provided!")
            return False

        print(f"📝 Registering {name} with {len(face_images)} face samples...")

        embeddings = []
        for i, face_img in enumerate(face_images):
            print(f"   Processing sample {i+1}/{len(face_images)}...")
            embedding = self._extract_face_embedding(face_img)
            if embedding is not None:
                embeddings.append(embedding.tolist())
            else:
                print(f"   ⚠️ Failed to extract embedding from sample {i+1}")

        if not embeddings:
            print("❌ Failed to extract any embeddings!")
            return False

        # Save to database
        self.database[name.lower().replace(' ', '_')] = {
            'name': name,
            'embeddings': embeddings,
            'registered_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'samples_count': len(embeddings)
        }

        self._save_database()
        print(f"✅ {name} registered successfully with {len(embeddings)} embeddings!")
        return True

    def recognize_face(self, face_image, threshold=0.45):
        """
        Recognize face dari database

        Args:
            face_image: numpy array gambar wajah
            threshold: Similarity threshold (0-1)

        Returns:
            dict: Recognition result
        """
        embedding = self._extract_face_embedding(face_image)
        if embedding is None:
            return {'name': 'Unknown', 'confidence': 0.0, 'status': 'extraction_failed'}

        best_match = None
        best_similarity = 0.0

        # Compare dengan semua orang di database
        for person_key, person_data in self.database.items():
            for stored_embedding in person_data['embeddings']:
                stored_emb = np.array(stored_embedding)
                similarity = self._cosine_similarity(embedding, stored_emb)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = person_data

        if best_match and best_similarity >= threshold:
            return {
                'name': best_match['name'],
                'confidence': float(best_similarity),
                'status': 'recognized'
            }
        else:
            return {
                'name': 'Unknown',
                'confidence': float(best_similarity) if best_similarity > 0 else 0.0,
                'status': 'not_recognized'
            }

    def detect_and_recognize(self, frame, conf_threshold=0.5, recognition_threshold=0.4):
        """
        Detect faces and recognize them

        Args:
            frame: Webcam frame
            conf_threshold: Face detection confidence
            recognition_threshold: Face recognition threshold

        Returns:
            annotated_frame: Frame dengan bounding boxes dan nama
            results: List hasil recognition
        """
        results = []

        # Face detection dengan YOLOv8
        detections = self.detection_model(frame, conf=conf_threshold, verbose=False)

        for detection in detections:
            if detection.boxes is not None:
                for box in detection.boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    confidence = box.conf[0].cpu().numpy()

                    # Crop face dari frame
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue

                    # Recognize face
                    recognition_result = self.recognize_face(face_crop, recognition_threshold)

                    # Draw bounding box
                    if recognition_result['status'] == 'recognized':
                        color = (0, 255, 0)  # Green for recognized
                        label = f"{recognition_result['name']} ({recognition_result['confidence']:.2f})"
                    else:
                        color = (0, 0, 255)  # Red for unknown
                        label = f"Unknown ({recognition_result['confidence']:.2f})"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Draw label background
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                                (x1 + label_size[0], y1), color, -1)

                    # Draw label text
                    cv2.putText(frame, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    results.append({
                        'bbox': [x1, y1, x2, y2],
                        'detection_confidence': float(confidence),
                        'recognition': recognition_result
                    })

        return frame, results

    def run_registration_mode(self, camera_index=3):
        """
        Run registration mode untuk mendaftarkan orang baru
        """
        print("\n" + "="*60)
        print("FACE REGISTRATION MODE")
        print("="*60)
        print("Instruksi:")
        print("1. Masukkan nama orang yang akan didaftarkan")
        print("2. Posisikan wajah di depan kamera")
        print("3. Tekan 'SPACE' untuk capture face sample")
        print("4. Lakukan minimal 3-5 capture dari berbagai angle")
        print("5. Tekan 'ENTER' untuk selesai dan simpan")
        print("6. Tekan 'ESC' untuk cancel")
        print("="*60)

        # Input nama
        name = input("\nMasukkan nama orang: ").strip()
        if not name:
            print("❌ Nama tidak boleh kosong!")
            return

        # Buka kamera
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("❌ Tidak dapat membuka kamera!")
            return

        face_samples = []
        sample_count = 0

        print(f"\n📸 Mulai capture face samples untuk {name}")
        print("Tekan SPACE untuk capture, ENTER untuk selesai, ESC untuk cancel")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip frame horizontally untuk mirror effect
            frame = cv2.flip(frame, 1)

            # Detect faces untuk preview
            detections = self.detection_model(frame, conf=0.5, verbose=False)

            # Draw detection preview
            for detection in detections:
                if detection.boxes is not None:
                    for box in detection.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Draw UI
            cv2.putText(frame, f"Registering: {name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {sample_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "SPACE: Capture | ENTER: Save | ESC: Cancel", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Face Registration', frame)

            key = cv2.waitKey(1) & 0xFF

            if key == 32:  # SPACE - Capture sample
                if detections[0].boxes is not None and len(detections[0].boxes) > 0:
                    # Ambil face terbesar
                    boxes = detections[0].boxes
                    best_box = max(boxes, key=lambda x: (x.xyxy[0][2] - x.xyxy[0][0]) * (x.xyxy[0][3] - x.xyxy[0][1]))

                    x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        face_samples.append(face_crop)
                        sample_count += 1
                        print(f"✅ Sample {sample_count} captured!")
                else:
                    print("❌ Tidak ada wajah terdeteksi!")

            elif key == 13:  # ENTER - Save and finish
                if sample_count >= 3:
                    success = self.register_person(name, face_samples)
                    if success:
                        print(f"✅ {name} berhasil didaftarkan!")
                    else:
                        print(f"❌ Gagal mendaftarkan {name}")
                else:
                    print(f"❌ Minimal 3 samples diperlukan! (current: {sample_count})")
                    continue
                break

            elif key == 27:  # ESC - Cancel
                print("❌ Registration cancelled")
                break

        cap.release()
        cv2.destroyAllWindows()

    def run_recognition_mode(self, camera_index=3):
        """
        Run recognition mode untuk mengenali orang
        """
        print("\n" + "="*60)
        print("FACE RECOGNITION MODE")
        print("="*60)
        print(f"Database: {len(self.database)} orang terdaftar")
        print("Tekan 'q' untuk keluar")
        print("="*60)

        if len(self.database) == 0:
            print("❌ Database kosong! Lakukan registration terlebih dahulu.")
            return

        # Buka kamera
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("❌ Tidak dapat membuka kamera!")
            return

        fps_counter = 0
        fps_start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Flip frame untuk mirror effect
                frame = cv2.flip(frame, 1)

                # Detect and recognize faces
                annotated_frame, recognition_results = self.detect_and_recognize(frame)

                # Calculate FPS
                fps_counter += 1
                current_time = time.time()
                if current_time - fps_start_time >= 1.0:
                    fps = fps_counter / (current_time - fps_start_time)
                    fps_counter = 0
                    fps_start_time = current_time
                else:
                    fps = 0

                # Draw UI
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"People: {len(self.database)}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Faces: {len(recognition_results)}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, "Press 'q' to exit", (10, annotated_frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow('Face Recognition System', annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\n👋 Recognition stopped by user")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Recognition mode finished")

    def run_main_menu(self):
        """
        Run main menu system
        """
        while True:
            print("\n" + "="*60)
            print("🤖 FACE RECOGNITION SYSTEM")
            print("="*60)
            print(f"Database: {len(self.database)} orang terdaftar")
            print("\nMenu:")
            print("1. 📝 Register new person")
            print("2. 👀 Recognition mode")
            print("3. 📊 View database")
            print("4. 🗑️  Clear database")
            print("5. ❌ Exit")
            print("="*60)

            try:
                choice = input("Pilih menu (1-5): ").strip()

                if choice == '1':
                    self.run_registration_mode()
                elif choice == '2':
                    self.run_recognition_mode()
                elif choice == '3':
                    self._show_database()
                elif choice == '4':
                    if input("Yakin hapus semua data? (y/N): ").lower() == 'y':
                        self.database = {}
                        self._save_database()
                        print("✅ Database cleared!")
                elif choice == '5':
                    print("👋 Terima kasih!")
                    break
                else:
                    print("❌ Pilihan tidak valid!")

            except KeyboardInterrupt:
                print("\n👋 Program stopped by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def _show_database(self):
        """Show database contents"""
        if not self.database:
            print("📝 Database kosong")
            return

        print("\n" + "="*60)
        print("👥 REGISTERED PEOPLE")
        print("="*60)
        print("<10")
        print("-" * 60)

        for i, (key, data) in enumerate(self.database.items(), 1):
            print("<10")
        print("="*60)


def main():
    """Main function"""
    try:
        # Initialize system
        system = FaceRecognitionSystem()

        # Run main menu
        system.run_main_menu()

    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()