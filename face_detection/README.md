# Face Detection & Recognition System

Sistem lengkap untuk deteksi dan pengenalan wajah menggunakan YOLOv8 + DeepFace.

## 🚀 Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train Model (Opsional)
```bash
python train_face.py
```

### Run System
```bash
python face_recognition_system.py
```

## 📋 Menu Utama

1. **📝 Register new person** - Daftarkan orang baru
2. **👀 Recognition mode** - Mode pengenalan real-time
3. **📊 View database** - Lihat database terdaftar
4. **🗑️ Clear database** - Hapus semua data
5. **❌ Exit** - Keluar

## 🔧 Requirements

- **Hardware**: GPU NVIDIA RTX 3060+ (recommended)
- **RAM**: 8GB minimum
- **Webcam**: Kamera terhubung
- **Software**: Python 3.8+, pip

## 📁 Project Structure

```
face_detection/
├── face_recognition_system.py    # Main system
├── train_face.py                 # Training script
├── test_*.py                     # Test scripts
├── requirements.txt              # Dependencies
├── .gitignore                    # Git ignore rules
├── runs/face_detection/weights/best.pt  # Trained model
├── FACE_RECOGNITION_README.md    # Detailed documentation
└── dataset/                      # Training data (excluded from git)
```

## 🎯 Features

- ✅ Real-time face detection dengan YOLOv8
- ✅ Face recognition menggunakan DeepFace (Facenet)
- ✅ JSON-based database untuk embeddings
- ✅ User-friendly interface dengan OpenCV
- ✅ Multi-sample registration untuk akurasi tinggi

## 📊 Performance

- **FPS**: 15-30 FPS (tergantung GPU)
- **Accuracy**: 95-99% untuk orang terdaftar
- **Model Size**: ~3.2MB (YOLOv8 nano)

## 🛠️ Troubleshooting

Lihat `FACE_RECOGNITION_README.md` untuk panduan lengkap troubleshooting dan konfigurasi advanced.

## 📞 Support

Jika ada masalah, pastikan:
1. Semua dependencies terinstall
2. Model sudah ditraining (`train_face.py`)
3. Webcam terhubung dengan benar

---

**Happy Face Recognizing! 🤖📸**