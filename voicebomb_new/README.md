# VoiceBomb

A comprehensive AI/ML project featuring voice processing and computer vision capabilities.

## 📁 Project Structure

```
VoiceBomb/
├── training/
│   └── face_detection/          # Face Detection & Recognition System
│       ├── face_recognition_system.py    # Main face recognition system
│       ├── train_face.py                 # Model training script
│       ├── realtime_*.py                 # Real-time detection scripts
│       ├── test_*.py                     # Testing scripts
│       ├── runs/face_result/weights/best.pt  # Trained YOLOv8 model
│       ├── requirements.txt              # Python dependencies
│       └── README.md                     # Face detection documentation
└── README.md                             # This file
```

## 🎯 Features

### Face Detection & Recognition System
- **Face Detection**: YOLOv8-powered real-time face detection
- **Face Recognition**: DeepFace with Facenet embeddings
- **Real-time Processing**: Webcam interface with live recognition
- **Database Management**: JSON-based face embeddings storage
- **Multi-person Support**: Register and recognize multiple individuals

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Webcam (for real-time features)
- NVIDIA GPU (recommended for better performance)

### Installation
```bash
# Clone the repository
git clone https://github.com/Gaben69181/VoiceBomb.git
cd VoiceBomb

# Install dependencies
pip install -r training/face_detection/requirements.txt
```

### Run Face Recognition System
```bash
cd training/face_detection
python face_recognition_system.py
```

## 📖 Documentation

- [Face Detection System Guide](training/face_detection/README.md)
- [Detailed Face Recognition Documentation](training/face_detection/FACE_RECOGNITION_README.md)

## 🛠️ Technologies Used

- **Computer Vision**: OpenCV, YOLOv8 (Ultralytics)
- **Face Recognition**: DeepFace, TensorFlow
- **Machine Learning**: PyTorch
- **Data Processing**: NumPy

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

Project Link: [https://github.com/Gaben69181/VoiceBomb](https://github.com/Gaben69181/VoiceBomb)

---

**Happy AI Building! 🤖✨**