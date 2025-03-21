# Face Recognition-Based Attendance System

This project is a **Face Recognition-Based Attendance System** that uses real-time face detection and recognition to log attendance. It includes features like liveness detection to prevent proxy attendance and a user-friendly GUI for managing attendance logs.

---

## **Project Overview**

The system consists of three main components:

1. **Adding Faces**: Captures and stores user face images for training.
2. **Training the Model**: Uses FaceNet embeddings and machine learning models (SVM, RandomForest, XGBoost) to train a face recognition model.
3. **Attendance Logging**: A GUI-based application that detects faces in real-time, verifies liveness, and logs attendance.

---

## **Features**

- **Real-Time Face Detection**: Uses Haar Cascade and FaceNet for face detection and recognition.
- **Liveness Detection**: Prevents proxy attendance by detecting live faces using eye blink detection.
- **Attendance Logging**: Logs attendance with timestamps and saves it in a CSV file.
- **User-Friendly GUI**: Built with Tkinter for easy interaction.

---

## **Installation and Setup**

### **Prerequisites**
- Python 3.x
- pip (Python package installer)

### **Steps to Run the Project**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/face-recognition-attendance.git
   cd face-recognition-attendance
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
3. **Download Pre-trained models**:
   Download Pre-trained Models:
   Download the Haar Cascade file (haarcascade_frontalface_default.xml)
   and dlib shape predictor (shape_predictor_68_face_landmarks.dat).
4. **Run the Scripts**:
   **1.Add faces 2.Train the model 3.Run the attendace logger**
   ```bash
   python add_faces.py
   python train_model_2.py
   python Attendance_logger.py.py
5.**Contributing
Contributions are welcome! If you have any suggestions or improvements, please open an issue or submit a pull request.**
