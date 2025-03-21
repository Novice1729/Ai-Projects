import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import pickle
from keras_facenet import FaceNet
from datetime import datetime
import csv
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib

# Liveness detection setup
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\\Users\\ADMIN\\Desktop\\sumago\\Face_Recognition\\shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Load pre-trained models
with open('C:\\Users\\ADMIN\\Desktop\\sumago\\Face_Recognition\\face_classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

with open('C:\\Users\\ADMIN\\Desktop\\sumago\\Face_Recognition\\encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

model = FaceNet()
face_detect = cv2.CascadeClassifier('C:\\Users\\ADMIN\\Desktop\\sumago\\Face_Recognition\\Images\\haarcascade_frontalface_default.xml')

# Attendance logging
attendance_log = {}

def log_attendance(name, in_time, out_time=None):
    with open('attendance_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if out_time:
            writer.writerow([name, in_time, out_time, datetime.now().date()])
        else:
            writer.writerow([name, in_time, '', datetime.now().date()])

def mark_in_time(name):
    if name not in attendance_log:
        in_time = datetime.now().strftime('%H:%M:%S')
        attendance_log[name] = in_time
        log_attendance(name, in_time)

def mark_out_time(name):
    if name in attendance_log:
        out_time = datetime.now().strftime('%H:%M:%S')
        log_attendance(name, attendance_log[name], out_time)
        del attendance_log[name]

# GUI Application
class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Attendance Logger")
        self.root.geometry("800x600")

        self.video_frame = tk.Label(root)
        self.video_frame.pack()

        self.start_button = tk.Button(root, text="Start", command=self.start_camera)
        self.start_button.pack()

        self.stop_button = tk.Button(root, text="Stop", command=self.stop_camera)
        self.stop_button.pack()

        self.video_capture = cv2.VideoCapture(0)
        self.is_camera_on = False

    def start_camera(self):
        self.is_camera_on = True
        self.show_frame()

    def stop_camera(self):
        self.is_camera_on = False

    def show_frame(self):
        if self.is_camera_on:
            ret, frame = self.video_capture.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_detect.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    face_img = frame[y:y+h, x:x+w]
                    if face_img.shape != (160, 160, 3):
                        face_img = cv2.resize(face_img, (160, 160))

                    # Liveness detection
                    liveness_detected = self.detect_liveness(frame)
                    if liveness_detected:
                        try:
                            embedding = model.embeddings([face_img])[0]
                            predictions = classifier.predict_proba([embedding])
                            best_idx = np.argmax(predictions[0])
                            confidence = predictions[0][best_idx]

                            if confidence > 0.7:  # Confidence threshold
                                name = encoder.inverse_transform([best_idx])[0]
                                mark_in_time(name)
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (10, 250, 10), 2)
                                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 250, 10), 2)
                        except Exception as e:
                            print(f"Error: {e}")
                    else:
                        cv2.putText(frame, "Liveness check failed", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_frame.imgtk = imgtk
                self.video_frame.configure(image=imgtk)
                self.video_frame.after(10, self.show_frame)
            else:
                self.stop_camera()

    def detect_liveness(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            if ear < 0.2:  # Threshold for blink detection
                return True
        return False

    def __del__(self):
        if self.video_capture.isOpened():
            self.video_capture.release()
        for name in list(attendance_log.keys()):
            mark_out_time(name)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceApp(root)
    root.mainloop()