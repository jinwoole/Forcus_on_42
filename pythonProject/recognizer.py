import cv2
import dlib
import face_recognition
import numpy as np
import pickle
import os
import sys
import time

sys.path.insert(0, "face_recognition_package")
sys.path.insert(0, "face_recognition_models_package")

face_detector = dlib.get_frontal_face_detector()

data_folder = getattr(sys, "_MEIPASS", os.path.abspath(os.path.dirname(__file__)))
shape_predictor_path = os.path.join(data_folder, "shape_predictor_68_face_landmarks.dat")
face_recognition_model_path = os.path.join(data_folder, "dlib_face_recognition_resnet_model_v1.dat")

shape_predictor = dlib.shape_predictor(shape_predictor_path)
face_recognition_model = dlib.face_recognition_model_v1(face_recognition_model_path)

encodings_path = "face_encodings.pkl"
if os.path.exists(encodings_path):
    with open(encodings_path, "rb") as f:
        face_encodings = pickle.load(f)
else:
    face_encodings = {}

video_capture = cv2.VideoCapture(0)

...
saved_name = None

while True:
    time.sleep(1)
    ret, frame = video_capture.read()

    if not ret or frame is None:
        print("Error: Unable to capture a frame from the webcam.")
        break

    frame_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

    face_rectangles = face_detector(rgb_frame_small, 1)

    if len(face_rectangles) > 0:
        face_rectangle = face_rectangles[0]
        shape = shape_predictor(rgb_frame_small, face_rectangle)
        face_encoding = face_recognition_model.compute_face_descriptor(rgb_frame_small, shape, num_jitters=15)
        face_encoding = np.array(face_encoding)

        if saved_name is None:
            name = "Unknown Person"
        else:
            name = saved_name

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("g") and saved_name is None:
            saved_name = input("Enter the person's name: ")
            face_encodings[saved_name] = face_encoding

            with open(encodings_path, "wb") as f:
                pickle.dump(face_encodings, f)

            # Sending string "a" to runner.py
            sys.stdout.write("a")
            sys.stdout.flush()

video_capture.release()
cv2.destroyAllWindows()
