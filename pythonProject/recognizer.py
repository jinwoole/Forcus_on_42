import cv2
import dlib
import face_recognition
import numpy as np
import pickle
import os
import sys

sys.path.insert(0, "face_recognition_package")
sys.path.insert(0, "face_recognition_models_package")
# Set up face detector, face recognizer, and shape predictor
face_detector = dlib.get_frontal_face_detector()

# Update the paths to the .dat files
data_folder = getattr(sys, "_MEIPASS", os.path.abspath(os.path.dirname(__file__)))
shape_predictor_path = os.path.join(data_folder, "shape_predictor_5_face_landmarks.dat")
face_recognition_model_path = os.path.join(data_folder, "dlib_face_recognition_resnet_model_v1.dat")

shape_predictor = dlib.shape_predictor(shape_predictor_path)
face_recognition_model = dlib.face_recognition_model_v1(face_recognition_model_path)

# Load or create face encodings dictionary
encodings_path = "face_encodings.pkl"
if os.path.exists(encodings_path):
    with open(encodings_path, "rb") as f:
        face_encodings = pickle.load(f)
else:
    face_encodings = {}

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    if not ret or frame is None:
        print("Error: Unable to capture a frame from the webcam.")
        break

    frame_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_rectangles = face_detector(rgb_frame_small, 1)
    face_names = []

    for face_rectangle in face_rectangles:
        shape = shape_predictor(rgb_frame_small, face_rectangle)
        face_encoding = face_recognition_model.compute_face_descriptor(rgb_frame_small, shape)
        face_encoding = np.array(face_encoding)

        matches = face_recognition.compare_faces([enc for _, enc in face_encodings.items()], face_encoding,
                                                 tolerance=0.6)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = list(face_encodings.keys())[first_match_index]

        face_names.append(name)

    # Display face boxes and names
    for face_rectangle, name in zip(face_rectangles, face_names):
        left, top, right, bottom = face_rectangle.left(), face_rectangle.top(), face_rectangle.right(), face_rectangle.bottom()
        left *= 4
        right *= 4
        top *= 4
        bottom *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a background rectangle for the text
        text_width, text_height = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]
        cv2.rectangle(frame, (left, top - text_height - 10), (left + text_width + 12, top), (0, 0, 255), -1)

        # Write the name on the frame
        cv2.putText(frame, name, (left + 6, top - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Video", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("g"):
        if len(face_rectangles) == 1:
            shape = shape_predictor(rgb_frame_small, face_rectangles[0])
            face_encoding = face_recognition_model.compute_face_descriptor(rgb_frame_small, shape)
            face_encoding = np.array(face_encoding)
            name = input("Enter the person's name: ")
            face_encodings[name] = face_encoding

            with open(encodings_path, "wb") as f:
                pickle.dump(face_encodings, f)

video_capture.release()
cv2.destroyAllWindows()
