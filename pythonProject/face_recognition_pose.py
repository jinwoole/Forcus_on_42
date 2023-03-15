import cv2
import dlib
import face_recognition
import numpy as np
import pickle
import os
import sys
import mediapipe as mp
import math

# Set the known distance between two facial landmarks in meters
KNOWN_DISTANCE = 0.5

# Set the known size of the face in meters
KNOWN_FACE_WIDTH = 0.15

# Set the focal length of the camera in pixels
FOCAL_LENGTH = 640

#안면인식모델 정확도
REC_JITTER = 15;

#안면인식 판단 정확도 -> 낮을수록 엄격
STRICT_RATIO = 0.4;

#안면인식 데이터 저장되는 곳
encodings_path = "face_encodings.pkl"


def initialize_face_recognition():
    face_detector = dlib.get_frontal_face_detector()
    shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
    face_recognition_model_path = "dlib_face_recognition_resnet_model_v1.dat"

    shape_predictor = dlib.shape_predictor(shape_predictor_path)
    face_recognition_model = dlib.face_recognition_model_v1(face_recognition_model_path)

    if os.path.exists(encodings_path):
        with open(encodings_path, "rb") as f:
            face_encodings = pickle.load(f)
    else:
        face_encodings = {}

    return face_detector, shape_predictor, face_recognition_model, face_encodings


def process_face_recognition(face_detector, shape_predictor, face_recognition_model, face_encodings, rgb_frame_small):
    face_rectangles = face_detector(rgb_frame_small, 1)
    face_names = []

    for face_rectangle in face_rectangles:
        shape = shape_predictor(rgb_frame_small, face_rectangle)
        face_encoding = face_recognition_model.compute_face_descriptor(rgb_frame_small, shape, num_jitters=REC_JITTER)
        face_encoding = np.array(face_encoding)

        if face_encodings:
            face_distances = face_recognition.face_distance([enc for _, enc in face_encodings.items()], face_encoding)

            if len(face_distances) > 0:
                min_distance_index = np.argmin(face_distances)
                min_distance = face_distances[min_distance_index]

                if min_distance < STRICT_RATIO:
                    name = list(face_encodings.keys())[min_distance_index]
                else:
                    name = "Unknown Person"
            else:
                name = "Unknown Person"
        else:
            name = "Unknown Person"

        face_names.append(name)

    return face_rectangles, face_names

def draw_face_rectangles_and_names(frame, face_rectangles, face_names):
    for face_rectangle, name in zip(face_rectangles, face_names):
        left, top, right, bottom = face_rectangle.left(), face_rectangle.top(), face_rectangle.right(), face_rectangle.bottom()
        left *= 4
        right *= 4
        top *= 4
        bottom *= 4

        if name == "Unknown":
            rectangle_color = (0, 0, 255)
        else:
            rectangle_color = (255, 0, 0)

        cv2.rectangle(frame, (left, top), (right, bottom), rectangle_color, 2)

        text_width, text_height = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]
        cv2.rectangle(frame, (left, top - text_height - 10), (left + text_width + 12, top), rectangle_color, -1)

        cv2.putText(frame, name, (left + 6, top - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    return frame

def initialize_face_pose_estimation():
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    return face_mesh, mp_drawing, mp_face_mesh

def process_face_pose_estimation(face_mesh, frame):
    results = face_mesh.process(frame)
    return results

def draw_face_pose_information(image, results, mp_drawing, mp_face_mesh):
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

            # Calculate face angles
            x = []
            y = []
            z = []
            for landmark in face_landmarks.landmark:
                x.append(landmark.x)
                y.append(landmark.y)
                z.append(landmark.z)

            nose_tip = (x[5], y[5], z[5])
            left_eye = ((x[33] + x[133]) / 2, (y[33] + y[133]) / 2, (z[33] + z[133]) / 2)
            right_eye = ((x[362] + x[263]) / 2, (y[362] + y[263]) / 2, (z[362] + z[263]) / 2)

            # Calculate face distance
            distance = (KNOWN_FACE_WIDTH * FOCAL_LENGTH) / (2 * (right_eye[0] - left_eye[0]))

            # Calculate yaw
            roll = math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])

            # Draw face pose information
            cv2.putText(image, f"Distance: {distance:.2f} cm", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Roll: {roll * 180 / math.pi:.2f} degrees", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

    return image

def main():
    if len(sys.argv) != 3:
        print("Usage: python face_recognition_pose.py <lock_flag> <turtle_flag>")
        sys.exit(1)

    lock_flag = sys.argv[1].lower() == "true"
    turtle_flag = sys.argv[2].lower() == "true"

    face_detector, shape_predictor, face_recognition_model, face_encodings = initialize_face_recognition()
    face_mesh, mp_drawing, mp_face_mesh = initialize_face_pose_estimation()

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()

        if not ret or frame is None:
            print("Error: Unable to capture a frame from the webcam.")
            break

        frame_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

        if lock_flag:
            face_rectangles, face_names = process_face_recognition(face_detector, shape_predictor,
                                                                   face_recognition_model, face_encodings,
                                                                   rgb_frame_small)
            frame = draw_face_rectangles_and_names(frame, face_rectangles, face_names)

        if turtle_flag:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = process_face_pose_estimation(face_mesh, image)
            image = draw_face_pose_information(image, results, mp_drawing, mp_face_mesh)
        else:
            image = frame

        cv2.imshow('Combined Video Stream', image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("g") and lock_flag:
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



if __name__ == "__main__":
    main()
