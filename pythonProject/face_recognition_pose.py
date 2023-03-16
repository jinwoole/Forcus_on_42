import cv2
import dlib
import face_recognition
import numpy as np
import pickle
import os
import sys
import mediapipe as mp
import math
import atexit #얼굴 정보 삭제를 위함
import signal
import time #time.sleep() -> 대기 및
import pyautogui #화면잠금 하려고 한건데, 클러스터 맥에선 권한문제때문에 displaysleep이 한계일듯
import subprocess

# pose 정보를 얻기 위한 단위 : 카메라와 유저 사이의 거리 변수
KNOWN_DISTANCE = 0.5

# pose 정보를 얻기 위한 단위 : 얼굴 너비
KNOWN_FACE_WIDTH = 0.15

# 미니rt 해봤으면 알 것
FOCAL_LENGTH = 640

#안면인식모델 정확도
REC_JITTER = 10;

#안면인식 판단 정확도 -> 낮을수록 엄격
STRICT_RATIO = 0.4;

#안면인식 데이터 저장되는 곳
encodings_path = "face_encodings.pkl"

#거북목 변수
DISTANCE_THRESHOLD = 30  # in centimeters
ROLL_THRESHOLD = 30       # in degrees
VERTICAL_THRESHOLD = 200   # in pixels

def delete_face_encodings():
    if os.path.exists(encodings_path):
        os.remove(encodings_path)
        print("face_encodings.pkl has been deleted.")
    else:
        print("face_encodings.pkl not found.")

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
        with open(encodings_path, "wb") as f:
            pickle.dump(face_encodings, f)
        print("face_encodings.pkl has been created.")

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

def draw_face_pose_information(image, results, mp_drawing, mp_face_mesh, vertical_distance=None):  # Modify this line
    distance = None
    roll = None

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

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
            vertical_distance = y[5] * image.shape[0]

            distance = (KNOWN_FACE_WIDTH * FOCAL_LENGTH) / (2 * (right_eye[0] - left_eye[0]))
            roll = math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])

            cv2.putText(image, f"Distance: {distance:.2f} cm", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Roll: {roll * 180 / math.pi:.2f} degrees", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Vertical Distance: {vertical_distance:.2f} pixels", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Add this line

    return image, distance, roll, vertical_distance

def main():
    if len(sys.argv) != 3:
        print("Usage: python face_recognition_pose.py <lock_flag> <turtle_flag>")
        sys.exit(1)
    def signal_handler(sig, frame):
        delete_face_encodings()
        sys.exit(0)

    atexit.register(delete_face_encodings)
    signal.signal(signal.SIGINT, signal_handler)


    lock_flag = sys.argv[1].lower() == "true"
    turtle_flag = sys.argv[2].lower() == "true"

    face_detector, shape_predictor, face_recognition_model, face_encodings = initialize_face_recognition()
    face_mesh, mp_drawing, mp_face_mesh = initialize_face_pose_estimation()

    video_capture = cv2.VideoCapture(0)

    init_mode = True

    init_user_not_detected_counter = 0  #락스크린
    while True:
        if not init_mode:
            time.sleep(1)
            if init_user_not_detected_counter >= 5:
                os.system("pmset displaysleepnow") #이 친구는 될것이지만, 디스플레이만 꺼준다. 해봐야 알것같다.
                init_user_not_detected_counter = 0
        ret, frame = video_capture.read()

        if not ret or frame is None:
            print("Error: Unable to capture a frame from the webcam.")
            break

        if init_mode:
            cv2.putText(frame, "Press G when you are ready", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        frame_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

        if lock_flag:
            face_rectangles, face_names = process_face_recognition(face_detector, shape_predictor,
                                                                   face_recognition_model, face_encodings,
                                                                   rgb_frame_small)
            frame = draw_face_rectangles_and_names(frame, face_rectangles,
                                                                      face_names)

            if not init_mode:
                if init_recognition_id in face_names:
                    if face_names == "init_user": #락스크린
                        init_user_not_detected_counter = 0
                else:
                    init_user_not_detected_counter += 1

        if turtle_flag:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = process_face_pose_estimation(face_mesh, image)
            image, distance, roll, vertical_distance = draw_face_pose_information(image, results, mp_drawing, mp_face_mesh)
            #거북목 판별
            if not init_mode:
                if not init_mode:
                    if distance is not None and abs(distance - init_distance) > DISTANCE_THRESHOLD:
                        os.system(
                            "osascript -e 'display notification \"Alert! Distance changed.\" with title \"Face Pose Changed\"'")

                    if roll is not None and abs(roll - init_roll) * 180 / math.pi > ROLL_THRESHOLD:
                        os.system(
                            "osascript -e 'display notification \"Alert! Roll changed.\" with title \"Face Pose Changed\"'")

                    if vertical_distance is not None and abs(vertical_distance - init_vertical) > VERTICAL_THRESHOLD:
                        os.system(
                            "osascript -e 'display notification \"Alert! Vertical distance changed.\" with title \"Face Pose Changed\"'")

        else:
            image = frame

        cv2.imshow('Combined Video Stream', image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("g") and init_mode:
            if lock_flag and not turtle_flag:
                if len(face_rectangles) == 1:
                    shape = shape_predictor(rgb_frame_small, face_rectangles[0])
                    face_encoding = face_recognition_model.compute_face_descriptor(rgb_frame_small, shape)
                    face_encoding = np.array(face_encoding)
                    name = "init_user"
                    face_encodings[name] = face_encoding

                    init_recognition_id = name

                    print(f"init_recognition_id: {init_recognition_id}")
                    init_mode = False
                else:
                    print("Error: Unable to store init_user, make sure there's only one person in front of the camera")
            elif not lock_flag and turtle_flag:
                if results.multi_face_landmarks and len(results.multi_face_landmarks) == 1:
                    init_distance = distance
                    init_roll = roll
                    init_vertical = vertical_distance

                    init_mode = False
                    print(f"init_distance: {init_distance}, init_roll: {init_roll}, init_vert: {init_vertical}")
                else:
                    print("Error: Unable to store init_distance and init_roll, make sure there's only one person in front of the camera")
            elif lock_flag and turtle_flag:
                if len(face_rectangles) == 1 and results.multi_face_landmarks and len(results.multi_face_landmarks) == 1:
                    shape = shape_predictor(rgb_frame_small, face_rectangles[0])
                    face_encoding = face_recognition_model.compute_face_descriptor(rgb_frame_small, shape)
                    face_encoding = np.array(face_encoding)
                    name = "init_user"
                    face_encodings[name] = face_encoding

                    init_distance = distance
                    init_roll = roll
                    init_vertical = vertical_distance
                    init_recognition_id = name

                    init_mode = False
                    print(f"init_recognition_id: {init_recognition_id}, init_distance: {init_distance}, init_roll: {init_roll}, init_vert: {init_vertical}")
                else:
                    print("!Error: Unable to store init_user, init_distance, and init_roll, make sure there's only one person in front of the camera")
                    #아무것도 감지되지 않는 상태에서 G눌렀을때 여기서 에러 띄워야 함 물론 저 위쪽도
            else:
                print("Error: Invalid configuration. Please set at least one of lock_flag or turtle_flag to True.")

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
