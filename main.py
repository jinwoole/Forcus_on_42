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

gui_mode = True
auto_lock_toggle = False
anti_turtle_toggle = False

def initialize_face_recognition():
    face_detector = dlib.get_frontal_face_detector()
    shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
    face_recognition_model_path = "dlib_face_recognition_resnet_model_v1.dat"

    shape_predictor = dlib.shape_predictor(shape_predictor_path)
    face_recognition_model = dlib.face_recognition_model_v1(face_recognition_model_path)

    encodings_path = "face_encodings.pkl"
    if os.path.exists(encodings_path):
        with open(encodings_path, "rb") as f:
            face_encodings = pickle.load(f)
    else:
        face_encodings = {}

    return face_detector, shape_predictor, face_recognition_model, face_encodings


def process_face_recognition(face_detector, shape_predictor, face_recognition_model, face_encodings, rgb_image_small):
    face_rectangles = face_detector(rgb_image_small, 1)
    face_names = []

    for face_rectangle in face_rectangles:
        shape = shape_predictor(rgb_image_small, face_rectangle)
        face_encoding = face_recognition_model.compute_face_descriptor(rgb_image_small, shape, num_jitters=10)
        face_encoding = np.array(face_encoding)

        if face_encodings:
            face_distances = face_recognition.face_distance([enc for _, enc in face_encodings.items()], face_encoding)

            if len(face_distances) > 0:
                min_distance_index = np.argmin(face_distances)
                min_distance = face_distances[min_distance_index]

                if min_distance < 0.4:
                    name = list(face_encodings.keys())[min_distance_index]
                else:
                    name = "Unknown Person"
            else:
                name = "Unknown Person"
        else:
            name = "Unknown Person"

        face_names.append(name)

    return face_rectangles, face_names

def draw_face_rectangles_and_names(image, face_rectangles, face_names):
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

        cv2.rectangle(image, (left, top), (right, bottom), rectangle_color, 2)

        text_width, text_height = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]
        cv2.rectangle(image, (left, top - text_height - 10), (left + text_width + 12, top), rectangle_color, -1)

        cv2.putText(image, name, (left + 6, top - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    return image

def initialize_face_pose_estimation():
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    return face_mesh, mp_drawing, mp_face_mesh

def process_face_pose_estimation(face_mesh, image):
    results = face_mesh.process(image)
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

def mouse_event_callback(event, x, y, flags, data):
    global auto_lock_toggle
    global anti_turtle_toggle

    # 마우스 이벤트 처리 코드 작성
    if event == cv2.EVENT_LBUTTONDOWN:
        if 370 <= x and x <= 400 and 175 <= y and y <= 205:
            if auto_lock_toggle is False:
                auto_lock_toggle = True
            else:
                auto_lock_toggle = False
        if 370 <= x and x <= 400 and 225 <= y and y <= 255:
            if anti_turtle_toggle is False:
                anti_turtle_toggle = True
            else:
                anti_turtle_toggle = False

def check_box(image, content, x, y):
    global auto_lock_toggle
    global anti_turtle_toggle

    text_color = (0, 0, 0)
    fill = 2
    if content == "Auto Screen-Lock" and auto_lock_toggle is True:
        fill = -1
    if content == "Anti Turtle" and anti_turtle_toggle is True:
        fill = -1

    # anchor_point_x, anchor_point_y = 50, 200
    function_name_size, check_box_size = 350, 30
    cv2.putText(image, content, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    cv2.rectangle(
        image,
        (x + function_name_size - check_box_size, y - check_box_size // 2 - 10),
        (x + function_name_size, y + check_box_size // 2 - 10),
        text_color,
        fill)
    data = {
        "x0": x + function_name_size - check_box_size,
        "x1": x + function_name_size,
        "y0": y + check_box_size // 2 - 10,
        "y1": y - check_box_size // 2 - 10
    }
    cv2.setMouseCallback("42focus", mouse_event_callback, data)
    return image

def draw_gui(image):
    cv2.rectangle(
        image,
        (30, 150),
        (430, 380),
        (255, 255, 255),
        -1,
        cv2.LINE_AA,
        )

    image = check_box(image, "Auto Screen-Lock", 50, 200)
    image = check_box(image, "Anti Turtle", 50, 250)
    
    return image

def main():
    
    face_detector, shape_predictor, face_recognition_model, face_encodings = initialize_face_recognition()
    face_mesh, mp_drawing, mp_face_mesh = initialize_face_pose_estimation()

    video_capture = cv2.VideoCapture(0)
    

    while True:
        success, image = video_capture.read()
        if not success or image is None:
            break

        
        image = draw_gui(image)
        

        if not success or image is None:
            print("Error: Unable to capture a image from the webcam.")
            break

        if auto_lock_toggle is True:
            image_small = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
            rgb_image_small = cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)
            
            face_rectangles, face_names = process_face_recognition(face_detector, shape_predictor, face_recognition_model, face_encodings, rgb_image_small)
            image = draw_face_rectangles_and_names(image, face_rectangles, face_names)

        if anti_turtle_toggle is True:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = process_face_pose_estimation(face_mesh, image)
            image = draw_face_pose_information(image, results, mp_drawing, mp_face_mesh)
            
        cv2.imshow('42focus', image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        
        

    video_capture.release()
    cv2.destroyAllWindows()

main()