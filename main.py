import cv2
import dlib
import face_recognition
import numpy as np
import pickle
import os
import sys
import mediapipe as mp
import math
import atexit # 얼굴 정보 삭제를 위함
import signal
import time
import subprocess

# Set the known distance between two facial landmarks in meters
KNOWN_DISTANCE = 0.5

# Set the known size of the face in meters
KNOWN_FACE_WIDTH = 0.15

# Set the focal length of the camera in pixels
FOCAL_LENGTH = 640

# 안면인식모델 정확도
REC_JITTER = 10

# 안면인식 판단 정확도 -> 낮을수록 엄격
STRICT_RATIO = 0.4

# 안면인식 데이터 저장되는 곳
FACE_SAVE_PATH = "face_encodings.pkl"

# 거북목 변수
DISTANCE_THRESHOLD = 30  # in centimeters
ROLL_THRESHOLD = 30       # in degrees
VERTICAL_THRESHOLD = 200   # in pixels


class Mouse:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.click_event = False

class Button:
    def __init__(self, x, y, content):
        self.x = x
        self.y = y
        self.content = content
        self.width = 150
        self.height = 50
        self.mouse_on = False
        self.clicked = False
        self.button_color = (0, 0, 0)
        self.text_color = (255, 255, 255)
        self.button_color_hover = (0, 255, 0)
        self.text_color_hover = (255, 255, 255)

    def draw(self, image):
        cv2.rectangle(
            image,
            (self.x, self.y),
            (self.x + self.width, self.y + self.height),
            self.button_color,
            -1)
        cv2.putText(image, self.content, (self.x + 30, self.y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, self.text_color, 2)

    def hover_event(self, mouse_x, mouse_y):
        if self.x <= mouse_x <= self.x + self.width and self.y <= mouse_y <= self.y + self.height:
            self.button_color = (0, 255, 0)
            self.text_color = (255, 255, 255)
        else:
            self.button_color = (0, 0, 0)
            self.text_color = (255, 255, 255)

class Checkbox:
    def __init__(self, x, y, content):
        self.x = x
        self.y = y
        self.content = content
        self.width = 350
        self.height = 50
        self.mouse_on = False
        self.clicked = False
        self.box_width = 30
        self.box_height = 30
    
    def draw(self, image):
        text_color = (0, 0, 0)
        fill = 2
        if self.clicked is True:
            fill = -1

        cv2.putText(image, self.content, (self.x, self.y), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv2.rectangle(
            image,
            (self.x + self.width - self.box_width, self.y - self.box_height),
            (self.x + self.width, self.y),
            text_color,
            fill)
        
    def click_event(self, mouse_x, mouse_y):
        if self.x + self.width - self.box_width <= mouse_x <= self.x + self.width and self.y - self.height <= mouse_y <= self.y:
            if self.clicked is True:
                self.clicked = False
            else:
                self.clicked = True

class Interface:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 400
        self.height = 400
        self.auto_lock_check_box = Checkbox(x + 30, y + 80, "Auto Screen-Lock")
        self.anti_turtle_check_box = Checkbox(x + 30, y + 130, "Anti Turtle")
        self.start_button = Button(x + 120, y + 180, "Start")
        self.background_color = (255, 255, 255)
    
    def draw(self, image):
        cv2.rectangle(image, (self.x, self.y), (self.x + self.width, self.y + self.height), self.background_color, -1, cv2.LINE_AA)
        self.auto_lock_check_box.draw(image)
        self.anti_turtle_check_box.draw(image)
        self.start_button.draw(image)

class FaceRecognizer:
    def __init__(self):
        self.delete_face_encodings()
        self.face_detector, self.shape_predictor, self.face_recognition_model, self.face_encodings = self.initialize_face_recognition()
        self.rgb_image_small = None
        self.face_rectangles = []
        self.face_names = []
        self.user_not_detected_counter = 0
        

    def initialize_face_recognition(self):
        face_detector = dlib.get_frontal_face_detector()
        shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
        face_recognition_model_path = "dlib_face_recognition_resnet_model_v1.dat"

        shape_predictor = dlib.shape_predictor(shape_predictor_path)
        face_recognition_model = dlib.face_recognition_model_v1(face_recognition_model_path)

        encodings_path = FACE_SAVE_PATH

        if os.path.exists(encodings_path):
            with open(encodings_path, "rb") as f:
                face_encodings = pickle.load(f)
        else:
            face_encodings = {}
            with open(encodings_path, "wb") as f:
                pickle.dump(face_encodings, f)
            print("face_encodings.pkl has been created.")

        return face_detector, shape_predictor, face_recognition_model, face_encodings
    
    def process_face_recognition(self):
        self.face_rectangles = self.face_detector(self.rgb_image_small, 1)
        for face_rectangle in self.face_rectangles:
            shape = self.shape_predictor(self.rgb_image_small, face_rectangle)
            face_encoding = self.face_recognition_model.compute_face_descriptor(self.rgb_image_small, shape, num_jitters=10)
            face_encoding = np.array(face_encoding)

            if self.face_encodings:
                face_distances = face_recognition.face_distance([enc for _, enc in self.face_encodings.items()], face_encoding)

                if len(face_distances) > 0:
                    min_distance_index = np.argmin(face_distances)
                    min_distance = face_distances[min_distance_index]

                    if min_distance < 0.4:
                        name = list(self.face_encodings.keys())[min_distance_index]
                    else:
                        name = "Unknown"
                else:
                    name = "Unknown"
            else:
                name = "Unknown"

            self.face_names.append(name)
    
    def draw_face_rectangles_and_names(self, image):
        for face_rectangle, name in zip(self.face_rectangles, self.face_names):
            left, top, right, bottom = face_rectangle.left(), face_rectangle.top(), face_rectangle.right(), face_rectangle.bottom()
            left *= 4
            right *= 4
            top *= 4
            bottom *= 4
            if name == "Unknown":
                rectangle_color = (0, 255, 255)
            else:
                rectangle_color = (0, 255, 0)
                text_width, text_height = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]
                cv2.rectangle(image, (left, top - text_height - 10), (left + text_width + 12, top), rectangle_color, -1)
                cv2.putText(image, name, (left + 6, top - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.rectangle(image, (left, top), (right, bottom), rectangle_color, 2)

    def draw_face_rect(self, image, mouse, step):
        image_small = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        self.rgb_image_small = cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)

        if step == 0:
            self.face_rectangles = self.face_detector(self.rgb_image_small, 1)
            for idx, face_rectangle in enumerate(self.face_rectangles):
                left, top, right, bottom = face_rectangle.left(), face_rectangle.top(), face_rectangle.right(), face_rectangle.bottom()
                left *= 4
                right *= 4
                top *= 4
                bottom *= 4
                rectangle_color = (0, 255, 255)
                thickness = 2
                if left <= mouse.x <= right and top <= mouse.y <= bottom:
                    rectangle_color = (0, 255, 0)
                    thickness = 3
                cv2.rectangle(image, (left, top), (right, bottom), rectangle_color, thickness)
        elif step == 1:
            self.process_face_recognition()
            self.draw_face_rectangles_and_names(image)

    def rect_click_event(self, mouse_x, mouse_y):
        for face_rectangle in self.face_rectangles:
            left, top, right, bottom = face_rectangle.left(), face_rectangle.top(), face_rectangle.right(), face_rectangle.bottom()
            left *= 4
            right *= 4
            top *= 4
            bottom *= 4
            if left <= mouse_x <= right and top <= mouse_y <= bottom:
                shape = self.shape_predictor(self.rgb_image_small, face_rectangle)
                face_encoding = self.face_recognition_model.compute_face_descriptor(self.rgb_image_small, shape)
                face_encoding = np.array(face_encoding)
                name = "You"
                self.face_encodings[name] = face_encoding
                print("detecting you...")                
                return True
        return False
            
    def delete_face_encodings(self):
        if os.path.exists(FACE_SAVE_PATH):
            os.remove(FACE_SAVE_PATH)
            print("face_encodings.pkl has been deleted.")
        else:
            print("face_encodings.pkl not found.")

class AntiTurtle:
    def __init__(self):
        self.face_mesh, self.mp_drawing, self.mp_face_mesh = self.initialize_face_pose_estimation()
        self.distance = None
        self.roll = None
        self.vertical_distance = None

    def initialize_face_pose_estimation(self):
        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        return face_mesh, mp_drawing, mp_face_mesh

    def draw_mask(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.process_face_pose_estimation(self.face_mesh, image)
        image = self.draw_face_pose_information(image, results)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image
    
    def process_face_pose_estimation(self, face_mesh, image):
        return face_mesh.process(image)
    
    def draw_face_pose_information(self, image, results):  # Modify this line
        self.distance = None
        self.roll = None
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(image, face_landmarks, self.mp_face_mesh.FACEMESH_TESSELATION)

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
                self.vertical_distance = y[5] * image.shape[0]
                self.distance = (KNOWN_FACE_WIDTH * FOCAL_LENGTH) / (2 * (right_eye[0] - left_eye[0]))
                self.roll = math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])

                # cv2.putText(image, f"Distance: {self.distance:.2f} cm", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # cv2.putText(image, f"Roll: {self.roll * 180 / math.pi:.2f} degrees", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # cv2.putText(image, f"Vertical Distance: {self.vertical_distance:.2f} pixels", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Add this line
        return image
    
class Focus:
    def __init__(self):
        self.step = 0
        self.mouse = Mouse()
        self.interface = Interface(20, 20)
        self.face_recognizer = FaceRecognizer()
        self.anti_turtle = AntiTurtle()
        self.image = None
        self.video_capture = cv2.VideoCapture(0)
        self.init_distance = None
        self.init_roll = None
        self.init_vertical = None

def mouse_event_callback(event, x, y, flags, focus):
    if focus.step == 0:
        if event == cv2.EVENT_MOUSEMOVE:
            focus.mouse.x = x
            focus.mouse.y = y
        if event == cv2.EVENT_LBUTTONDOWN:
            if focus.face_recognizer.rect_click_event(x, y) is True:
                focus.step = 1

    elif focus.step == 1:
        if event == cv2.EVENT_MOUSEMOVE:
            focus.interface.start_button.hover_event(x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            focus.interface.auto_lock_check_box.click_event(x, y)
            focus.interface.anti_turtle_check_box.click_event(x, y)

def main():
    focus = Focus()

    while True:
        success, focus.image = focus.video_capture.read()
        if not success or focus.image is None:
            break

        cv2.setMouseCallback("42focus", mouse_event_callback, focus)
        if focus.step == 0:
            focus.face_recognizer.draw_face_rect(focus.image, focus.mouse, focus.step)
        if focus.step == 1:
            # focus.interface.draw(focus.image)
            focus.image = focus.anti_turtle.draw_mask(focus.image)
            

        cv2.imshow('42focus', focus.image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    focus.video_capture.release()
    cv2.destroyAllWindows()

main()