import cv2
import dlib
import pickle
import os
import numpy as np
import face_recognition
from constants import Constants

class FaceRecognizer:
    def __init__(self):
        self.delete_face_encodings()
        self.face_detector, self.shape_predictor, self.face_recognition_model, self.face_encodings = self.initialize_face_recognition()
        self.rgb_image_small = None
        self.face_rectangles = []
        self.face_names = []
        self.user_not_detected_counter = 0
        self.user_face_rect_pos = (0, 0, 0, 0)
        
    def initialize_face_recognition(self):
        face_detector = dlib.get_frontal_face_detector()
        shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
        face_recognition_model_path = "dlib_face_recognition_resnet_model_v1.dat"

        shape_predictor = dlib.shape_predictor(shape_predictor_path)
        face_recognition_model = dlib.face_recognition_model_v1(face_recognition_model_path)

        encodings_path = Constants.FACE_SAVE_PATH

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
        self.face_names = []
        self.face_rectangles = self.face_detector(self.rgb_image_small, 1)
        for face_rectangle in self.face_rectangles:
            shape = self.shape_predictor(self.rgb_image_small, face_rectangle)
            face_encoding = self.face_recognition_model.compute_face_descriptor(self.rgb_image_small, shape, num_jitters=Constants.REC_JITTER)
            face_encoding = np.array(face_encoding)
            name = "Unknown"
            if self.face_encodings:
                face_distances = face_recognition.face_distance([enc for _, enc in self.face_encodings.items()], face_encoding)
                if len(face_distances) > 0:
                    min_distance_index = np.argmin(face_distances)
                    min_distance = face_distances[min_distance_index]
                    if min_distance < Constants.STRICT_RATIO:
                        name = list(self.face_encodings.keys())[min_distance_index]
            self.face_names.append(name)
    
    def draw_face_rectangles_and_names(self, image, step):
        self.user_face_rect_pos = (0, 0, 0, 0)
        if step == 2:
            self.user_not_detected_counter += 1
        for face_rectangle, name in zip(self.face_rectangles, self.face_names):
            left, top, right, bottom = face_rectangle.left() * 4, face_rectangle.top() * 4, face_rectangle.right() * 4, face_rectangle.bottom() * 4
            if name == "You":
                rectangle_color = (0, 255, 0)
                text_width, text_height = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]
                cv2.rectangle(image, (left, top - text_height - 10), (left + text_width + 12, top), rectangle_color, -1)
                cv2.putText(image, name, (left + 6, top - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                cv2.rectangle(image, (left, top), (right, bottom), rectangle_color, 2)
                self.user_face_rect_pos = (left, top, right, bottom)
                self.user_not_detected_counter = 0
            else:
                rectangle_color = (0, 255, 255)
                cv2.rectangle(image, (left, top), (right, bottom), rectangle_color, 2)

    def draw_face_rect(self, image, mouse, step):
        image_small = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        self.rgb_image_small = cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)

        if step == 0:
            self.face_rectangles = self.face_detector(self.rgb_image_small, 1)
            for face_rectangle in self.face_rectangles:
                left, top, right, bottom = face_rectangle.left() * 4, face_rectangle.top() * 4, face_rectangle.right() * 4, face_rectangle.bottom() * 4
                rectangle_color = (0, 255, 255)
                thickness = 2
                if left <= mouse.x <= right and top <= mouse.y <= bottom:
                    rectangle_color = (0, 255, 0)
                    thickness = 3
                cv2.rectangle(image, (left, top), (right, bottom), rectangle_color, thickness)
        elif step == 1:
            self.process_face_recognition()
            self.draw_face_rectangles_and_names(image, step)
        elif step == 2:
            self.process_face_recognition()
            self.draw_face_rectangles_and_names(image, step)

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
                return True
        return False
            
    def delete_face_encodings(self):
        if os.path.exists(Constants.FACE_SAVE_PATH):
            os.remove(Constants.FACE_SAVE_PATH)
        else:
            print("face_encodings.pkl not found.")

    def check_is_user_face(self):
        if self.user_not_detected_counter >= Constants.USER_NOT_DETECTED_COUNT_MAX:
            os.system("pmset displaysleepnow")
            self.user_not_detected_counter = 0