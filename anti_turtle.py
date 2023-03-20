import mediapipe as mp
import numpy as np
import cv2
import math
from constants import Constants
import subprocess

class AntiTurtle:
    def __init__(self):
        self.face_mesh, self.mp_drawing, self.mp_face_mesh = self.initialize_face_pose_estimation()
        self.init_distance = None
        self.init_roll = None
        self.init_vertical_distance = None
        self.distance = None
        self.roll = None
        self.vertical_distance = None

    def initialize_face_pose_estimation(self):
        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        return face_mesh, mp_drawing, mp_face_mesh
    
    def check_pose(self):
        distance_diff = 0
        roll_diff = 0
        vertical_diff = 0

        if self.distance is not None:
            distance_diff = (self.distance - self.init_distance)
            # cv2.putText(image, f"Distance diff: {distance_diff:.2f} cm", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
            #             1.1, (255, 0, 0), 2)
            print(f"Distance diff: {distance_diff:.2f} cm")
            if -1 * distance_diff > Constants.DISTANCE_THRESHOLD:
                self.display_alert()

        if self.roll is not None:
            roll_diff = abs(self.roll - self.init_roll) * 180 / math.pi
            # cv2.putText(image, f"Roll diff: {roll_diff:.2f} degrees", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
            #             (255, 0, 0), 2)
            if roll_diff > Constants.ROLL_THRESHOLD:
                self.display_alert()

        if self.vertical_distance is not None:
            vertical_diff = abs(self.vertical_distance - self.init_vertical_distance)
            # cv2.putText(image, f"Vertical diff: {vertical_diff:.2f} pixels", (20, 120),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 0), 2)
            if vertical_diff > Constants.VERTICAL_THRESHOLD:
                self.display_alert()

    def draw_mask(self, image, face_recognizer):
        # 제외할 부분의 좌표 (x1, y1, x2, y2)
        exclude_area = face_recognizer.user_face_rect_pos
        # 제외할 부분을 제외한 모든 픽셀을 검은색으로 칠한 이미지 생성
        mask = np.zeros_like(image)
        mask[exclude_area[1]:exclude_area[3], exclude_area[0]:exclude_area[2], :] = 255
        # 이미지에서 제외할 부분을 제외한 부분을 검은색으로 칠함
        edited_image = cv2.bitwise_and(image, mask)

        edited_image = cv2.cvtColor(edited_image, cv2.COLOR_BGR2RGB)
        results = self.process_face_pose_estimation(self.face_mesh, edited_image)
        image = self.draw_face_pose_information(image, results)
        
        return image
    
    def process_face_pose_estimation(self, face_mesh, image):
        return face_mesh.process(image)
    
    def draw_face_pose_information(self, image, results):
        # self.distance = None
        # self.roll = None
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
                self.distance = (Constants.KNOWN_FACE_WIDTH * Constants.FOCAL_LENGTH) / (2 * (right_eye[0] - left_eye[0]))
                self.roll = math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])

                # cv2.putText(image, f"Distance: {self.distance:.2f} cm", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # cv2.putText(image, f"Roll: {self.roll * 180 / math.pi:.2f} degrees", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # cv2.putText(image, f"Vertical Distance: {self.vertical_distance:.2f} pixels", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Add this line
        return image
    
    def set_init_data(self):
        self.init_distance = self.distance
        self.init_roll = self.roll
        self.init_vertical_distance = self.vertical_distance
    
    def display_alert(self, delay=1):
        message = "You are becoming a turtle!"
        applescript = f'display dialog "{message}" with title "Alert" buttons {{"OK"}} default button "OK" giving up after {delay}'
        subprocess.Popen(['osascript', '-e', applescript])