import cv2
from mouse import Mouse
from interface import Interface
from face_recognizer import FaceRecognizer
from anti_turtle import AntiTurtle

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
        self.exit_on = False