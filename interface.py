import cv2
from checkbox import Checkbox
from button import Button

class Interface:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 400
        self.height = 400
        self.auto_lock_check_box = Checkbox(x + 30, y + 80, "Auto Screen-Lock")
        self.anti_turtle_check_box = Checkbox(x + 30, y + 130, "Anti Turtle")
        self.start_button = Button(x + 30, y + 180, "START")
        self.exit_button = Button(x + 220, y + 180, "EXIT")
        self.background_color = (255, 255, 255)
    
    def draw(self, image):
        cv2.rectangle(image, (self.x, self.y), (self.x + self.width, self.y + self.height), self.background_color, -1, cv2.LINE_AA)
        self.auto_lock_check_box.draw(image)
        self.anti_turtle_check_box.draw(image)
        self.start_button.draw(image)
        self.exit_button.draw(image)