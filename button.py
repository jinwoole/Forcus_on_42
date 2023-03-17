import cv2

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