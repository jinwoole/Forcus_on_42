import cv2

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
        self.disable = False
    
    def draw(self, image):
        if self.disable is True:
            text_color = (100, 100, 100)
        else:
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
        if self.disable is False and self.x + self.width - self.box_width <= mouse_x <= self.x + self.width and self.y - self.height <= mouse_y <= self.y:
            if self.clicked is True:
                self.clicked = False
            else:
                self.clicked = True
    
    def set_disable(self, state):
        self.disable = state