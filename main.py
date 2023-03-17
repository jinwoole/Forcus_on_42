import cv2
import dlib

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
from focus import Focus

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
            focus.interface.draw(focus.image)
            focus.face_recognizer.draw_face_rect(focus.image, focus.mouse, focus.step)
            if focus.interface.anti_turtle_check_box.clicked:
                focus.image = focus.anti_turtle.draw_mask(focus.image)

        cv2.imshow('42focus', focus.image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    focus.video_capture.release()
    cv2.destroyAllWindows()

main()