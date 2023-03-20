import cv2
from focus import Focus
import time
from constants import Constants

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
            focus.interface.exit_button.hover_event(x, y)

        if event == cv2.EVENT_LBUTTONDOWN:    
            focus.interface.auto_lock_check_box.click_event(x, y)
            focus.interface.anti_turtle_check_box.click_event(x, y)
            if focus.interface.exit_button.click_event(x, y):
                print("Exit!")
                focus.exit_on = True
            if focus.interface.start_button.click_event(x, y):
                focus.step = 2
                focus.anti_turtle.set_init_data()
                focus.interface.auto_lock_check_box.set_disable(True)
                focus.interface.anti_turtle_check_box.set_disable(True)
                focus.interface.start_button.set_disable(True)
    
    elif focus.step == 2:
        if event == cv2.EVENT_MOUSEMOVE:
            focus.interface.exit_button.hover_event(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            if focus.interface.exit_button.click_event(x, y):
                print("Exit!")
                focus.exit_on = True
            focus.interface.auto_lock_check_box.click_event(x, y)
            focus.interface.anti_turtle_check_box.click_event(x, y)
        
def main():
    focus = Focus()

    while True:
        
        success, focus.image = focus.video_capture.read()
        if not success or focus.image is None:
            break

        cv2.setMouseCallback(Constants.TITLE, mouse_event_callback, focus)

        
        focus.face_recognizer.draw_face_rect(focus.image, focus.mouse, focus.step)
            
        if focus.step == 1:
            # time.sleep(0.5)
            focus.interface.draw(focus.image)
            # focus.face_recognizer.draw_face_rect(focus.image, focus.mouse, focus.step)
            if focus.interface.anti_turtle_check_box.clicked:
                focus.image = focus.anti_turtle.draw_mask(focus.image, focus.face_recognizer)
            
        elif focus.step == 2:
            # time.sleep(0.5)
            # focus.face_recognizer.draw_face_rect(focus.image, focus.mouse, focus.step)
            focus.interface.draw(focus.image)
            if focus.interface.anti_turtle_check_box.clicked is True:
                focus.image = focus.anti_turtle.draw_mask(focus.image, focus.face_recognizer)
                focus.anti_turtle.check_pose()
            if focus.interface.auto_lock_check_box.clicked is True:
                focus.face_recognizer.check_is_user_face()
        
        cv2.imshow(Constants.TITLE, focus.image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27 or focus.exit_on is True:
            break

    focus.video_capture.release()
    cv2.destroyAllWindows()

main()