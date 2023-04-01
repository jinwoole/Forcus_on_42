import os
import sys

#import packages! ha!
cwd = os.getcwd()
print(cwd)
packages_path = os.path.join(cwd, "packages")
sys.path.insert(0, packages_path)

import subprocess
import threading
import pystray
from PIL import Image

def run_face_recognition(lock_flag, turtle_flag):
    if not lock_flag and not turtle_flag:
        print("Error: Both lock_flag and turtle_flag are False. At least one should be True.")
        return

    cmd = ["python3", "face_recognition_pose.py", str(lock_flag), str(turtle_flag)]

    def run_script():
        global process
        process = subprocess.Popen(cmd)
        process.wait()
        print("Process completed.")

    thread = threading.Thread(target=run_script)
    thread.start()

def stop_script():
    global process
    if process:
        process.terminate()
        print("Process terminated.")

def delete_face_encodings_file():
    file_path = "face_encodings.pkl"
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        print("face_encodings.pkl file not found.")

def exit_app(icon, item):
    if process and process.poll() is None:
        process.terminate()
        delete_face_encodings_file()
    icon.stop()

def on_start_click(icon, lock_flag, turtle_flag):
    run_face_recognition(lock_flag, turtle_flag)

def on_stop_click(icon):
    stop_script()

def toggle_flag(flag):
    return not flag

def create_status_icon():
    icon = pystray.Icon("face_recognition_gui")

    image = Image.open("icon.png")  # Provide the path to your icon image
    icon.icon = image

    flags = {'lock_flag': True, 'turtle_flag': False}

    def on_flag_click(item, flag_name):
        flags[flag_name] = not flags[flag_name]

    icon.menu = pystray.Menu(pystray.MenuItem('Lock', lambda item: on_flag_click(item, 'lock_flag'), checked=lambda item: flags['lock_flag']),
                             pystray.MenuItem('Turtle', lambda item: on_flag_click(item, 'turtle_flag'), checked=lambda item: flags['turtle_flag']),
                             pystray.MenuItem('Start', lambda icon, item: on_start_click(icon, flags['lock_flag'], flags['turtle_flag'])),
                             pystray.MenuItem('Stop', on_stop_click),
                             pystray.MenuItem('Exit', exit_app))
    return icon

if __name__ == "__main__":
    process = None
    status_icon = create_status_icon()
    status_icon.run()
