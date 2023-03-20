import os
import sys
import subprocess
import pystray
from PIL import Image
from filelock import FileLock, Timeout
import threading

lock_file = 'tray.lock'
icon_path = "data/icon.png"
encodings_file = "face_encodings.pkl"

process = None

def run_main():
    global process
    process = subprocess.Popen([sys.executable, 'main.py'])

def on_select_start(icon, item):
    global process
    if process is None or process.poll() is not None:
        main_thread = threading.Thread(target=run_main)
        main_thread.start()

def on_select_stop(icon, item):
    global process
    if process and process.poll() is None:
        process.terminate()
        process = None

def on_select_exit(icon, item):
    global process
    if process:
        process.terminate()
    if os.path.exists(encodings_file):
        os.remove(encodings_file)
    icon.stop()

def create_menu_items():
    start_item = pystray.MenuItem('Start', on_select_start)
    stop_item = pystray.MenuItem('Stop', on_select_stop)
    exit_item = pystray.MenuItem('Exit', on_select_exit)
    return (start_item, stop_item, exit_item)

def create_tray_icon():
    image = Image.open(icon_path)
    icon = pystray.Icon("tray", image, "Tray", create_menu_items())
    on_select_start(icon, create_menu_items()[0])  # 일단 열면 실행해야함
    icon.run()

def main():
    with FileLock(lock_file, timeout=0):
        create_tray_icon()

if __name__ == "__main__":
    try:
        main()
    except Timeout:
        print("나쁜 녀석! 두개 돌리지 마!")