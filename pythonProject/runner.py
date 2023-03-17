import tkinter as tk
from tkinter import ttk
import subprocess
import threading
import pystray
from PIL import Image

def run_face_recognition():
    lock_flag = lock_flag_var.get()
    turtle_flag = turtle_flag_var.get()

    if not lock_flag and not turtle_flag:
        result_string.set("Error: Both lock_flag and turtle_flag are False. At least one should be True.")
        return

    start_button.config(state="disabled")
    stop_button.config(state="normal")
    cmd = ["python", "face_recognition_pose.py", str(lock_flag), str(turtle_flag)]

    def run_script():
        global process
        process = subprocess.Popen(cmd)
        process.wait()
        result_string.set("Process completed.")
        start_button.config(state="normal")
        stop_button.config(state="disabled")

    thread = threading.Thread(target=run_script)
    thread.start()

def stop_script():
    global process
    if process:
        process.terminate()
        result_string.set("Process terminated.")
        start_button.config(state="normal")
        stop_button.config(state="disabled")

def exit_app_gui():
    if process and process.poll() is None:
        process.terminate()
    root.quit()
    status_icon.stop()

root = tk.Tk()
root.title("Face Recognition GUI")

main_frame = ttk.Frame(root, padding="10")
main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

lock_flag_var = tk.BooleanVar()
turtle_flag_var = tk.BooleanVar()
result_string = tk.StringVar()

lock_flag_checkbox = ttk.Checkbutton(main_frame, text="Lock flag", variable=lock_flag_var)
turtle_flag_checkbox = ttk.Checkbutton(main_frame, text="Turtle flag", variable=turtle_flag_var)
start_button = ttk.Button(main_frame, text="Start", command=run_face_recognition)
stop_button = ttk.Button(main_frame, text="Stop", command=stop_script, state="disabled")
exit_button = ttk.Button(main_frame, text="Exit", command=exit_app_gui)
result_label = ttk.Label(main_frame, textvariable=result_string)

lock_flag_checkbox.grid(row=0, column=0, sticky=tk.W, pady=5)
turtle_flag_checkbox.grid(row=1, column=0, sticky=tk.W, pady=5)
start_button.grid(row=2, column=0, sticky=tk.W, pady=5)
stop_button.grid(row=2, column=1, sticky=tk.W, pady=5)
exit_button.grid(row=3, column=0, columnspan=2, pady=5)
result_label.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=5)

def on_icon_click(icon, item):
    if root.state() == "withdrawn":
        root.deiconify()
        icon.stop()
    else:
        root.withdraw()
        icon.run()

def create_status_icon():
    icon = pystray.Icon("face_recognition_gui")

    image = Image.open("icon.png")  # Provide the path to your icon image
    icon.icon = image

    icon.menu = pystray.Menu(pystray.MenuItem('Open Control', on_icon_click),
                             pystray.MenuItem('Exit', exit_app))
    return icon

def on_root_close():
    root.withdraw()
    status_icon.run()

def exit_app(icon, item):
    if process and process.poll() is None:
        process.terminate()
    root.quit()
    icon.stop()

status_icon = create_status_icon()
threading.Thread(target=status_icon.run).start()
root.protocol("WM_DELETE_WINDOW", on_root_close)

root.mainloop()