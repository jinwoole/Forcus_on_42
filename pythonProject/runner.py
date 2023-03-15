import subprocess

def main():
    lock_flag = True
    turtle_flag = True

    if not lock_flag and not turtle_flag:
        print("Error: Both lock_flag and turtle_flag are False. At least one should be True.")
        return

    cmd = ["python", "face_recognition_pose.py", str(lock_flag), str(turtle_flag)]
    process = subprocess.Popen(cmd)

    # Wait for the process to complete
    process.wait()

if __name__ == "__main__":
    main()
