import subprocess

process = subprocess.Popen(["python", "recognizer.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

while True:
    face_name = process.stdout.readline().strip()

    if not face_name:
        break

    print(f"Face name received: {face_name}")

return_code = process.wait()
print(f"Recognizer process exited with code {return_code}")
