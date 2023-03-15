import cv2
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Set the known distance between two facial landmarks in meters
KNOWN_DISTANCE = 0.5

# Set the known size of the face in meters
KNOWN_FACE_WIDTH = 0.15

# Set the focal length of the camera in pixels
FOCAL_LENGTH = 640

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_center = (frame_width // 2, frame_height // 2)

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Failed to capture video stream.")
      break

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        # Draw facial landmarks
        mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

        # Calculate face pose
        x = []
        y = []
        z = []
        for landmark in face_landmarks.landmark:
          x.append(landmark.x)
          y.append(landmark.y)
          z.append(landmark.z)

        nose_tip = (x[5], y[5], z[5])
        left_eye = ((x[33] + x[133])/2, (y[33] + y[133])/2, (z[33] + z[133])/2)
        right_eye = ((x[362] + x[263])/2, (y[362] + y[263])/2, (z[362] + z[263])/2)

        # Calculate face distance
        distance = (KNOWN_FACE_WIDTH * FOCAL_LENGTH) / (2 * (right_eye[0] - left_eye[0]))

        # Calculate face angles
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        dz = right_eye[2] - left_eye[2]
        roll = math.atan2(dy, dx)
        pitch = math.atan2(dz, math.sqrt(dx ** 2 + dy ** 2))
        yaw = math.atan2(-1 * (nose_tip[1] - (left_eye[1] + right_eye[1])/2), (nose_tip[0] - (left_eye[0] + right_eye[0])/2))

        # Calculate offsets
        roll_offset = math.atan2(frame_center[1] - (left_eye[1] + right_eye[1]) / 2, frame_center[0] - (left_eye[0] + right_eye[0]) / 2)
        pitch_offset = math.atan2(0, dx)
        yaw_offset = math.atan2(-1 * (frame_center[1] - (left_eye[1] + right_eye[1]) / 2), (frame_center[0] - (left_eye[0] + right_eye[0]) / 2))
        # Subtract offsets from face angles
        roll -= roll_offset
        pitch -= pitch_offset
        yaw -= yaw_offset

        # Draw face pose information
        cv2.putText(image, f"Distance: {distance:.2f} cm", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Yaw: {yaw * 180 / math.pi:.2f} degrees", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)
        cv2.putText(image, f"Pitch: {pitch * 180 / math.pi:.2f} degrees", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)
        cv2.putText(image, f"Roll: {roll * 180 / math.pi:.2f} degrees", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

      cv2.imshow('Face Pose Estimation', image)
      if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()