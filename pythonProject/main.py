# import cv2
# import mediapipe as mp
# import math
# import subprocess

# import dlib
# import face_recognition
# import numpy as np
# import pickle
# import os
# import sys
# import time

# TITLE = "42focus"

# # Set the known distance between two facial landmarks in meters
# KNOWN_DISTANCE = 0.5
# # Set the known size of the face in meters
# KNOWN_FACE_WIDTH = 0.15
# # Set the focal length of the camera in pixels
# FOCAL_LENGTH = 640

# def anti_turtle_setting(cap):
#   anti_turtle_data = {}
#   anti_turtle_data["mp_drawing"] = mp.solutions.drawing_utils
#   anti_turtle_data["mp_face_mesh"] = mp.solutions.face_mesh
#   anti_turtle_data["image_width"] = int(cap.get(cv2.CAP_PROP_image_WIDTH))
#   anti_turtle_data["image_height"] = int(cap.get(cv2.CAP_PROP_image_HEIGHT))
#   anti_turtle_data["image_center"] = (anti_turtle_data["image_width"] // 2, anti_turtle_data["image_height"] // 2)
#   anti_turtle_data["face_mesh"] = anti_turtle_data["mp_face_mesh"].FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
#   return anti_turtle_data

# def anti_turtle(image, data):
#   image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
#   results = data["face_mesh"].process(image)
#   if results.multi_face_landmarks:
#     for face_landmarks in results.multi_face_landmarks:
#       # Draw facial landmarks
#       data["mp_drawing"].draw_landmarks(image, face_landmarks, data["mp_face_mesh"].FACEMESH_TESSELATION)

#       # Calculate face pose
#       x = []
#       y = []
#       z = []
#       for landmark in face_landmarks.landmark:
#         x.append(landmark.x)
#         y.append(landmark.y)
#         z.append(landmark.z)

#       nose_tip = (x[5], y[5], z[5])
#       left_eye = ((x[33] + x[133])/2, (y[33] + y[133])/2, (z[33] + z[133])/2)
#       right_eye = ((x[362] + x[263])/2, (y[362] + y[263])/2, (z[362] + z[263])/2)

#       # Calculate face distance
#       distance = (KNOWN_FACE_WIDTH * FOCAL_LENGTH) / (2 * (right_eye[0] - left_eye[0]))

#       # Calculate face angles
#       dx = right_eye[0] - left_eye[0]
#       dy = right_eye[1] - left_eye[1]
#       dz = right_eye[2] - left_eye[2]
#       # roll = math.atan2(dy, dx)
#       # pitch = math.atan2(dz, math.sqrt(dx ** 2 + dy ** 2))
#       yaw = math.atan2(-1 * (nose_tip[1] - (left_eye[1] + right_eye[1])/2), (nose_tip[0] - (left_eye[0] + right_eye[0])/2))

#       # Calculate offsets
#       # roll_offset = math.atan2(image_center[1] - (left_eye[1] + right_eye[1]) / 2, image_center[0] - (left_eye[0] + right_eye[0]) / 2)
#       # pitch_offset = math.atan2(0, dx)
#       yaw_offset = math.atan2(-1 * (data["image_center"][1] - (left_eye[1] + right_eye[1]) / 2), (data["image_center"][0] - (left_eye[0] + right_eye[0]) / 2))
#       # Subtract offsets from face angles
#       # roll -= roll_offset
#       # pitch -= pitch_offset
#       yaw -= yaw_offset

#       # Draw face pose information
#       cv2.putText(image, f"Distance: {distance:.2f} cm", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#       cv2.putText(image, f"Yaw: {yaw * 180 / math.pi:.2f} degrees", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1,
#                   (0, 255, 0), 2)
#       # cv2.putText(image, f"Pitch: {pitch * 180 / math.pi:.2f} degrees", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1,
#       #             (0, 255, 0), 2)
#       # cv2.putText(image, f"Roll: {roll * 180 / math.pi:.2f} degrees", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1,
#       #             (0, 255, 0), 2)

#     # cv2.imshow(TITLE, image)
#     return image

# def auto_lock_setting():
#   auto_lock_data = {}
#   sys.path.insert(0, "face_recognition_package")
#   sys.path.insert(0, "face_recognition_models_package")
#   auto_lock_data["face_detector"] = dlib.get_frontal_face_detector()
#   data_folder = getattr(sys, "_MEIPASS", os.path.abspath(os.path.dirname(__file__)))
#   shape_predictor_path = os.path.join(data_folder, "shape_predictor_68_face_landmarks.dat")
#   face_recognition_model_path = os.path.join(data_folder, "dlib_face_recognition_resnet_model_v1.dat")
#   auto_lock_data["shape_predictor"] = dlib.shape_predictor(shape_predictor_path)
#   auto_lock_data["face_recognition_model"] = dlib.face_recognition_model_v1(face_recognition_model_path)
#   encodings_path = "face_encodings.pkl"
#   if os.path.exists(encodings_path):
#       with open(encodings_path, "rb") as f:
#           auto_lock_data["face_encodings"] = pickle.load(f)
#   else:
#       auto_lock_data["face_encodings"] = {}
#   auto_lock_data["saved_name"] = None
#   return auto_lock_data

# def auto_lock(image, data):
#   image_small = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
#   rgb_image_small = cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)
#   face_names = []
#   face_rectangles = data["face_detector"](rgb_image_small, 1)

#   for face_rectangle in face_rectangles:
#     shape = data["shape_predictor"](rgb_image_small, face_rectangle)
#     face_encoding = data["face_recognition_model"].compute_face_descriptor(rgb_image_small, shape, num_jitters=10)
#     face_encoding = np.array(face_encoding)

#     if data["face_encodings"]:
#         face_distances = face_recognition.face_distance([enc for _, enc in data["face_encodings"].items()], face_encoding)

#         if len(face_distances) > 0:
#             min_distance_index = np.argmin(face_distances)
#             min_distance = face_distances[min_distance_index]

#             if min_distance < 0.4:
#                 name = list(data["face_encodings"].keys())[min_distance_index]
#             else:
#                 name = "Unknown Person"
#         else:
#             name = "Unknown Person"
#     else:
#         name = "Unknown Person"

#     face_names.append(name)

#   for face_rectangle, name in zip(face_rectangles, face_names):
#     left, top, right, bottom = face_rectangle.left(), face_rectangle.top(), face_rectangle.right(), face_rectangle.bottom()
#     left *= 4
#     right *= 4
#     top *= 4
#     bottom *= 4

#     if name == "Unknown":
#         rectangle_color = (255, 0, 0)
#     else:
#         rectangle_color = (0, 0, 255)

#     cv2.rectangle(image, (left, top), (right, bottom), rectangle_color, 2)

#     text_width, text_height = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]
#     cv2.rectangle(image, (left, top - text_height - 10), (left + text_width + 12, top), rectangle_color, -1)

#     cv2.putText(image, name, (left + 6, top - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

#     # Display face size information
#     face_width = right - left
#     face_height = bottom - top
#     size_text = f"Size: {face_width}x{face_height}"
#     size_width, size_height = cv2.getTextSize(size_text, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]
#     cv2.putText(image, size_text, (left, bottom + size_height + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255),
#                 1)

#   # cv2.imshow(TITLE, image)
#   return image

#     # key = cv2.waitKey(1) & 0xFF
#     # if key == ord("q"):
#     #     break
#     # elif key == ord("g"):
#     #     if len(face_rectangles) == 1:
#     #         shape = shape_predictor(rgb_image_small, face_rectangles[0])
#     #         face_encoding = face_recognition_model.compute_face_descriptor(rgb_image_small, shape)
#     #         face_encoding = np.array(face_encoding)
#     #         name = input("Enter the person's name: ")
#     #         face_encodings[name] = face_encoding

#   # if len(face_rectangles) > 0:
#   #     face_rectangle = face_rectangles[0]
      
#   #     face_names = []
      
#   #     shape = data["shape_predictor"](rgb_image_small, face_rectangle)
#   #     # face_encoding = data["face_recognition_model"].compute_face_descriptor(rgb_image_small, shape, num_jitters=15)
#   #     face_encoding = data["face_recognition_model"].compute_face_descriptor(rgb_image_small, shape)
#   #     face_encoding = np.array(face_encoding)

#   #     if data["saved_name"] is None:
#   #         name = "Unknown Person"
#   #     else:
#   #         name = data["saved_name"]

#       # key = cv2.waitKey(1) & 0xFF
#       # if key == ord("q"):
#       #     break
#       # elif key == ord("g") and saved_name is None:
#       #     saved_name = input("Enter the person's name: ")
#       #     face_encodings[saved_name] = face_encoding

#       #     with open(encodings_path, "wb") as f:
#       #         pickle.dump(face_encodings, f)

#       #     # Sending string "a" to runner.py
#       #     sys.stdout.write("a")
#       #     sys.stdout.flush()

# def lock_screen():
#   result = subprocess.run(['ls'], stdout=subprocess.PIPE)

# def main():
#   print("start")

#   anti_turtle_toggle = True
#   auto_lock_toggle = True

#   cap = cv2.VideoCapture(0)
#   anti_turtle_data = anti_turtle_setting(cap)
#   auto_lock_data = auto_lock_setting()

#   while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#       print("Failed to capture video stream.")
#       break
    
#     if auto_lock_toggle:
#       image = auto_lock(image, auto_lock_data)
#     if anti_turtle_toggle:
#       image = anti_turtle(image, anti_turtle_data)
    
#     cv2.imshow(TITLE, image)
    
#     if cv2.waitKey(5) & 0xFF == 27:
#       print("exit")
#       break
#     # time.sleep(1)

#   cap.release()
#   cv2.destroyAllWindows()

# main()

import cv2
import dlib
import face_recognition
import numpy as np
import pickle
import os
import sys
import mediapipe as mp
import math

# Set the known distance between two facial landmarks in meters
KNOWN_DISTANCE = 0.5

# Set the known size of the face in meters
KNOWN_FACE_WIDTH = 0.15

# Set the focal length of the camera in pixels
FOCAL_LENGTH = 640


def initialize_face_recognition():
    face_detector = dlib.get_frontal_face_detector()
    shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
    face_recognition_model_path = "dlib_face_recognition_resnet_model_v1.dat"

    shape_predictor = dlib.shape_predictor(shape_predictor_path)
    face_recognition_model = dlib.face_recognition_model_v1(face_recognition_model_path)

    encodings_path = "face_encodings.pkl"
    if os.path.exists(encodings_path):
        with open(encodings_path, "rb") as f:
            face_encodings = pickle.load(f)
    else:
        face_encodings = {}

    return face_detector, shape_predictor, face_recognition_model, face_encodings


def process_face_recognition(face_detector, shape_predictor, face_recognition_model, face_encodings, rgb_image_small):
    face_rectangles = face_detector(rgb_image_small, 1)
    face_names = []

    for face_rectangle in face_rectangles:
        shape = shape_predictor(rgb_image_small, face_rectangle)
        face_encoding = face_recognition_model.compute_face_descriptor(rgb_image_small, shape, num_jitters=10)
        face_encoding = np.array(face_encoding)

        if face_encodings:
            face_distances = face_recognition.face_distance([enc for _, enc in face_encodings.items()], face_encoding)

            if len(face_distances) > 0:
                min_distance_index = np.argmin(face_distances)
                min_distance = face_distances[min_distance_index]

                if min_distance < 0.4:
                    name = list(face_encodings.keys())[min_distance_index]
                else:
                    name = "Unknown Person"
            else:
                name = "Unknown Person"
        else:
            name = "Unknown Person"

        face_names.append(name)

    return face_rectangles, face_names

def draw_face_rectangles_and_names(image, face_rectangles, face_names):
    for face_rectangle, name in zip(face_rectangles, face_names):
        left, top, right, bottom = face_rectangle.left(), face_rectangle.top(), face_rectangle.right(), face_rectangle.bottom()
        left *= 4
        right *= 4
        top *= 4
        bottom *= 4

        if name == "Unknown":
            rectangle_color = (0, 0, 255)
        else:
            rectangle_color = (255, 0, 0)

        cv2.rectangle(image, (left, top), (right, bottom), rectangle_color, 2)

        text_width, text_height = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]
        cv2.rectangle(image, (left, top - text_height - 10), (left + text_width + 12, top), rectangle_color, -1)

        cv2.putText(image, name, (left + 6, top - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    return image

def initialize_face_pose_estimation():
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    return face_mesh, mp_drawing, mp_face_mesh

def process_face_pose_estimation(face_mesh, image):
    results = face_mesh.process(image)
    return results

def draw_face_pose_information(image, results, mp_drawing, mp_face_mesh):
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

            # Calculate face angles
            x = []
            y = []
            z = []
            for landmark in face_landmarks.landmark:
                x.append(landmark.x)
                y.append(landmark.y)
                z.append(landmark.z)

            nose_tip = (x[5], y[5], z[5])
            left_eye = ((x[33] + x[133]) / 2, (y[33] + y[133]) / 2, (z[33] + z[133]) / 2)
            right_eye = ((x[362] + x[263]) / 2, (y[362] + y[263]) / 2, (z[362] + z[263]) / 2)

            # Calculate face distance
            distance = (KNOWN_FACE_WIDTH * FOCAL_LENGTH) / (2 * (right_eye[0] - left_eye[0]))

            # Calculate yaw
            roll = math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])

            # Draw face pose information
            cv2.putText(image, f"Distance: {distance:.2f} cm", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Roll: {roll * 180 / math.pi:.2f} degrees", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

    return image

def init_app():
    print("START 42FOCUS!")
    while True:
      user_name = input("User name: ")
      if user_name != "":
          break
      
    while True:
      auto_lock_toggle = input("Auto Screen-Lock On (y/n): ")
      if auto_lock_toggle == "y" or auto_lock_toggle == "n":
          break
    
    while True:
      anti_turtle_toggle = input("Anti-Turtle On (y/n): ")
      if anti_turtle_toggle == "y" or anti_turtle_toggle == "n":
          break

    if auto_lock_toggle == "n" and anti_turtle_toggle == "n":
        print("Ok, Bye...")
        exit(0)
    
    return user_name, auto_lock_toggle == "y", anti_turtle_toggle == "y"

def main():
    user_name, auto_lock_toggle, anti_turtle_toggle = init_app()
    
    face_detector, shape_predictor, face_recognition_model, face_encodings = initialize_face_recognition()
    face_mesh, mp_drawing, mp_face_mesh = initialize_face_pose_estimation()

    video_capture = cv2.VideoCapture(0)

    while True:
        success, image = video_capture.read()
        
        if not success or image is None:
            print("Error: Unable to capture a image from the webcam.")
            break

        if auto_lock_toggle is True:
          image_small = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
          rgb_image_small = cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)
          
          face_rectangles, face_names = process_face_recognition(face_detector, shape_predictor, face_recognition_model, face_encodings, rgb_image_small)
          image = draw_face_rectangles_and_names(image, face_rectangles, face_names)

        if anti_turtle_toggle is True:
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          results = process_face_pose_estimation(face_mesh, image)
          image = draw_face_pose_information(image, results, mp_drawing, mp_face_mesh)
        
        # cv2.imshow('42focus', image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        # elif key == ord("g"):
        #     if len(face_rectangles) == 1:
        #         shape = shape_predictor(rgb_image_small, face_rectangles[0])
        #         face_encoding = face_recognition_model.compute_face_descriptor(rgb_image_small, shape)
        #         face_encoding = np.array(face_encoding)
        #         name = input("Enter the person's name: ")
        #         face_encodings[name] = face_encoding

        #         with open(encodings_path, "wb") as f:
        #             pickle.dump(face_encodings, f)

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()