class Constants:
	TITLE = "42focus"

	KNOWN_DISTANCE = 0.5 # Set the known distance between two facial landmarks in meters
	KNOWN_FACE_WIDTH = 0.15 # Set the known size of the face in meters
	FOCAL_LENGTH = 640 # Set the focal length of the camera in pixels
	REC_JITTER = 10
	STRICT_RATIO = 0.4 # 안면인식 판단 정확도 -> 낮을수록 엄격
	FACE_SAVE_PATH = "face_encodings.pkl" # 안면인식 데이터 저장되는 곳

	SHAPE_PREDICTOR_PATH = "./data/shape_predictor_68_face_landmarks.dat"
	FACE_RECOGNITION_MODEL_PATH = "./data/dlib_face_recognition_resnet_model_v1.dat"
	
	# 거북목 판정 세팅
	DISTANCE_THRESHOLD = 70  # in centimeters
	ROLL_THRESHOLD = 30       # in degrees
	VERTICAL_THRESHOLD = 200   # in pixels

	USER_NOT_DETECTED_COUNT_MAX = 50