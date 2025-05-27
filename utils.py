import os
import cv2
import face_recognition
import numpy as np
import mediapipe as mp

# Setup MediaPipe FaceMesh for liveliness + blink
mesh_model = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Eye landmark indices for EAR
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

def get_img(path):
    return cv2.imread(path)

def find_faces_img(img):
    return face_recognition.face_locations(img)

def find_faces_frame(f):
    return face_recognition.face_locations(f)

def eye_aspect_ratio(landmarks, eye_indices):
    p = [np.array([landmarks[i].x, landmarks[i].y]) for i in eye_indices]
    vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])
    return vertical / (2.0 * horizontal)

def detect_blink_from_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mesh_model.process(rgb)
    if not results.multi_face_landmarks:
        return False
    for face_landmarks in results.multi_face_landmarks:
        left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE_LANDMARKS)
        right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE_LANDMARKS)
        avg_ear = (left_ear + right_ear) / 2.0
        if avg_ear < 0.3:
            return True
    return False

def is_real_person(full_frame, require_blink=False):
    rgb = cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB)
    try:
        res = mesh_model.process(rgb)
        if res.multi_face_landmarks:
            if not require_blink:
                return True
            cap = cv2.VideoCapture(0)
            blink_detected = False
            for _ in range(30):  # about 1 sec
                success, frame = cap.read()
                if not success:
                    break
                if detect_blink_from_frame(frame):
                    blink_detected = True
                    break
            cap.release()
            return blink_detected
    except Exception as err:
        print("FaceMesh failed:", err)
    return False

def analyze_vid(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return -1, -1

    known_faces = []
    total_new_faces = 0
    real_humans = 0

    frame_index = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_index += 1
        locations = face_recognition.face_locations(frame)
        encodings = face_recognition.face_encodings(frame, locations)

        for idx, (enc, (top, right, bottom, left)) in enumerate(zip(encodings, locations)):
            matches = face_recognition.compare_faces(known_faces, enc, tolerance=0.6)
            if not any(matches):
                known_faces.append(enc)
                total_new_faces += 1
                face_img = frame[top:bottom, left:right]
                try:
                    if is_real_person(face_img):
                        real_humans += 1
                except Exception as e:
                    print(f"Error in liveliness check at frame {frame_index}, face #{idx}: {e}")
                    continue
    cap.release()
    return total_new_faces, real_humans


