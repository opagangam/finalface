import numpy as np

def calculate_EAR(landmarks, image_shape):
    # EAR landmarks for right eye (adjust if you prefer left)
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    def _distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    h, w = image_shape[:2]
    coords = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in RIGHT_EYE]

    # Calculate EAR
    A = _distance(coords[1], coords[5])
    B = _distance(coords[2], coords[4])
    C = _distance(coords[0], coords[3])
    ear = (A + B) / (2.0 * C)
    return ear
