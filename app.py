import os
import gradio as gr
from utils import get_img, find_faces_img, is_real_person, analyze_vid
from db import setup_db, record_attendance
import cv2
import numpy as np
import mediapipe as mp
import time
from something import calculate_EAR

setup_db()
media_dir = 'test_media'

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye_indices):
    p = [np.array([landmarks[i].x, landmarks[i].y]) for i in eye_indices]
    vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])
    return vertical / (2.0 * horizontal)

def blink_and_face_scan(duration=3):
    import cv2
    import time
    import mediapipe as mp

    print(">>> Starting blink and face scan")

    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    mp_drawing = mp.solutions.drawing_utils

    # Open webcam
    cap = cv2.VideoCapture(0)

    # Blink detection settings
    blink_detected = False
    max_face_count = 0
    frame_count = 0
    blink_counter = 0
    blink_threshold = 0.25
    consecutive_frames = 2

    # Timer
    start_time = time.time()

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        print("Frame captured:", ret)
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            current_faces = len(result.multi_face_landmarks)
            max_face_count = max(max_face_count, current_faces)
            for landmarks in result.multi_face_landmarks:
                # Draw facial landmarks
                mp_drawing.draw_landmarks(
                    frame, landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
                )

                # Calculate EAR
                ear = calculate_EAR(landmarks, frame.shape)
                print("EAR:", ear)

                # Blink logic
                if ear < blink_threshold:
                    blink_counter += 1
                    
                else:
                    if blink_counter >= consecutive_frames:
                        blink_detected = True
                        print("Blink Detected!")
                    blink_counter = 0

                # Display EAR on screen
                cv2.putText(frame, f"EAR: {ear:.3f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                if blink_detected:
                    cv2.putText(frame, "Blink Detected", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else:
            cv2.putText(frame, "No Face Detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show the frame
        cv2.imshow("Blink & Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        time.sleep(0.05)

    cap.release()
    cv2.destroyAllWindows()
    return blink_detected, max_face_count


def handle_file(file=None):
    path = None
    if file is not None:
        path = file

    if not path or not os.path.exists(path):
        msg = f"Can't find file: {path}"
        return msg

    fname = os.path.basename(path).lower()

    if fname.endswith((".jpg", ".jpeg", ".png", ".webp")):
        img = get_img(path)
        face_boxes = find_faces_img(img)
        seen = len(face_boxes)
        real = 0

        for box in face_boxes:
            t, r, b, l = box
            try:
                snippet = img[t:b, l:r]
                if is_real_person(snippet):
                    real += 1
            except:
                continue

        record_attendance(seen, real)
        msg = f"[IMG] {fname} — {seen} seen / {real} real"
        print(msg)
        return msg

    elif fname.endswith((".mp4", ".webm", ".avi", ".mov")):
        seen, real = analyze_vid(path)
        record_attendance(seen, real)
        msg = f"[VID] {fname} — {seen} seen / {real} real"
        print(msg)
        return msg

    else:
        msg = f"File format not supported -> {fname}"
        print(msg)
        return msg


with gr.Blocks() as iface:
    gr.Markdown("### Face Recognition")
    
    with gr.Row():
        file_input = gr.File(label="Upload image or video (optional)")

    output = gr.Textbox(label='Result')

    def process(file):
        if file is not None and os.path.exists(file):
            return handle_file(file)
        
        print("No file uploaded — launching OpenCV camera for blink detection.")
        blinked, count = blink_and_face_scan()
        return f"Blink Detected: {blinked}, Faces Seen: {count}"


    submit = gr.Button("Analyse")
    submit.click(fn=process, inputs=[file_input], outputs=output)

iface.launch(share=True)

def go_through_folder():
    if not os.path.isdir(media_dir):
        print("media folder missing.")
        return

    things = os.listdir(media_dir)
    if not things:
        print("Empty folder.")
        return

    print(f"Checking {len(things)} items in '{media_dir}'...\n")

    for item in things:
        full = os.path.join(media_dir, item)
        if os.path.isfile(full):
            handle_file(full)

if __name__ == '__main__':
    setup_db()
    go_through_folder()
