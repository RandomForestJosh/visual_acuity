import cv2
import mediapipe as mp
import numpy as np
from math import hypot

# MediaPipe solutions
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Eye landmark indices (MediaPipe Face Mesh)
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [263, 387, 385, 362, 380, 373]

# Frame smoothing
NUM_FRAMES = 5
left_history = []
right_history = []

def eye_aspect_ratio(landmarks, eye_indices, frame_w, frame_h):
    p = [(int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h)) for i in eye_indices]
    vertical1 = hypot(p[1][0]-p[5][0], p[1][1]-p[5][1])
    vertical2 = hypot(p[2][0]-p[4][0], p[2][1]-p[4][1])
    horizontal = hypot(p[0][0]-p[3][0], p[0][1]-p[3][1])
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def eye_bbox(landmarks, eye_indices, frame_w, frame_h, pad_ratio=0.25):
    """Returns pixel bbox (x1, y1, x2, y2) for eye region."""
    xs = [int(landmarks[i].x * frame_w) for i in eye_indices]
    ys = [int(landmarks[i].y * frame_h) for i in eye_indices]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad_x = int((x_max - x_min) * pad_ratio)
    pad_y = int((y_max - y_min) * pad_ratio)
    return max(0, x_min - pad_x), max(0, y_min - pad_y), min(frame_w, x_max + pad_x), min(frame_h, y_max + pad_y)

def bbox_overlap(b1, b2):
    """Returns True if two bboxes overlap."""
    x1, y1, x2, y2 = b1
    x3, y3, x4, y4 = b2
    return not (x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    with mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as face_mesh, \
         mp_hands.Hands(max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Face landmarks
            face_results = face_mesh.process(rgb)
            hand_results = hands.process(rgb)

            label_text = "No face detected"
            left_occluded = False
            right_occluded = False

            eye_bboxes = []

            if face_results.multi_face_landmarks:
                landmarks = face_results.multi_face_landmarks[0].landmark

                # EAR detection (blinks)
                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_LANDMARKS, w, h)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_LANDMARKS, w, h)
                left_occluded = left_ear < 0.2
                right_occluded = right_ear < 0.2

                # Eye bounding boxes
                left_bbox = eye_bbox(landmarks, LEFT_EYE_LANDMARKS, w, h)
                right_bbox = eye_bbox(landmarks, RIGHT_EYE_LANDMARKS, w, h)
                eye_bboxes = [left_bbox, right_bbox]

                # Draw face mesh
                mp_drawing.draw_landmarks(
                    frame, face_results.multi_face_landmarks[0],
                    mp_face.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1))

            # Hand detection
            if hand_results.multi_hand_landmarks:
                for hand_lms in hand_results.multi_hand_landmarks:
                    # Hand bbox
                    xs = [lm.x * w for lm in hand_lms.landmark]
                    ys = [lm.y * h for lm in hand_lms.landmark]
                    hand_bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
                    # Draw hand bbox
                    cv2.rectangle(frame, (hand_bbox[0], hand_bbox[1]),
                                  (hand_bbox[2], hand_bbox[3]), (0,0,255), 2)
                    # Check overlap with eyes
                    for i, eye_bbox_coords in enumerate(eye_bboxes):
                        if bbox_overlap(hand_bbox, eye_bbox_coords):
                            if i == 0:
                                left_occluded = True
                            else:
                                right_occluded = True

            # Update history buffers for smoothing
            left_history.append(left_occluded)
            right_history.append(right_occluded)
            if len(left_history) > NUM_FRAMES:
                left_history.pop(0)
            if len(right_history) > NUM_FRAMES:
                right_history.pop(0)

            left_smoothed = sum(left_history) > (NUM_FRAMES / 2)
            right_smoothed = sum(right_history) > (NUM_FRAMES / 2)

            # Determine label
            if not left_smoothed and not right_smoothed:
                label_text = "Eyes uncovered"
            elif left_smoothed and not right_smoothed:
                label_text = "Left eye covered"
            elif not left_smoothed and right_smoothed:
                label_text = "Right eye covered"
            else:
                label_text = "Both eyes covered"

            # Overlay label
            cv2.putText(frame, label_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 3, cv2.LINE_AA)

            cv2.imshow("Eye + Hand Occlusion Detection", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
