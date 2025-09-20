import cv2
import mediapipe as mp
import numpy as np
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

LABELS = ['Point Right', 'Point Left', 'Point Down', 'Point Up']

def detect_point_direction(landmarks):
    """
    Determine hand pointing direction based on wrist -> middle fingertip vector.
    landmarks: list of normalized landmarks
    """
    wrist = landmarks[0]
    middle_tip = landmarks[12]

    dx = middle_tip.x - wrist.x
    dy = middle_tip.y - wrist.y

    angle = math.degrees(math.atan2(-dy, dx))  # invert y because OpenCV coordinates
    # angle in [-180, 180]
    if -45 <= angle <= 45:
        return 'Point Right'
    elif 45 < angle <= 135:
        return 'Point Up'
    elif -135 <= angle < -45:
        return 'Point Down'
    else:
        return 'Point Left'

def landmarks_to_bbox(landmarks, frame_w, frame_h, pad=0.05):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x_min = max(0, min(xs) - pad)
    x_max = min(1, max(xs) + pad)
    y_min = max(0, min(ys) - pad)
    y_max = min(1, max(ys) + pad)
    return int(x_min * frame_w), int(y_min * frame_h), int((x_max - x_min) * frame_w), int((y_max - y_min) * frame_h)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            label_text = ''

            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
                hand_lms = results.multi_hand_landmarks[0]

                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                # Bounding box
                x, y, w, h = landmarks_to_bbox(hand_lms.landmark, frame_w, frame_h)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 3)

                # Detect pointing direction
                label_text = detect_point_direction(hand_lms.landmark)

            # Overlay label
            cv2.putText(frame, label_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('Hand Point Detection', frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
