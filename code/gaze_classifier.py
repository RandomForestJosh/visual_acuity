import sys
import os
import cv2

# Add the gaze_tracking folder explicitly to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "gaze_tracking"))

from gaze_tracking import GazeTracking

# Initialize GazeTracking
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

def classify_region(gaze):
    """
    Classify gaze into 'top', 'middle', 'bottom' based on vertical ratio.
    """
    vertical_ratio = gaze.vertical_ratio()
    if vertical_ratio is None:
        return "undetected"
    if vertical_ratio < 0.4:
        return "top"
    elif vertical_ratio > 0.6:
        return "bottom"
    else:
        return "middle"

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    gaze.refresh(frame)
    frame = gaze.annotated_frame()

    region = classify_region(gaze)
    cv2.putText(frame, f"Gaze region: {region}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gaze Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
