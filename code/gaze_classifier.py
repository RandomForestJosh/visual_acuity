import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# -------------------
# PyTorch gaze classifier
# -------------------
class GazeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 3)  # top, middle, bottom

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# -------------------
# Dataset class
# -------------------
class GazeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------------------
# MediaPipe setup
# -------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def get_eye_landmarks(landmarks, image_shape):
    h, w, _ = image_shape
    left_eye = landmarks[33]  # left pupil approx
    right_eye = landmarks[263]  # right pupil approx
    return np.array([left_eye.x*w, left_eye.y*h, right_eye.x*w, right_eye.y*h], dtype=np.float32)

# -------------------
# Data collection
# -------------------
def collect_data(samples_per_class=50):
    cap = cv2.VideoCapture(0)
    data_X, data_y = [], []

    class_names = ["top", "middle", "bottom"]
    for label, name in enumerate(class_names):
        print(f"Look at the {name.upper()} region. Press 's' to start collecting {samples_per_class} frames...")
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            cv2.putText(frame, f"Look at: {name.upper()}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Data Collection", frame)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break

        count = 0
        while count < samples_per_class:
            ret, frame = cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                coords = get_eye_landmarks(landmarks, frame.shape)
                coords_norm = coords / np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]], dtype=np.float32)
                data_X.append(coords_norm)
                data_y.append(label)
                count += 1
            cv2.putText(frame, f"Collecting {name.upper()} {count}/{samples_per_class}", (50,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Data Collection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return np.array(data_X), np.array(data_y)

    cap.release()
    cv2.destroyAllWindows()
    return np.array(data_X), np.array(data_y)

# -------------------
# Training
# -------------------
def train_model(X, y, epochs=50):
    dataset = GazeDataset(X, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = GazeClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    return model

# -------------------
# Real-time prediction
# -------------------
def run_gaze_tracking(model):
    cap = cv2.VideoCapture(0)
    model.eval()
    class_names = ["top", "middle", "bottom"]

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            coords = get_eye_landmarks(landmarks, frame.shape)
            coords_norm = coords / np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]], dtype=np.float32)
            eye_tensor = torch.tensor(coords_norm, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pred = torch.argmax(model(eye_tensor), dim=1).item()
            gaze_direction = class_names[pred]
            cv2.putText(frame, f"Gaze: {gaze_direction}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Gaze Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------------
# Main workflow
# -------------------
if __name__ == "__main__":
    X, y = collect_data(samples_per_class=50)
    model = train_model(X, y, epochs=30)
    run_gaze_tracking(model)
