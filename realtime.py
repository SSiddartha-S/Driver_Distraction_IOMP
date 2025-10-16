# realtime.py
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from kalman import KeypointKalman
from model import SimpleTransformerClassifier

SEQ_LEN = 16
MODEL_PATH = "models/model.pth"
CLASSES = ["safe","text_right","phone_right","text_left","phone_left","radio","drinking","reach","hair","talk_passenger"]

# init detectors
yolo = YOLO("yolov8n.pt")
mp_pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, min_detection_confidence=0.5)
mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, min_detection_confidence=0.5)
kalman = KeypointKalman()

def detect_crop(frame):
    res = yolo(frame, verbose=False)
    if len(res) == 0 or len(res[0].boxes) == 0:
        return frame
    boxes = res[0].boxes
    areas = (boxes.xyxy[:,2]-boxes.xyxy[:,0]) * (boxes.xyxy[:,3]-boxes.xyxy[:,1])
    idx = int(np.argmax(areas))
    x1,y1,x2,y2 = boxes.xyxy[idx].cpu().numpy().astype(int)
    h,w = frame.shape[:2]
    x1 = max(0,x1); y1 = max(0,y1); x2 = min(w-1,x2); y2 = min(h-1,y2)
    return frame[y1:y2, x1:x2].copy()

def extract_keypoints_from_crop(crop):
    h, w = crop.shape[:2]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # pose
    pose_res = mp_pose.process(rgb)
    sel = [0,11,12,13,14,15,16,23,24,25,26]
    if pose_res.pose_landmarks:
        pts = pose_res.pose_landmarks.landmark
        pose_kps = np.array([[pts[i].x * w, pts[i].y * h] for i in sel], dtype=np.float32)
    else:
        pose_kps = np.zeros((11,2), dtype=np.float32)

    # face
    face_res = mp_face.process(rgb)
    face_idx = [1,33,263,61,291]
    if face_res.multi_face_landmarks:
        lm = face_res.multi_face_landmarks[0]
        face_kps = np.array([[lm.landmark[i].x * w, lm.landmark[i].y * h] for i in face_idx], dtype=np.float32)
    else:
        face_kps = np.zeros((5,2), dtype=np.float32)

    # hands
    hands_res = mp_hands.process(rgb)
    left = np.zeros((21,2), dtype=np.float32)
    right = np.zeros((21,2), dtype=np.float32)
    if hands_res.multi_hand_landmarks and hands_res.multi_handedness:
        for hl, hh in zip(hands_res.multi_hand_landmarks, hands_res.multi_handedness):
            label = hh.classification[0].label
            arr = np.array([[lm.x * w, lm.y * h] for lm in hl.landmark], dtype=np.float32)
            if label == "Left":
                left = arr
            else:
                right = arr
    kps = np.concatenate([pose_kps, face_kps, left, right], axis=0)  # (58,2)
    return kps

# Warmup: collect initial buffer so we know K
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Make sure camera is connected.")

print("Warming up camera... please sit in front of camera.")
buffer = []
while len(buffer) < SEQ_LEN:
    ret, frame = cap.read()
    if not ret:
        continue
    crop = detect_crop(frame)
    kps = extract_keypoints_from_crop(crop)
    buffer.append(kps)
    cv2.imshow("Warmup", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyWindow("Warmup")

K = buffer[0].shape[0]
input_dim = K * 2

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleTransformerClassifier(input_dim=input_dim, num_classes=len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

seq_kps = buffer.copy()  # list of T entries: (K,2)
print("Starting real-time inference. Press ESC to exit.")
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    crop = detect_crop(frame)
    kps = extract_keypoints_from_crop(crop)
    seq_kps.append(kps)
    if len(seq_kps) > SEQ_LEN:
        seq_kps.pop(0)

    if len(seq_kps) == SEQ_LEN:
        seq = np.stack(seq_kps, axis=0)  # T x K x 2
        seq_sm = kalman.smooth_sequence(seq)
        flat = seq_sm.reshape(SEQ_LEN, -1).astype(np.float32)
        mean = flat.mean(); std = flat.std() if flat.std() > 0 else 1.0
        flat = (flat - mean) / std
        xb = torch.from_numpy(flat).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(xb)
            pred = out.argmax(dim=1).item()
            score = torch.softmax(out, dim=1)[0, pred].item()
            label = CLASSES[pred]
        cv2.putText(frame, f"{label} {score:.2f}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)

    cv2.imshow("Driver Monitor", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
