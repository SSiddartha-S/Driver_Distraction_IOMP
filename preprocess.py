# preprocess.py
"""
Preprocess images to extract keypoint sequences.
- Uses YOLOv8 (ultralytics) to detect driver bbox.
- Uses MediaPipe (pose + face + hands) to get keypoints within crop.
- Combines a fixed set of keypoints into K points per frame.
- Builds sliding-window sequences of length SEQ_LEN and saves .npz files in data/sequences.
"""

import os, glob, math
import numpy as np
import cv2
from ultralytics import YOLO
import mediapipe as mp
from tqdm import tqdm
from kalman import KeypointKalman

RAW_DIR = "data/raw"
SEQ_DIR = "data/sequences"
SEQ_LEN = 16

os.makedirs(SEQ_DIR, exist_ok=True)

# initialize detectors
yolo = YOLO("yolov8n.pt")  # will auto-download if needed
mp_pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, min_detection_confidence=0.5)
mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, min_detection_confidence=0.5)

kalman = KeypointKalman()

def detect_crop(img):
    res = yolo(img, verbose=False)
    if len(res) == 0 or len(res[0].boxes) == 0:
        return img  # fallback
    boxes = res[0].boxes
    areas = (boxes.xyxy[:,2]-boxes.xyxy[:,0]) * (boxes.xyxy[:,3]-boxes.xyxy[:,1])
    idx = int(np.argmax(areas))
    x1,y1,x2,y2 = boxes.xyxy[idx].cpu().numpy().astype(int)
    h,w = img.shape[:2]
    x1 = max(0,x1); y1 = max(0,y1); x2 = min(w-1,x2); y2 = min(h-1,y2)
    crop = img[y1:y2, x1:x2].copy()
    return crop

def extract_keypoints(crop):
    """
    Returns keypoints (K,2) in pixel coordinates relative to crop (w,h).
    We'll assemble: pose 33 -> select 11 (major joints); face -> select 5 landmarks (eyes/nose); hands -> 21 each.
    Total K = 11 + 5 + 21 + 21 = 58
    If detection missing for a part, fill zeros.
    """
    h, w = crop.shape[:2]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # pose
    pose_res = mp_pose.process(rgb)
    pose_kps = None
    if pose_res.pose_landmarks:
        pts = pose_res.pose_landmarks.landmark
        # choose a subset indices for robustness: nose (0), left/right shoulders (11,12), elbows (13,14), wrists (15,16), hips (23,24), knees (25,26)
        sel = [0,11,12,13,14,15,16,23,24,25,26]
        pose_kps = []
        for i in sel:
            lm = pts[i]
            pose_kps.append([lm.x * w, lm.y * h])
        pose_kps = np.array(pose_kps, dtype=np.float32)
    else:
        pose_kps = np.zeros((11,2), dtype=np.float32)

    # face (face_mesh has 468 points) - pick nose tip-ish (1), left eye (33), right eye (263), left mouth corner (61), right mouth corner (291)
    face_res = mp_face.process(rgb)
    if face_res.multi_face_landmarks:
        lm = face_res.multi_face_landmarks[0]
        idxs = [1,33,263,61,291]
        face_kps = []
        for i in idxs:
            p = lm.landmark[i]
            face_kps.append([p.x * w, p.y * h])
        face_kps = np.array(face_kps, dtype=np.float32)
    else:
        face_kps = np.zeros((5,2), dtype=np.float32)

    # hands (left then right). mp_hands returns list of hands with classification
    hands_res = mp_hands.process(rgb)
    left_hand = np.zeros((21,2), dtype=np.float32)
    right_hand = np.zeros((21,2), dtype=np.float32)
    if hands_res.multi_hand_landmarks and hands_res.multi_handedness:
        # loop and assign
        for hand_landmarks, handedness in zip(hands_res.multi_hand_landmarks, hands_res.multi_handedness):
            label = handedness.classification[0].label  # 'Left' or 'Right'
            pts = []
            for lm in hand_landmarks.landmark:
                pts.append([lm.x * w, lm.y * h])
            arr = np.array(pts, dtype=np.float32)
            if label == "Left":
                left_hand = arr
            else:
                right_hand = arr

    # combine
    kps = np.concatenate([pose_kps, face_kps, left_hand, right_hand], axis=0)  # (58,2)
    return kps

def process_class(class_dir, label_idx):
    imgs = sorted(glob.glob(os.path.join(class_dir, "*.jpg")))
    kp_buffer = []
    count = 0
    for p in tqdm(imgs, desc=os.path.basename(class_dir)):
        img = cv2.imread(p)
        if img is None:
            continue
        crop = detect_crop(img)
        kps = extract_keypoints(crop)  # (58,2)
        kp_buffer.append(kps)
        if len(kp_buffer) >= SEQ_LEN:
            seq = np.stack(kp_buffer[-SEQ_LEN:], axis=0)  # T x K x 2
            seq_sm = kalman.smooth_sequence(seq)
            out_name = os.path.join(SEQ_DIR, f"{label_idx}_{os.path.basename(class_dir)}_{count:06d}.npz")
            np.savez_compressed(out_name, keypoints=seq_sm, label=label_idx)
            count += 1

def main():
    classes = sorted([d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))])
    print("Found classes:", classes)
    for idx, cls in enumerate(classes):
        class_dir = os.path.join(RAW_DIR, cls)
        process_class(class_dir, idx)
    print("Done. Sequences in", SEQ_DIR)

if __name__ == "__main__":
    main()
