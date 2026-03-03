"""
Simple video training script (template).

- Expects a CSV at data/video_labels.csv with columns: filename,label
- Expects video files under data/videos/ matching filenames in the CSV
- Extracts Mediapipe face landmarks per frame, aggregates per-video (mean/std)
  and trains an XGBoost classifier on the aggregated features.

This is a lightweight starting point — adapt paths/features as needed.
"""
import os
import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import joblib
import config

VIDEO_DIR = os.path.join('data', 'videos')
LABEL_CSV = os.path.join('data', 'video_labels.csv')

mp_face = mp.solutions.face_mesh

def extract_video_features(video_path, max_frames=300, frame_step=5):
    """Extract aggregated facial-landmark statistics for a video file.
    Returns a 1D numpy array feature vector or None on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return None

    face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1)
    landmarks_list = []
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue
        # Resize/convert
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]
            coords = []
            h, w, _ = frame.shape
            for point in lm.landmark:
                coords.append(point.x)
                coords.append(point.y)
                coords.append(point.z)
            landmarks_list.append(coords)
        frame_idx += 1
        if frame_idx >= max_frames:
            break

    cap.release()
    face_mesh.close()

    if len(landmarks_list) == 0:
        return None

    arr = np.array(landmarks_list)  # shape: (num_samples, 468*3)
    # Aggregate by column: mean and std
    feats = np.concatenate([np.mean(arr, axis=0), np.std(arr, axis=0)])
    return feats


def main():
    if not os.path.exists(LABEL_CSV):
        print(f"Label CSV not found: {LABEL_CSV}. Create data/video_labels.csv with columns filename,label")
        return

    df = pd.read_csv(LABEL_CSV)
    X = []
    y = []
    for _, row in df.iterrows():
        video_path = os.path.join(VIDEO_DIR, row['filename'])
        if not os.path.exists(video_path):
            print(f"Missing video: {video_path}")
            continue
        feats = extract_video_features(video_path)
        if feats is not None:
            X.append(feats)
            y.append(row['label'])
        else:
            print(f"No features for {video_path}, skipping")

    if len(X) == 0:
        print("No video features extracted. Provide videos and labels.")
        return

    X = np.array(X)
    y = np.array(y)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(Xs, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    clf = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save artifacts
    out_model = os.path.join(config.DATA_OUTPUT, 'alzheimers_video_xgb_model.pkl')
    joblib.dump(clf, out_model)
    joblib.dump(le, os.path.join(config.DATA_OUTPUT, 'video_label_encoder.pkl'))
    joblib.dump(scaler, os.path.join(config.DATA_OUTPUT, 'video_scaler.pkl'))
    print("Saved video model and preprocessors to:", config.DATA_OUTPUT)

if __name__ == '__main__':
    main()
