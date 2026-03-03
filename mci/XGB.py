import pandas as pd
import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import config

# Read audio labels CSV and folder from config
labels_df = pd.read_csv(config.AUDIO_LABELS)
folder = config.DATA_OUTPUT

# Extract features
X = []
y = []
for idx, row in labels_df.iterrows():
    file_path = os.path.join(folder, row['filename'])
    if not os.path.exists(file_path):
        print(f"Missing file: {file_path}, skipping...")
        continue
    try:
        audio, sr = librosa.load(file_path)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        X.append(mfcc_mean)
        y.append(row['label'])
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

X_np = np.array(X)
y_np = np.array(y)

# Encode string labels to numeric
le = LabelEncoder()
y_encoded = le.fit_transform(y_np)

if len(X_np) > 8:
    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    clf = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=le.classes_))
else:
    clf = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    if len(X_np) > 1:
        scores = cross_val_score(clf, X_np, y_encoded, cv=len(X_np), error_score='raise')
        print("Cross-validation scores:", scores)
        print("Mean accuracy:", scores.mean())
    else:
        print("Not enough data available for training.")

# Save the trained model and label encoder
joblib.dump(clf, config.XGB_MODEL)
joblib.dump(le, config.LABEL_ENCODER)

