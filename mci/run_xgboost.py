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

# Use only the filtered CSV with existing files
labels_df = pd.read_csv(config.AUDIO_LABELS)
folder = config.DATA_OUTPUT

print(f"Loading audio files from: {folder}")
print(f"Total files in CSV: {len(labels_df)}")

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

print(f"\nProcessed files: {len(X_np)}")
print(f"Labels distribution: {np.unique(y_np, return_counts=True)}")

# Encode string labels to numeric
le = LabelEncoder()
y_encoded = le.fit_transform(y_np)

# Configure XGBoost with proper parameters
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'learning_rate': 0.1,
    'max_depth': 3,
    'n_estimators': 100,
    'base_score': 0.5,  # This fixes the base_score error
    'random_state': 42
}

if len(X_np) > 8:
    # Regular train/test split for larger datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    clf = xgb.XGBClassifier(**xgb_params)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
else:
    # Cross-validation for small datasets
    clf = xgb.XGBClassifier(**xgb_params)
    if len(X_np) > 1:
        scores = cross_val_score(clf, X_np, y_encoded, cv=min(len(X_np), 5))
        print("\nCross-validation scores:", scores)
        print("Mean accuracy:", scores.mean())
        
        # Fit on all data for final model
        clf.fit(X_np, y_encoded)
    else:
        print("Not enough data available for training")

# Save the trained model and label encoder
joblib.dump(clf, config.XGB_MODEL)
joblib.dump(le, config.LABEL_ENCODER)
print(f"\nSaved model to: {config.XGB_MODEL}")
print(f"Saved label encoder to: {config.LABEL_ENCODER}")
