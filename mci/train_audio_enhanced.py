import pandas as pd
import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import joblib
import config

def extract_audio_features(audio_path, fixed_length=None):
    """Extract multiple audio features including MFCC, spectral, and rhythm features."""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, duration=30)  # limit to 30 seconds for consistency
        
        if fixed_length is not None:
            if len(y) > fixed_length:
                y = y[:fixed_length]
            else:
                y = np.pad(y, (0, max(0, fixed_length - len(y))), mode='constant')

        # Extract features
        features = []
        
        # 1. MFCC (flattened statistics)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_stds = np.std(mfccs, axis=1)
        features.extend(mfcc_means)
        features.extend(mfcc_stds)
        
        # 2. Spectral Centroid (global statistics)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features.append(np.mean(spectral_centroids))
        features.append(np.std(spectral_centroids))
        
        # 3. Spectral Rolloff (global statistics)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features.append(np.mean(spectral_rolloff))
        features.append(np.std(spectral_rolloff))
        
        # 4. Zero Crossing Rate (global statistics)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features.append(np.mean(zcr))
        features.append(np.std(zcr))
        
        # 5. Root Mean Square Energy (global statistics)
        rmse = librosa.feature.rms(y=y)[0]
        features.append(np.mean(rmse))
        features.append(np.std(rmse))
        
        # 6. Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append(tempo)
        
        # Verify feature vector length (should be 13*2 + 2*4 + 1 = 35)
        expected_length = 35
        if len(features) != expected_length:
            print(f"Warning: Feature vector length {len(features)} != expected {expected_length}")
            # Print debug info about elements
            for i, el in enumerate(features):
                try:
                    shape = np.shape(el)
                except Exception:
                    shape = 'unknown'
                print(f"  idx {i}: type={type(el).__name__}, shape={shape}")
            return None

        # Coerce all elements to scalar floats; if any element is an array with >1 element, skip file
        features_clean = []
        for i, el in enumerate(features):
            if isinstance(el, np.ndarray):
                if el.shape == () or el.size == 1:
                    features_clean.append(float(el))
                else:
                    print(f"Non-scalar feature at idx {i}, shape {el.shape} - skipping file")
                    return None
            elif isinstance(el, (list, tuple)):
                if len(el) == 1:
                    features_clean.append(float(el[0]))
                else:
                    print(f"Non-scalar feature (list/tuple) at idx {i}, len {len(el)} - skipping file")
                    return None
            else:
                try:
                    features_clean.append(float(el))
                except Exception:
                    print(f"Could not convert feature idx {i} of type {type(el)} to float - skipping file")
                    return None

        return np.array(features_clean)
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

print("Loading data...")
# Load labels
labels_df = pd.read_csv(config.AUDIO_LABELS)
folder = config.DATA_OUTPUT

print(f"Processing {len(labels_df)} files from {folder}")

# Extract features
X = []
y = []
# Calculate fixed length for all audio files (30 seconds * sample rate)
FIXED_LENGTH = 30 * 22050  # 30 seconds at 22050 Hz

for idx, row in labels_df.iterrows():
    file_path = os.path.join(folder, row['filename'])
    if not os.path.exists(file_path):
        print(f"Missing file: {file_path}")
        continue
        
    features = extract_audio_features(file_path, fixed_length=FIXED_LENGTH)
    if features is not None:
        X.append(features)
        y.append(row['label'])

# Convert to numpy arrays
if len(X) == 0:
    print("No valid features extracted. Please check the audio files and preprocessing.")
    exit(1)

X_np = np.array(X)
y_np = np.array(y)

print(f"\nProcessed files: {len(X_np)}")
print(f"Features per file: {X_np.shape[1]}")
print(f"Labels distribution: {np.unique(y_np, return_counts=True)}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_np)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y_np)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Define XGBoost parameters for grid search
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200],
    'min_child_weight': [1, 3],
    'gamma': [0, 0.1],
    'subsample': [0.8, 1.0]
}

# Compute scale_pos_weight to help with imbalance (neg/pos ratio)
neg_count = np.sum(y_train == 0)
pos_count = np.sum(y_train == 1)
if pos_count == 0:
    scale_pos = 1.0
else:
    scale_pos = float(neg_count) / float(pos_count)

# Base classifier with fixed params (don't pass deprecated use_label_encoder)
base_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    base_score=0.5,
    random_state=42,
    scale_pos_weight=scale_pos
)

print("\nPerforming randomized search (quick diagnostics)...")
# Use randomized search to be faster on small datasets
random_search = RandomizedSearchCV(
    estimator=base_clf,
    param_distributions=param_grid,
    n_iter=20,
    cv=3,
    scoring='f1_macro',
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("\nBest parameters:", random_search.best_params_)
print("Best cross-validation f1_macro:", random_search.best_score_)

# Get best model
best_model = random_search.best_estimator_

# Evaluate on test set
y_pred = best_model.predict(X_test)
print("\nTest Set Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Feature importances (if available)
try:
    importances = best_model.feature_importances_
    print("Feature importances (first 10):", importances[:10])
except Exception:
    print("Model does not expose feature_importances_")

# Save models and preprocessing objects
print("\nSaving models and preprocessors...")
joblib.dump(best_model, config.XGB_MODEL)
joblib.dump(le, config.LABEL_ENCODER)
joblib.dump(scaler, os.path.join(config.DATA_OUTPUT, 'audio_scaler.pkl'))

print(f"Saved to {config.DATA_OUTPUT}:"
      f"\n - Model: {os.path.basename(config.XGB_MODEL)}"
      f"\n - Label Encoder: {os.path.basename(config.LABEL_ENCODER)}"
      f"\n - Scaler: audio_scaler.pkl")