import joblib
import numpy as np
import librosa
import config
import os

def load_model():
    """Load the RF model and return it."""
    try:
        return joblib.load(config.RF_MODEL)
    except Exception:
        return None

def get_feature_names():
    """Get ordered feature names from top_features.txt."""
    with open('top_features.txt') as f:
        return [line.strip() for line in f if line.strip()]

def extract_audio_features(audio_path):
    """Extract MFCC features from audio file."""
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=8)
    mfccs_mean = np.mean(mfccs, axis=1)
    return [mfccs_mean]

def predict_from_values(model, values):
    """Make prediction from manual input values."""
    import pandas as pd
    feature_names = get_feature_names()
    df_input = pd.DataFrame([values], columns=feature_names)
    if model is None:
        raise RuntimeError('Model not found: ' + config.RF_MODEL)
    return model.predict(df_input)[0]

def predict_from_audio(model, audio_path):
    """Make prediction from audio file."""
    features = extract_audio_features(audio_path)
    if model is None:
        raise RuntimeError('Model not found: ' + config.RF_MODEL)
    return model.predict(features)[0]

def save_recorded_audio(audio_data, sample_rate):
    """Save recorded audio to a temporary WAV file and return the path."""
    import tempfile
    from scipy.io.wavfile import write as wav_write
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        wav_write(tmp.name, sample_rate, audio_data)
        return tmp.name