# Early Detection of Alzheimer's Disease

Multi-modal Alzheimer's disease detection system using clinical features, audio analysis, and MRI scans. The app helps predict if a user has Alzheimer's by analyzing:
- Clinical features (tabular data)
- Voice samples (audio analysis)
- Brain MRI scans (deep learning)

## Features

- Clinical feature analysis (tabular data)
- Audio-based detection (voice samples)
- MRI scan analysis with deep learning (VGG16/DenseNet)
- Dual interface: GUI and Web (Streamlit)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Data structure:
```
data/
├── Combined Dataset/
│   ├── train/
│   │   ├── Mild_Demented/
│   │   ├── Moderate_Demented/
│   │   ├── Non_Demented/
│   │   └── Very_Mild_Demented/
│   └── test/
│       ├── Mild_Demented/
│       ├── Moderate_Demented/
│       ├── Non_Demented/
│       └── Very_Mild_Demented/
├── alzheimers_data.csv     # Clinical features dataset
└── output/
    └── audio_labels_filtered.csv  # Audio analysis labels
```

## Running the Application

# Early Detection of Alzheimer's Disease (Unified)

This repository implements a small research/demo application for early detection of Alzheimer's Disease using three complementary modalities:

- Clinical tabular features (Random Forest)
- Voice/audio features (MFCC + XGBoost/RF)
- Brain MRI scans (CNN-based classifier)
- Real-time facial/video analysis (MediaPipe Face Mesh) — desktop GUI only

This README consolidates usage, developer notes, and troubleshooting for the unified desktop GUI (`app.py`) which exposes four pages: Tabular Prediction, Audio Prediction (upload + record), MRI Analysis (image upload), and Video Prediction (camera with landmark visualization).

## Quickstart

1. Create a Python environment (recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate
pip install -r requirements.txt
```

2. Run the desktop GUI (single unified application):

```powershell
python app.py
```

Notes:
- `app.py` is the merged GUI with all pages. There is a backup `app_with_facial.py` that was used during development but `app.py` is the authoritative unified app.
- If you intend to run the Streamlit web UI (experimental), see `streamlit_app.py` (if present) and run `streamlit run streamlit_app.py`.

## UI Overview (app.py)

The GUI contains a left-side menu with four buttons that toggle the main content pane:

- Tabular Prediction — enter clinical features manually and click Predict.
- Audio Prediction — upload an audio file or record 5 seconds directly and run prediction.
- MRI Analysis — upload a brain MRI image and run the image classifier.
- Video Prediction — start/stop your webcam and view MediaPipe face-mesh landmarks in real time.

Each page shows helpful status text and result panels. There is a right-side Visuals frame that displays a sample brain image (if `data/brain.jpg` exists) and can be extended to show other visuals.

## Files of interest

- `app.py` — Unified desktop GUI (Tabular / Audio / MRI / Video).
- `app_with_facial.py` — Development copy containing facial analysis helpers (kept for reference).
- `alzhimer.py` — Tabular model training / feature selection pipeline. Produces `top_features.txt` and `alzheimers_rf_small_model.pkl`.
- `run_xgboost.py`, `XGB.py`, `load_data.py` — Audio feature extraction and model training pipelines (expect CSV in `data/output`).
- `mri_model.py` — MRI model wrapper (AlzheimerMRIClassifier) used by the GUI (expects a `.pth` model file).
- `config.py` — Centralized paths and filenames (models, data locations). Edit this file to change where models are read/written.

## Configuration and paths

The app uses `config.py` to locate models and data. Default locations in the repo:

- `config.RF_MODEL` -> `alzheimers_rf_small_model.pkl` (tabular RF model)
- `config.RF_AUDIO_MODEL` -> `data/output/alzheimers_rf_model.pkl` (optional audio model)
- `config.XGB_MODEL` -> `data/output/alzheimers_xgb_model.pkl` (optional)
- `config.BRAIN_IMAGE` -> `data/brain.jpg` (optional displayed image)

If you move models to another folder, update `config.py` or drop model files at the above paths.

## Tabular (clinical) prediction

Inputs and behavior:

- The app loads `top_features.txt` to determine the canonical feature ordering used by the saved RF model. When you press Predict, the GUI reads each feature entry (empty values will raise an error) and feeds the model.
- Feature names expected (example): `FunctionalAssessment`, `ADL`, `MMSE`, `MemoryComplaints`, `BehavioralProblems`, `CholesterolLDL`, `DietQuality`, `CholesterolTriglycerides`.

Model file: `config.RF_MODEL` (joblib pickle).

Errors: If the tabular model is missing the app will show an error in the results box — produce the model by running `python alzhimer.py`.

## Audio prediction

Behavior:

- Upload: Select an audio file (WAV/FLAC/MP3). The GUI extracts 8 MFCC coefficients (mean over frames) and passes a 1×8 feature vector to the audio classifier.
- Record: Record 5 seconds (desktop microphone) and run the same extraction pipeline.

Model fallback:

- The app will try `config.RF_AUDIO_MODEL` first. If missing, the tabular RF model `config.RF_MODEL` is used as a fallback to keep the UI functional (useful in demos, not ideal for production).

Notes:

- Recording uses `sounddevice` — ensure microphone access and that the package works on your Windows setup.
- For long audio or batch processing, use the training scripts (`run_xgboost.py`) and produce a dedicated audio model.

## MRI analysis

- Upload an MRI image (PNG/JPG). The GUI displays a preview and calls `AlzheimerMRIClassifier.predict_image(path)` from `mri_model.py`.
- Ensure the expected MRI model file (e.g., `vgg_mri_model.pth`) is present or adjust the code to load the correct filename.

Training: see `train_mri.py` or `mri_model.py` for training scripts and options. Use the `--mode fast` option for quick iterations while developing.

## Video / Facial analysis

- Uses OpenCV to capture webcam frames and MediaPipe Face Mesh to detect and render 468 facial landmarks.
- Start Camera toggles the webcam and draws green dots on detected landmarks in the GUI.
- No production-ready expression classifier is bundled; the interface provides a placeholder for adding expression detection (compute landmark-derived features and pass them to a classifier).

Practical notes:

- Camera device index defaults to 0. If you have multiple cameras, edit `app.py` where `cv2.VideoCapture(0)` is called.
- MediaPipe and OpenCV can be CPU intensive; on low-RAM or CPU-only machines reduce frame refresh (increase `after()` delay) or disable the camera.

## Development and training tips

1. Install dependencies in a virtual environment:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate
pip install -r requirements.txt
```

2. Produce the tabular model (if missing):

```powershell
python alzhimer.py
```

3. Train audio model (example):

```powershell
python run_xgboost.py
```

4. Train MRI quick smoke test:

```powershell
python train_mri.py --model vgg --mode fast
```

Performance and monitoring

- GPU: `nvidia-smi` to monitor memory/usage.
- Windows: Task Manager for CPU/Memory.
- For PyTorch training, enable AMP on GPU for speed/memory.

## Troubleshooting

- ModelNotFound: Verify model files exist at paths defined in `config.py`.
- Audio recording errors: check microphone permissions, sample rate support and `sounddevice` availability on Windows.
- Camera fails to open: try a different device index, or ensure no other process (e.g., Teams/Zoom) occupies the camera.
- MediaPipe errors: upgrade `mediapipe` and `opencv-python` to recent compatible versions from `requirements.txt`.

## Next steps / TODOs

- Add expression classification from face-mesh landmarks and wire it to the Video Prediction page.
- Add tests and CI smoke checks (launch short training and UI flows headlessly).
- Centralize config to accept environment variables for model paths (useful for deployment).

## Contact / Contributions

If you want changes or help migrating this app to a web service, open an issue or PR with a clear description of the desired behavior.

---

This README consolidates previous scattered notes and reflects the unified `app.py` that bundles Tabular, Audio, MRI, and Video (facial) predictions in a single desktop application.
