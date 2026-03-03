Alzheimer’s Multi-Modal Early Detection System

A machine learning prototype designed to assist in the early detection of Mild Cognitive Impairment (MCI) and early-stage Alzheimer’s Disease using a multi-modal AI approach.

This system integrates clinical test data, speech biomarkers, structural MRI scans, and facial behavioral markers into a unified predictive framework.

📌 Project Motivation

Early detection significantly improves patient outcomes in Alzheimer’s care.
Traditional diagnosis methods can be expensive, time-consuming, and inaccessible in early stages.

This project demonstrates a scalable, low-cost AI-powered screening prototype combining four independent modalities to evaluate cognitive health risk.

🛠 Tech Stack

Programming

Python

Machine Learning / AI

PyTorch (MRI CNN model)

Scikit-Learn (Clinical Tabular Data)
XGBoost (Audio Classification)
MediaPipe (Facial Landmark Detection)
Deployment
Streamlit (Web Interface)
Tkinter (Desktop GUI)

🚀 Key Features
📊 1. Tabular Clinical Analysis

Uses MMSE & ADL cognitive test scores

Scikit-learn based predictive modeling

Risk scoring for early cognitive decline

🎙 2. Audio Biomarker Analysis

Feature extraction using librosa
Speech rhythm and pause pattern analysis
XGBoost-based classification

🧠 3. MRI Imaging Analysis
CNN-based classification
VGG-16 backbone architecture
PyTorch implementation
Structural MRI pattern detection

🙂 4. Facial Behavioral Analysis

Real-time facial landmark extraction
Expressive anomaly detection
MediaPipe integration

🏗 System Architecture

Multi-Modal Inputs:
Clinical Data
Audio Recording
MRI Image
Live Facial Feed
↓
Independent ML Models
↓
Combined Risk Assessment Output

⚙️ Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/DHARANI304/Early-detection-of-MCI-Mild-cognitive-impairment-Using-multimodel-AI.git
cd Early-detection-of-MCI-Mild-cognitive-impairment-Using-multimodel-AI
2️⃣ Create Virtual Environment
python -m venv .venv
.\.venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run Streamlit App
streamlit run streamlit_app.py
📂 Project Structure
├── streamlit_app.py
├── train_mri.py
├── train_audio_enhanced.py
├── XGB.py
├── load_data.py
├── utils.py
├── requirements.txt
├── README.md
📈 Future Improvements

Model ensemble weighting optimization
Integration of transformer-based speech models
Larger MRI dataset training
Cloud deployment (AWS / Azure)
Model explainability (SHAP / Grad-CAM)

⚠️ Medical Disclaimer

This project is a research prototype intended for educational and demonstration purposes only.
It is NOT a certified diagnostic tool and must not replace professional medical advice, diagnosis, or treatment.
