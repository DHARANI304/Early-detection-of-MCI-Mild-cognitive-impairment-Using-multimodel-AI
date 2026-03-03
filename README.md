🧠 Alzheimer’s Multi-Modal Early Detection System

An AI-powered multi-modal screening prototype designed to assist in the early detection of Mild Cognitive Impairment (MCI) and early-stage Alzheimer’s Disease.

This system integrates clinical cognitive scores, speech biomarkers, structural MRI imaging, and facial behavioral analysis into a unified machine learning framework for early cognitive risk assessment.

Built using PyTorch, XGBoost, and MediaPipe, the project demonstrates how multi-modal AI can enhance predictive reliability compared to single-source models.

🎯 Key Highlights

Developed 4 independent ML pipelines (Tabular, Audio, MRI, Facial)

Implemented CNN-based MRI classifier using VGG-16 backbone (PyTorch)

Designed speech feature extraction pipeline using librosa + XGBoost

Integrated facial landmark detection using MediaPipe

Built an interactive web interface using Streamlit

Structured system for scalable multi-modal fusion architecture

🧠 System Architecture

Multi-Modal Inputs

Clinical Test Scores (MMSE, ADL)

Audio Speech Samples

Structural MRI Scans

Real-Time Facial Video Feed

⬇

Independent ML Models

Scikit-learn classifier (Tabular)

XGBoost classifier (Audio)

CNN (MRI Imaging)

Facial Landmark Feature Model

⬇

Combined Risk Assessment Output

📊 Model Evaluation

Each model pipeline was evaluated using standard classification metrics:

Accuracy

Precision

Recall

F1-Score

🛠 Tech Stack
Programming

Python

Machine Learning & AI

PyTorch (CNN for MRI)

Scikit-Learn (Tabular Clinical Data)

XGBoost (Audio Classification)

MediaPipe (Facial Landmark Detection)

Librosa (Speech Feature Extraction)

Deployment

Streamlit (Web Application)

Tkinter (Desktop GUI)

⚙️ Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/DHARANI304/Early-detection-of-MCI-Mild-cognitive-impairment-Using-multimodel-AI.git
cd Early-detection-of-MCI-Mild-cognitive-impairment-Using-multimodel-AI
2️⃣ Create Virtual Environment
python -m venv .venv
.\.venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run Application
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
🚀 Future Improvements

Model ensemble weighting optimization

Integration of transformer-based speech models

Larger MRI dataset training

Cloud deployment (AWS / Azure)

Model explainability using SHAP / Grad-CAM

📦 Model Weights

Due to GitHub file size limitations, trained model weights (.pth) are not included in this repository.

You can:

Train the model using train_mri.py
OR

Provide a Google Drive link for pretrained weights

⚠️ Medical Disclaimer

This project is a research prototype intended for educational and demonstration purposes only.

It is NOT a certified diagnostic tool and must not replace professional medical advice, diagnosis, or treatment.
