import streamlit as st
import utils
import sounddevice as sd
import numpy as np
import config
from PIL import Image
import os
import cv2
from facial_analyzer import FacialAnalyzer
import tempfile
import base64

# Configure page
st.set_page_config(
    page_title="MCI Detect",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# Add custom card component
def styled_card(title, content, color="#ffffff", text_color="#000000"):
    return f"""
    <div class="prediction-box" style="background-color: {color}; color: {text_color}">
        <h3>{title}</h3>
        <p>{content}</p>
    </div>
    """

# Load model once
@st.cache_resource
def get_model():
    return utils.load_model()

model = get_model()

# Sidebar with custom styling
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h1 style='color: #2e4057;'>🧠 MCI Detect</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation with custom styling
    st.markdown("<div class='sidebar-nav'>", unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        ["Home", "Tabular Prediction", "Audio Prediction", "MRI Analysis", "Video Prediction"],
        format_func=lambda x: f"🏠 {x}" if x == "Home"
        else f"📊 {x}" if x == "Tabular Prediction"
        else f"🎵 {x}" if x == "Audio Prediction"
        else f"🔬 {x}" if x == "MRI Analysis"
        else f"📹 {x}"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Model status with styled card
    model_status = "✅ Model loaded" if model is not None else "❌ Model missing"
    st.markdown(
        styled_card(
            "System Status",
            model_status,
            "#f8f9fa"
        ),
        unsafe_allow_html=True
    )

    # Brain image with enhanced styling
    if os.path.exists(config.BRAIN_IMAGE):
        img = Image.open(config.BRAIN_IMAGE)
        # use_container_width replaces deprecated use_column_width
        st.image(img, caption="Brain Scan Reference", use_container_width=True)
        
    # Add version info
    st.markdown("""
    <div style='position: fixed; bottom: 0; padding: 1rem;'>
        <p style='color: #666; font-size: 0.8rem;'>MCI Detect v1.0</p>
    </div>
    """, unsafe_allow_html=True)

if page == "Home":
    st.markdown("""
        <h1 style='color: #2e4057; margin-bottom: 0.5rem;'>🏠 Welcome to MCI Detect</h1>
        <p style='color: #444; font-size: 1.05rem;'>Early detection of mild cognitive impairment and Alzheimer's disease.</p>
        <p style='color: #444; font-size: 1.05rem;'>Analyze clinical, audio, MRI and facial indicators quickly and easily.</p>
    """, unsafe_allow_html=True)

    # Show two attractive images side-by-side. Prefer local, otherwise use provided URLs.
    left_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR6AeV4myVbGe-hSkw_K9Jdty7UGtMbXgafBg&s"
    right_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSr_FZPCHrEKgnRVi-8DEfhe408FoT19me7ug&s"

    # Prefer local images added under data: 'images.jpeg' and 'images (1).jpeg'
    preferred_left = os.path.join(config.DATA_DIR, 'images.jpeg')
    preferred_right = os.path.join(config.DATA_DIR, 'images (1).jpeg')

    if os.path.exists(preferred_left):
        left_path = preferred_left
    elif os.path.exists(config.BRAIN_IMAGE):
        left_path = config.BRAIN_IMAGE
    else:
        left_path = left_url

    if os.path.exists(preferred_right):
        right_path = preferred_right
    else:
        # fallback: find any other image in data dir
        right_path = None
        if os.path.exists(config.DATA_DIR):
            for root, dirs, files in os.walk(config.DATA_DIR):
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                        candidate = os.path.join(root, f)
                        if os.path.abspath(candidate) not in (os.path.abspath(left_path), os.path.abspath(config.BRAIN_IMAGE)):
                            right_path = candidate
                            break
                if right_path:
                    break
        if not right_path:
            right_path = right_url

    c1, c2 = st.columns(2)
    with c1:
        try:
            st.image(left_path, caption="Brain scan overview", use_container_width=True)
        except Exception:
            st.write("")
    with c2:
        try:
            st.image(right_path, caption="Dataset sample", use_container_width=True)
        except Exception:
            st.write("")

    st.markdown("---")

if page == "Tabular Prediction":
    st.markdown("""
        <h1 style='color: #2e4057; margin-bottom: 1rem;'>📊 Clinical Features Assessment</h1>
        <p style='color: #666; font-size: 1.1rem;'>Enter patient clinical measurements for MCI prediction</p>
    """, unsafe_allow_html=True)

    # Enhanced input form with better styling
    with st.form("prediction_form", clear_on_submit=False):
        st.markdown("""
            <style>
            .stNumberInput > div > div > input {
                border-radius: 8px;
            }
            .stForm > div > div > div > div > div {
                gap: 1.5rem;
            }
            </style>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
            func_assess = st.number_input("Functional Assessment", 
                value=0, help="Patient's functional assessment score (0-100)")
            adl = st.number_input("ADL Score", 
                value=0, help="Activities of Daily Living score (0-100)")
            mmse = st.number_input("MMSE Score", 
                value=0, help="Mini-Mental State Examination score (0-30)")
            memory = st.number_input("Memory Complaints", 
                value=0, help="Frequency of memory-related complaints (0-10)")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
            behavioral = st.number_input("Behavioral Problems", 
                value=0, help="Severity of behavioral issues (0-10)")
            ldl = st.number_input("Cholesterol LDL", 
                value=0.0, help="LDL cholesterol level in mg/dL")
            diet = st.number_input("Diet Quality", 
                value=0, help="Diet quality assessment score (0-10)")
            triglycerides = st.number_input("Triglycerides", 
                value=0.0, help="Triglycerides level in mg/dL")
            st.markdown("</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("🔍 Analyze Patient Data")
        
        if submitted:
            with st.spinner("Analyzing patient data..."):
                try:
                    values = [
                        func_assess, adl, mmse, memory,
                        behavioral, ldl, diet, triglycerides
                    ]
                    prediction = utils.predict_from_values(model, values)
                    
                    # Animated prediction display
                    if prediction == 1:
                        st.markdown(
                            styled_card(
                                "Analysis Result",
                                "⚠️ Potential Alzheimer's Indicators Detected",
                                "#d72660",
                                "#ffffff"
                            ),
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            styled_card(
                                "Analysis Result",
                                "✅ No Significant Indicators Detected",
                                "#20bfa9",
                                "#ffffff"
                            ),
                            unsafe_allow_html=True
                        )
                        
                except Exception as e:
                    st.error(f"❌ Analysis Error: {str(e)}")

    # Add helpful information
    with st.expander("ℹ️ About Clinical Features"):
        st.markdown("""
        ### Understanding the Measurements
        
        - **Functional Assessment**: Overall patient functionality score
        - **ADL Score**: Ability to perform daily living activities
        - **MMSE Score**: Cognitive function assessment
        - **Memory Complaints**: Frequency of reported memory issues
        - **Behavioral Problems**: Severity of behavioral changes
        - **Cholesterol & Triglycerides**: Blood lipid measurements
        - **Diet Quality**: Assessment of nutritional habits
        
        All measurements are standardized for consistent analysis.
        """)

elif page == "Audio Prediction":
    st.title("MCI Detect — Audio Analysis")
    st.write("Upload or record an audio sample for prediction")

    # File upload for audio (moved from MRI page)
    audio_file = st.file_uploader("Upload Audio File", type=["wav", "flac"]) 

    if audio_file is not None:
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_file.getvalue())
                audio_path = tmp.name

            # Make prediction
            prediction = utils.predict_from_audio(model, audio_path)

            if prediction == 1:
                st.error("⚠️ Alzheimer's Detected (from audio)")
            else:
                st.success("✅ No Alzheimer's Detected (from audio)")
                
            # Clean up
            os.unlink(audio_path)
            
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")

    # Audio recording
    st.write("---")
    st.write("Or record audio:")
    
    if "recording" not in st.session_state:
        st.session_state.recording = False
        
    if not st.session_state.recording:
        if st.button("🎙️ Start Recording (90s)"):
            st.session_state.recording = True
            st.experimental_rerun()
    else:
        # Record for 90 seconds
        fs = 16000
        duration = 90
        st.warning(f"Recording for {duration} seconds...")
        
        try:
            audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
            sd.wait()
            
            # Save and predict
            audio_path = utils.save_recorded_audio(audio_data, fs)
            prediction = utils.predict_from_audio(model, audio_path)
            
            if prediction == 1:
                st.error("⚠️ Alzheimer's Detected (from recording)")
            else:
                st.success("✅ No Alzheimer's Detected (from recording)")
                
            # Clean up
            os.unlink(audio_path)
            
        except Exception as e:
            st.error(f"Error recording audio: {str(e)}")
        
        st.session_state.recording = False
    
elif page == "MRI Analysis":
    st.title("MCI Detect — MRI Analysis")
    st.write("Upload a brain MRI scan for analysis")
    
    # File upload for MRI
    mri_file = st.file_uploader("Upload MRI Scan", type=["jpg", "jpeg", "png"])
    
    if mri_file is not None:
        try:
            # Display uploaded image
            img = Image.open(mri_file)
            st.image(img, caption="Uploaded MRI Scan", width=400)
            
            # Save temporarily and predict
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                tmp.write(mri_file.getvalue())
                
            # Load classifier and predict
            from mri_model import AlzheimerMRIClassifier
            classifier = AlzheimerMRIClassifier()
            classifier.load('vgg_mri_model.pth')
            
            result = classifier.predict_image(tmp.name)
            
            # Show prediction with styled box
            prediction = result['class_name']
            confidence = result['confidence']
            
            # Color coding based on prediction
            if "Non" in prediction:
                box_color = "success"
            else:
                box_color = "error"
            
            st.markdown(f"""
            <div style='padding: 20px; border-radius: 10px; background-color: {'#20bfa9' if 'Non' in prediction else '#d72660'}; color: white;'>
                <h3>Prediction Result:</h3>
                <p style='font-size: 20px;'>{prediction}</p>
                <p>Confidence: {confidence:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Cleanup
            os.unlink(tmp.name)
            
        except Exception as e:
            st.error(f"Error analyzing MRI: {str(e)}")

elif page == "Video Prediction":
    st.title("MCI Detect — Video Analysis")
    st.write("Use your webcam for facial analysis")

    # Initialize facial analyzer if not exists
    if 'facial_analyzer' not in st.session_state:
        st.session_state.facial_analyzer = FacialAnalyzer()
        st.session_state.frame_count = 0
        st.session_state.last_prediction = None
        st.session_state.last_warnings = []

    # Create placeholder for video feed
    video_placeholder = st.empty()
    status_placeholder = st.empty()
    warning_placeholder = st.empty()
    result_placeholder = st.empty()

    # Add webcam input
    video_frame = st.camera_input("Webcam Feed")
    
    if video_frame is not None:
        bytes_data = video_frame.getvalue()
        
        # Convert to opencv format
        file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Analyze frame
        result, label, warnings = st.session_state.facial_analyzer.analyze_frame(frame)
        
        if result is not None:
            cls_idx, confidence, features = result
            
            # Update status
            status_text = f"Analyzing... Frame {st.session_state.frame_count}"
            status_placeholder.text(status_text)
            
            # Show warnings if any
            if warnings:
                warning_text = "⚠️ Suggestions:\n" + "\n".join([
                    "- Move closer" if w == "too_far" else
                    "- Face the camera directly" if w == "turn_head" else
                    "- Adjust head angle" if w == "tilt_head" else
                    "- Improve lighting" if w == "low_light" else w
                    for w in warnings
                ])
                warning_placeholder.warning(warning_text)
            else:
                warning_placeholder.empty()
            
            # Show prediction with confidence
            result_html = f"""
            <div style='padding: 20px; border-radius: 10px; background-color: {'#d72660' if cls_idx > 0 else '#20bfa9'}; color: white;'>
                <h3>Analysis Result:</h3>
                <p style='font-size: 20px;'>{label}</p>
                <p>Confidence: {confidence:.1%}</p>
                <p>Features:</p>
                <ul>
                    <li>Smile Index: {features['smile']:.3f}</li>
                    <li>Eye Openness: {features['eye_open']:.3f}</li>
                    <li>Brow Position: {features['brow']:.3f}</li>
                    <li>Expression Variance: {features['variance']:.3f}</li>
                </ul>
            </div>
            """
            result_placeholder.markdown(result_html, unsafe_allow_html=True)
            
            st.session_state.frame_count += 1
            st.session_state.last_prediction = result
            st.session_state.last_warnings = warnings
        else:
            status_placeholder.error("No face detected. Please ensure your face is visible in the camera.")
            result_placeholder.empty()