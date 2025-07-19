"""
Voice Emotion Detector v1.0 - Clean Production Version
Simple, reliable voice emotion detection using Wav2Vec2
Focus: Core functionality, performance, and reliability
"""

import streamlit as st
import librosa
import numpy as np
import tempfile
import os
import warnings

# Import transformers for the model
from transformers import Wav2Vec2ForSequenceClassification
import torch

warnings.filterwarnings('ignore')

# Configuration
MODEL_PATH = "./model"
EMOTIONS = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
EMOTION_EMOJIS = {
    "angry": "😠", "calm": "😌", "disgust": "🤢", "fearful": "😨",
    "happy": "😊", "neutral": "😐", "sad": "😢", "surprised": "😲"
}
EMOTION_COLORS = {
    "angry": "#ff4757",      # Đỏ rõ ràng
    "calm": "#2ed573",       # Xanh lá nhạt
    "disgust": "#ff6b35",    # Cam
    "fearful": "#5352ed",    # Tím xanh
    "happy": "#ffc048",      # Vàng
    "neutral": "#747d8c",    # Xám
    "sad": "#3742fa",        # Xanh dương
    "surprised": "#ff3838"   # Đỏ hồng
}

@st.cache_resource
def load_model():
    """Load the pre-trained emotion detection model"""
    try:
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            use_safetensors=True
        )
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.info("Please ensure model files are in the './model' directory")
        return None

def preprocess_audio(audio_file):
    """Preprocess audio file for the model"""
    try:
        # Read audio file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_file_path = tmp_file.name
        
        # Load audio with librosa
        audio, sample_rate = librosa.load(tmp_file_path, sr=16000, mono=True)
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        # Validate audio
        if len(audio) == 0:
            raise ValueError("Empty audio file")
        
        if len(audio) < 1600:  # Less than 0.1 seconds
            raise ValueError("Audio too short (minimum 0.1 seconds)")
            
        if len(audio) > 320000:  # More than 20 seconds
            audio = audio[:320000]  # Truncate to 20 seconds
        
        return audio, sample_rate
        
    except Exception as e:
        raise Exception(f"Audio preprocessing failed: {str(e)}")

def predict_emotion(audio, model):
    """Predict emotion from audio"""
    try:
        # Convert to tensor
        inputs = torch.tensor(audio).unsqueeze(0).float()
        
        # Model prediction
        with torch.no_grad():
            outputs = model(inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get probabilities
        probabilities = predictions.cpu().numpy()[0]
        
        # Get predicted emotion
        predicted_idx = np.argmax(probabilities)
        predicted_emotion = EMOTIONS[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        return predicted_emotion, confidence, probabilities
        
    except Exception as e:
        raise Exception(f"Emotion prediction failed: {str(e)}")

def main():
    """Main application"""
    st.set_page_config(
        page_title="Voice Emotion Detector",
        page_icon="🎭",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS with modern design
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #ff6b6b 50%, #4ecdc4 75%, #45b7d1 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .main-header {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #ff6b6b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    .section-card {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .upload-zone {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 2px dashed #667eea;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-zone:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        border-color: #764ba2;
        transform: translateY(-2px);
    }
    
    .emotion-result {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #ff6b6b 100%);
        border-radius: 20px;
        color: white;
        margin: 1.5rem 0;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
    }
    
    .emotion-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .emotion-name {
        font-size: 2rem;
        font-weight: 600;
        margin: 1rem 0;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .confidence-score {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 400;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.6);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.6);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .stAudio {
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🎭 Voice Emotion Detector</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-powered emotion recognition from voice recordings</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # File upload section
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 📁 Upload Audio File")
    
    st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
    st.markdown("**Drop your audio file here or click to browse**")
    st.markdown("*Supports: WAV, MP3, FLAC, M4A, OGG • Max: 20 seconds*")
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac', 'm4a', 'ogg'],
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # File info card
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            st.markdown(f"**📄 {uploaded_file.name}**")
        with col2:
            st.markdown(f"*{uploaded_file.size:,} bytes*")
        with col3:
            st.markdown("🎵")
        
        # Audio player
        st.audio(uploaded_file, format='audio/wav')
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process button
        if st.button("🎯 Analyze Emotion", type="primary", use_container_width=True):
            with st.spinner("🔄 Processing audio..."):
                try:
                    # Preprocess audio
                    audio, sample_rate = preprocess_audio(uploaded_file)
                    
                    # Predict emotion
                    predicted_emotion, confidence, probabilities = predict_emotion(audio, model)
                    
                    # Display main result
                    emoji = EMOTION_EMOJIS.get(predicted_emotion, "🎭")
                    st.markdown(f"""
                    <div class="emotion-result">
                        <span class="emotion-icon">{emoji}</span>
                        <div class="emotion-name">{predicted_emotion.title()}</div>
                        <div class="confidence-score">{confidence:.1%} confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detailed scores in compact format
                    st.markdown('<div class="section-card">', unsafe_allow_html=True)
                    st.markdown("### 📊 Emotion Scores")
                    
                    # Create two columns for compact layout
                    col1, col2 = st.columns(2)
                    
                    for i, emotion in enumerate(EMOTIONS):
                        score = float(probabilities[i])
                        emoji_icon = EMOTION_EMOJIS.get(emotion, "🎭")
                        emotion_color = EMOTION_COLORS.get(emotion, "#747d8c")
                        is_predicted = emotion == predicted_emotion
                        
                        # Alternate between columns
                        target_col = col1 if i % 2 == 0 else col2
                        
                        with target_col:
                            # Emotion label
                            label_text = f"{emoji_icon} **{emotion.title()}**"
                            if is_predicted:
                                label_text += " ⭐"
                            st.markdown(label_text)
                            
                            # Custom progress bar with specific color
                            progress_html = f"""
                            <div style="width: 100%; background-color: #f1f5f9; height: 20px; border-radius: 10px; overflow: hidden; margin: 5px 0;">
                                <div style="width: {score*100}%; height: 100%; background-color: {emotion_color}; border-radius: 10px; display: flex; align-items: center; justify-content: flex-end; padding-right: 8px; box-sizing: border-box;">
                                    <span style="color: white; font-weight: bold; font-size: 12px; text-shadow: 1px 1px 2px rgba(0,0,0,0.7);">{score:.0%}</span>
                                </div>
                            </div>
                            """
                            st.markdown(progress_html, unsafe_allow_html=True)
                            
                            # Add some spacing
                            st.markdown("")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Audio info in compact cards
                    st.markdown('<div class="section-card">', unsafe_allow_html=True)
                    st.markdown("### ℹ️ Audio Information")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size: 1.5rem; color: #667eea;">⏱️</div>
                            <div style="font-weight: 600;">{len(audio)/sample_rate:.2f}s</div>
                            <div style="color: #64748b; font-size: 0.9rem;">Duration</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size: 1.5rem; color: #667eea;">📡</div>
                            <div style="font-weight: 600;">{sample_rate} Hz</div>
                            <div style="color: #64748b; font-size: 0.9rem;">Sample Rate</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size: 1.5rem; color: #667eea;">🔢</div>
                            <div style="font-weight: 600;">{len(audio):,}</div>
                            <div style="color: #64748b; font-size: 0.9rem;">Samples</div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Processing failed: {str(e)}")
                    st.info("💡 Try uploading a different audio file or check file format")
    
    else:
        # Instructions in compact format
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 🚀 Quick Start")
        
        steps = [
            "📤 **Upload** an audio file (WAV, MP3, FLAC, M4A, OGG)",
            "🎧 **Preview** your audio with the built-in player", 
            "🎯 **Analyze** emotion with one click",
            "📊 **View** results and confidence scores"
        ]
        
        for step in steps:
            st.markdown(f"• {step}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Tips and model info in expandable sections
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("💡 Tips for Best Results"):
                st.markdown("""
                - **Clear speech** with minimal background noise
                - **3-10 seconds** duration works optimally  
                - **Higher quality** audio = better accuracy
                - **English speech** recommended
                """)
        
        with col2:
            with st.expander("🤖 Model Details"):
                st.markdown("""
                - **Wav2Vec2** architecture
                - **94.5M** parameters
                - **8 emotions** detected
                - **SafeTensors** format
                """)
    
    # Close main container
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
