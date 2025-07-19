# 🎭 Voice Emotion Detector v1.0

## Clean Production-Ready AI-Powered Emotion Recognition

### ⚡ Quick Start

```bash
# Method 1: Using launcher (Recommended)
python launch.py

# Method 2: Direct run
python vov-emo\python.exe -m streamlit run app.py

# Method 3: Windows batch file
run.bat
```

**Access**: <http://localhost:8501>

### 🎯 Features

- **Modern UI**: Glass morphism design with animated gradient background
- **File Upload**: Drag & drop support for WAV, MP3, FLAC, M4A, OGG
- **AI Analysis**: Wav2Vec2 model with 94.5M parameters  
- **8 Emotions**: angry, calm, disgust, fearful, happy, neutral, sad, surprised
- **Color-Coded Results**: Each emotion has distinct color for easy identification
- **Real-time Processing**: Fast CPU inference (~1-2 seconds)
- **Responsive Design**: Works on desktop and mobile browsers

### 🎨 UI Highlights

- **Dynamic Background**: 5-color gradient with smooth animation
- **Glass Morphism**: Transparent cards with backdrop blur effects
- **Custom Progress Bars**: Color-coded bars for each emotion
- **Professional Typography**: Inter font for modern appearance
- **Intuitive Layout**: Clean 2-column emotion display

### 📁 Project Structure

``` bash
Voice-Emotion-detector/
├── app.py              # Main Streamlit application
├── launch.py           # Production launcher with health checks
├── run.bat             # Windows batch launcher
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
├── .streamlit/         # Streamlit configuration
│   └── config.toml     # Custom theme settings
├── model/              # AI model files (360MB)
│   ├── config.json
│   ├── model.safetensors
│   └── training_args.bin
├── vov-emo/           # Conda virtual environment
└── final_backup/      # Backup of removed duplicate files
```

### 🤖 Model Information

- **Architecture**: Wav2Vec2ForSequenceClassification
- **Parameters**: 94,570,632 parameters
- **Input**: 16kHz mono audio, max 20 seconds
- **Output**: 8 emotion classes with confidence scores
- **Format**: SafeTensors optimization for fast loading
- **Model Size**: 360.8 MB

### 🎨 Emotion Color Scheme

| Emotion | Color | Hex Code |
|---------|-------|----------|
| 😠 Angry | Red | `#ff4757` |
| 😌 Calm | Light Green | `#2ed573` |
| 🤢 Disgust | Orange | `#ff6b35` |
| 😨 Fearful | Purple Blue | `#5352ed` |
| 😊 Happy | Yellow | `#ffc048` |
| 😐 Neutral | Gray | `#747d8c` |
| 😢 Sad | Blue | `#3742fa` |
| 😲 Surprised | Pink Red | `#ff3838` |

### �💡 Usage Tips

- **Audio Quality**: Clear speech, minimal background noise
- **Duration**: 3-10 seconds optimal (max 20 seconds)
- **Format**: WAV recommended for best results
- **Language**: Optimized for English speech
- **Environment**: Quiet room for better accuracy

### 🔧 Technical Requirements

- **Python**: 3.8+ (Conda environment included)
- **RAM**: 2GB minimum, 4GB recommended
- **Storage**: 1GB total (500MB for model + dependencies)
- **CPU**: Modern processor (GPU optional, CPU inference)
- **OS**: Windows (tested), Linux/macOS compatible

### 📊 Performance Metrics

- **Accuracy**: High accuracy on clear speech samples
- **Processing Speed**: 1-2 seconds per audio file
- **Memory Usage**: ~1GB RAM during inference
- **Model Loading**: ~2-3 seconds initial load
- **Supported Audio**: Up to 20 seconds, 16kHz auto-conversion

### 🚀 Recent Updates (v2.0)

- ✅ **Complete UI Redesign**: Modern glass morphism interface
- ✅ **Color-Coded Emotions**: Distinct colors for each emotion
- ✅ **Project Cleanup**: Removed duplicate files and optimized structure
- ✅ **Enhanced Launcher**: Professional launcher with health checks
- ✅ **Improved Performance**: Faster loading and processing
- ✅ **Better Documentation**: Updated README with comprehensive guide

### 🛠️ Development

```bash
# Project was cleaned up and optimized
# All duplicate files removed
# Production-ready codebase
# Backup files saved in final_backup/
```

### 🎯 Use Cases

- **Voice Analysis**: Analyze emotional tone in recordings
- **Research**: Study emotional patterns in speech
- **Education**: Demonstrate AI emotion recognition
- **Personal**: Check emotional state in voice messages
- **Development**: Base for emotion-aware applications

---

**Version**: 1.0 Production Clean  
**Last Updated**: July 19, 2025  
**Status**: ✅ Production Ready & Optimized  
**License**: Open Source  
**Maintainer**: Voice Emotion Detection Team
