#!/usr/bin/env python3
"""
Voice Emotion Detector v2.0 - Clean Production Launcher
Professional launcher with comprehensive health checks
"""

import os
import sys
import subprocess
import importlib.util

class EmotionDetectorLauncher:
    """Professional launcher for Voice Emotion Detector"""
    
    def __init__(self):
        self.required_packages = [
            'streamlit', 'torch', 'transformers', 
            'librosa', 'numpy', 'soundfile'
        ]
        self.model_files = [
            'model/config.json',
            'model/model.safetensors', 
            'model/preprocessor_config.json'
        ]
        
    def print_header(self):
        """Print application header"""
        print("=" * 60)
        print("🎭 Voice Emotion Detector v2.0 - Clean Production")
        print("   AI-Powered Emotion Recognition from Voice")
        print("=" * 60)
        print()
        
    def check_dependencies(self):
        """Check if all required packages are installed"""
        print("📦 [1/3] Checking dependencies...")
        missing = []
        
        for package in self.required_packages:
            if importlib.util.find_spec(package) is None:
                missing.append(package)
        
        if missing:
            print("❌ Missing dependencies:")
            for dep in missing:
                print(f"   - {dep}")
            print("\n💡 Install with: pip install -r requirements_clean.txt")
            return False
        else:
            print("✅ All dependencies installed")
            return True
    
    def check_model_files(self):
        """Check if model files exist"""
        print("\n🤖 [2/3] Checking model files...")
        missing = []
        
        for file_path in self.model_files:
            if not os.path.exists(file_path):
                missing.append(file_path)
        
        if missing:
            print("❌ Missing model files:")
            for file in missing:
                print(f"   - {file}")
            print("\n⚠️  Model files are required for emotion detection!")
            return False
        else:
            print("✅ All model files found")
            
            # Show model info
            model_size = os.path.getsize('model/model.safetensors')
            print(f"   Model size: {model_size / (1024*1024):.1f} MB")
            return True
    
    def launch_application(self):
        """Launch the Streamlit application"""
        print("\n🚀 [3/3] Starting application...")
        print("🌐 Web Interface: http://localhost:8501")
        print("🎯 Features: File Upload + AI Emotion Analysis")
        print("🤖 Model: Wav2Vec2 (94.5M parameters, 8 emotions)")
        print("⏹️  Press Ctrl+C to stop")
        print()
        
        try:
            # Launch streamlit
            cmd = [
                sys.executable, '-m', 'streamlit', 'run', 'app.py',
                '--server.port', '8501',
                '--server.headless', 'false',
                '--browser.gatherUsageStats', 'false'
            ]
            
            subprocess.run(cmd, check=True)
            
        except KeyboardInterrupt:
            print("\n\n👋 Application stopped by user")
            return 0
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Failed to start application: {e}")
            return 1
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            return 1
    
    def run(self):
        """Main launcher workflow"""
        self.print_header()
        
        # Health checks
        if not self.check_dependencies():
            return 1
            
        if not self.check_model_files():
            return 1
        
        # Launch application
        return self.launch_application()

def main():
    """Entry point"""
    launcher = EmotionDetectorLauncher()
    return launcher.run()

if __name__ == "__main__":
    sys.exit(main())
