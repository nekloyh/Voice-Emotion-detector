@echo off
title Voice Emotion Detector v2.0
cls
echo ============================================================
echo Voice Emotion Detector v2.0 - Clean Production
echo AI-Powered Emotion Recognition from Voice
echo ============================================================
echo.

rem Check if Python environment exists
echo [1/4] Checking Python environment...
if exist "vov-emo\python.exe" (
    echo [OK] Conda environment found
) else (
    echo [ERROR] Conda environment missing
    echo Please ensure vov-emo directory exists
    goto :error
)

echo.
echo [2/4] Checking model files...
if exist "model\config.json" (
    echo [OK] config.json found
) else (
    echo [ERROR] config.json missing
    goto :error
)

if exist "model\model.safetensors" (
    echo [OK] model.safetensors found ^(360.8 MB^)
) else (
    echo [ERROR] model.safetensors missing  
    goto :error
)

if exist "model\preprocessor_config.json" (
    echo [OK] preprocessor_config.json found
) else (
    echo [ERROR] preprocessor_config.json missing
    goto :error
)

if exist "model\training_args.bin" (
    echo [OK] training_args.bin found
) else (
    echo [ERROR] training_args.bin missing
    goto :error
)

echo.
echo [3/4] Checking application files...
if exist "app.py" (
    echo [OK] app.py found
) else (
    echo [ERROR] app.py missing
    goto :error
)

if exist "requirements.txt" (
    echo [OK] requirements.txt found
) else (
    echo [ERROR] requirements.txt missing
    goto :error
)

if exist ".streamlit\config.toml" (
    echo [OK] Streamlit config found
) else (
    echo [WARNING] Streamlit config missing ^(will use defaults^)
)

echo.
echo [4/4] Starting application...
echo.
echo Web Interface: http://localhost:8501
echo Features: File Upload + AI Emotion Analysis  
echo Model: Wav2Vec2 ^(94.5M parameters, 8 emotions^)
echo UI: Modern glass morphism design with color-coded emotions
echo Press Ctrl+C to stop
echo.

rem Use the conda environment to run streamlit
vov-emo\python.exe -m streamlit run app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false

goto :end

:error
echo.
echo [FAILED] Setup incomplete. Please fix the errors above.
echo.
echo Troubleshooting:
echo   - Ensure all model files are in the model/ directory
echo   - Check if conda environment vov-emo exists
echo   - Verify app.py and requirements.txt are present
echo   - Try running: python launch.py
echo.
pause
goto :end

:end
echo.
echo Voice Emotion Detector stopped.
pause
