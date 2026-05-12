@echo off
echo 🌾 Deploying Paddy Yield Predictor...

REM Create requirements.txt
echo streamlit==1.38.0 > requirements.txt
echo pandas==2.2.2 >> requirements.txt
echo scikit-learn==1.5.1 >> requirements.txt
echo numpy==2.1.1 >> requirements.txt

REM Git commands
git init
git add .
git commit -m "Initial: Paddy Yield Predictor v1.0"

REM Create GitHub repo link (replace YOUR_USERNAME)
set /p GITHUB_USER="Enter your GitHub username: "
git remote add origin https://github.com/rohinkumaryeturi-source/paddy-yield-predictor.git
git branch -M main
git push -u origin main

echo ✅ Repo created! Now deploy to Streamlit Cloud:
echo 1. Go to https://share.streamlit.io
echo 2. Sign in with GitHub
echo 3. Select your repo + app.py
echo 4. Deploy!
pause