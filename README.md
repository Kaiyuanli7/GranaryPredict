# 🌾 GranaryPredict

GranaryPredict is a lightweight end-to-end pipeline that ingests sensor data from grain warehouses, cleans & enriches it, trains a predictive model, and serves forecasts through an interactive Streamlit dashboard.

## Features
1. Data ingestion from CSV or REST APIs
2. Data cleaning & missing-value handling
3. Feature engineering (spatial + temporal)
4. Baseline RandomForest temperature model
5. 3-D grid visualization of silo temperatures
6. Alerting when predicted temps exceed safe thresholds
7. Time-series cross-validation and choice between RandomForest or HistGradientBoosting models

## Project Structure
Refer to `docs/PLAN.md`

## Full Setup Guide (Windows cmd.exe)

1. Download / clone the repository
   ```cmd
   git clone https://github.com/your-org/o3Granary.git
   cd o3Granary
   ```

2. Create a fresh virtual environment (uses Python 3.11 path—adjust if different)
   ```cmd
   "C:\Users\<you>\AppData\Local\Programs\Python\Python311\python.exe" -m venv .venv
   ```

3. Activate the virtual environment
   ```cmd
   .venv\Scripts\activate.bat        :: cmd.exe
   ````powershell
   .\.venv\Scripts\Activate.ps1      # PowerShell
   ```
   Your prompt will now start with `(.venv)`.

4. Upgrade pip and install project requirements (Tsinghua mirror used as example)
   ```cmd
   python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
   pip install -r requirements.txt     -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

5. Ensure Python can find the local `granarypredict/` package:
   ```cmd
   "pip install -e ." inside the virtual env
   :: close & reopen cmd, then `activate` again
   ```

6. (Optional) Generate a synthetic sensor CSV for testing
   ```cmd
   python scripts\generate_fake_sensor_data.py
   :: writes data\raw\synthetic_sensor_data.csv
   ```

7. Launch the Streamlit dashboard
   ```cmd
   streamlit run app\Dashboard.py
   ```
   Open the displayed URL (usually http://localhost:8501). Use the sidebar to upload a CSV, choose a model, adjust
   "Days ahead to predict", and train/evaluate.

### PowerShell differences
• Use `Activate.ps1` instead of `activate.bat` to enable the venv.  
• If scripts are blocked, run `Set-ExecutionPolicy Bypass -Scope Process -Force` once per session.

### Linux / macOS
Replace the activation command with `source .venv/bin/activate` and omit the `.exe` paths; the rest is the same.