# 🛢️ SiloFlow

SiloFlow is a lightweight end-to-end pipeline that ingests sensor data from grain warehouses, cleans & enriches it, trains a predictive model, and serves forecasts through an interactive Streamlit dashboard.

> **What's New (May 2025)**  
> • Project renamed to **SiloFlow** (imports remain backward-compatible via an alias).  
> • Supports the **StorePoint** CSV export format  
> • Cascaded *Warehouse → Silo* selectors in the UI  
> • Robust metric handling when some rows lack ground-truth temperatures  
> • Utility script `scripts/fix_training_csv.py` to strip errant commas in legacy dumps

## Features
1. Data ingestion from CSV or REST APIs
2. Data cleaning & missing-value handling
3. Feature engineering (spatial + temporal)
4. Baseline RandomForest temperature model
5. 3-D grid visualization of silo temperatures
6. Alerting when predicted temps exceed safe thresholds
7. Time-series cross-validation and choice between RandomForest or HistGradientBoosting models
8. Per-warehouse / per-silo evaluation & forecasts

## Project Structure
```
granarypredict/       # Core Python package (ingestion, cleaning, features, modelling)
app/                  # Streamlit dashboard – launch with `streamlit run app/Dashboard.py`
data/
  ├─ preloaded/       # Sample CSVs you can try immediately
  ├─ raw/             # Original dumps (unchanged)
  └─ processed/       # Cleaned CSVs (output of `fix_training_csv.py`)
models/               # Saved models (.joblib) – auto-created
scripts/              # Helper utilities (CSV fixer, synthetic data generator …)
```

## Full Setup Guide (Windows cmd.exe)

1. Download / clone the repository
   ```cmd
   git clone -b master https://github.com/kaiyaunli7/siloflow
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
   The sidebar lets you:
   1. Upload a CSV (or pick a bundled sample)
   2. Train / retrain a model (choose algorithm + iterations)
   3. **Select Warehouse → Silo** to focus on a single silo
   4. Evaluate (back-test) or Forecast (future) with adjustable horizons

## Working with StorePoint CSVs

The latest export format has the header:

```
storepoint_id,storepointName,kdjd,kdwd,kqdz,storeId,storeName,locatType,line_no,layer_no,batch,temp,x,y,z,avg_in_temp,max_temp,min_temp,indoor_temp,indoor_humidity,outdoor_temp,outdoor_humidity,storeType
```

`granarypredict.ingestion.standardize_result147` automatically renames these fields to the internal names used by the pipeline:

* `storepointName` → `granary_id`  (warehouse)
* `storeName` → `heap_id`  (silo)
* `kdjd` / `kdwd` → `longitude` / `latitude`
* `kqdz` → `address_cn`

So you don't need to alter the CSV – just upload it.

Then upload the fixed file.

### PowerShell differences
• Use `Activate.ps1` instead of `activate.bat` to enable the venv.  
• If scripts are blocked, run `Set-ExecutionPolicy Bypass -Scope Process -Force` once per session.

### Linux / macOS
Replace the activation command with `source .venv/bin/activate` and omit the `.exe` paths; the rest is the same.

CSV's should come with the format of 
storepoint_id,storepointName,storeId,storeName,locatType,line_no,layer_no,batch,temp,x,y,z,avg_in_temp,max_temp,min_temp,indoor_temp,indoor_humidity,outdoor_temp,outdoor_humidity,storeType

storepoint_id =  d