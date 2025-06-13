# GranaryPredict – Development Blueprint

## 0. Guiding Principles
- Start simple → iterate fast
- Maintain modular, testable code
- Keep data & credentials out of version control
- Prefer well-supported open-source tools

## 1. Repository Layout
```
granarypredict/       # Core python package
│
├─ data/              # Runtime data folder (ignored by git)
│   ├─ raw/           # Original CSV/XLSX dumps
│   └─ processed/     # Cleaned feature tables
│
├─ models/            # Serialized model artefacts (git-ignored)
│
├─ app/               # Streamlit dashboard code
│   ├─ __init__.py
│   └─ Dashboard.py
│
├─ notebooks/         # Jupyter exploration (optional, git-ignored)
│
├─ tests/             # Unit tests
│
└─ granarypredict/
    ├─ __init__.py
    ├─ config.py      # Paths & global settings
    ├─ ingestion.py   # API + file loaders
    ├─ cleaning.py    # Missing/outlier handling
    ├─ features.py    # Feature engineering helpers
    ├─ model.py       # Training / inference wrappers
    ├─ evaluate.py    # Metrics & validation
    └─ utils.py       # Generic helpers
```

## 2. Milestone Roadmap
1. Scaffolding
   - Create repo structure, gitignore, requirements, sample data stubs.
2. Data Ingestion
   - File loader (CSV/XLSX) supporting Chinese-encoded names
   - REST client for future meteorological API (placeholder)
3. Cleaning & Feature Engineering
   - Timestamp parsing, type coercion, label encoding for grain types
   - Spatial coordinates (grid_x, y, z) → flatten to single index & 3-D arrays
4. Baseline Modelling
   - Train/valid split by time
   - Gradient Boosting & RandomForest regressors as baseline
5. Model Evaluation & Persistence
   - MAE / RMSE metrics, cross-validation
   - Joblib serialization to `models/`
6. Interactive Dashboard (Streamlit)
   - Upload raw file → displays stats, 3-D grid (Plotly volume plot)
   - Predict future temps, show alerts if > threshold
7. CI & Tests
   - pytest covering utilities and model I/O
8. Packaging & Deployment
   - Dockerfile for reproducible runs
   - README with quick-start instructions

## 3. External Dependencies
- pandas, numpy, scikit-learn
- plotly, streamlit
- requests (API calls)
- joblib, tqdm

## 4. Security & Privacy
- `.env` for secrets, loaded via `python-dotenv`
- Never commit raw production data; use synthetic samples instead

## 5. Next Implementation Steps
1. Commit repo scaffolding & `requirements.txt`
2. Implement `granarypredict/config.py` with path helpers
3. Stub ingestion & cleaning modules with minimal working methods
4. Add Streamlit placeholder app that can be launched today 