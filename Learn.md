Multi-Sensor, Spatio-Temporal Modelling is Essential
Traditional single-sensor or layer-average forecasts ignore cross-location heat transfer inside the silo.
Both “Temperature Forecasting … GCN + Transformer” and the Broad-Learning-Network (BLN) paper stress that spatial adjacency and temporal dynamics must be modelled jointly to reach sub-1 °C MAE.
Graph Convolutional Networks (GCN) Capture Spatial Topology Best
Represent every probe as a node; connect edges by physical proximity (or learned heat-transfer weights).
GCN layers aggregate neighbour information, letting the model learn how heat diffuses through grain.
Compared with CNN/fully-connected features, GCN reduced RMSE by ≈ 12 % in the Shaanxi-granary study.
Transformers (or Attention) Handle Long-Range Temporal Patterns
Grain piles warm slowly; correlations span dozens of days.
Self-attention in Transformers sees the entire sequence at once, beating RNN/LSTM on long-horizon accuracy.
The hybrid “GCN + Transformer” (called CGTN in the paper) achieved MAE ≈ 0.38 °C at 7-day horizon, vs 0.55 °C for plain LSTM.
Multi-Output Forecasting Greatly Improves Efficiency
Predict all sensors simultaneously -> 1 model inference instead of N.
Allows consistent cross-sensor patterns (temperature field) and enables post-forecast 3-D interpolation of unsensored regions.
Broad Learning Network (Improved BLN) is a Fast, Competitive Baseline
BLN incrementally widens rather than deepens, giving shallow, highly parallel networks.
Training time ≪ deep nets while matching or beating classical ML (RF/XGB).
Good choice when GPUs are unavailable.
Interpolating a Continuous Temperature Field Adds Practical Value
After forecasting discrete probes, apply trilinear/cubic interpolation to visualise hot-spots in unsensored voxels.
Operators can act on a complete heat map, not isolated point curves.
Recommended upgrade path for SiloFlow
Data Pre-processing
Build an adjacency matrix once per silo (k-nearest sensors in 3-D Euclidean space).
Produce sequences of (T days × sensors × features) windows, centred on each forecasting target day.
Modelling – two solid options
CGTN (GCN encoder ➜ Transformer decoder)
PyTorch Geometric for GCN, PyTorch Lightning for training loops.
Output: tensor [sensors, horizon] so you keep multi-output benefit.
Improved BLN (fast CPU baseline)
scikit-bln implementation; widen until validation MAE plateaus.
Training regimen
Sliding-window dataset with 75 % train / 25 % time-ordered validation.
Huber loss (robust to rare hot-spots).
Early stopping on ∆MAE < 0.001 for 5 epochs.
Integration with SiloFlow
Wrap the model in a simple predict(X) interface; the dashboard code already aligns feature columns.
Save with torch.save() or joblib; add a new radio option “GCN-Transformer” / “BLN”.
Hardware expectations
CGTN: ~1 M parameters, < 5 s per epoch on a modern GPU; inference < 50 ms per silo.
BLN: < 1 s training on CPU, sub-millisecond inference.
Bottom line
For highest accuracy and interpretable heat-maps, move toward a GCN + Transformer hybrid; keep BLN as a fast-training, CPU-friendly fallback.

1. Forecasting granularity
Current SiloFlow
Trains one model on a flattened feature table.
Prediction target = temperature of a single probe at each row.
Multi-sensor insight is obtained only after prediction (aggregating by grid_x/y/z).
Papers (GCN+Transformer, BLN)
Model outputs a vector: temperatures for all probes (or a 3-D field) in one shot.
Allows spatially consistent forecasts and direct 3-D interpolation of unsensored voxels.
Benefit over current model → higher operational efficiency (one forward pass) and a continuous temperature field rather than isolated point curves.
2. Spatial modelling capability
Current SiloFlow
Spatial coordinates are encoded as integers plus the engineered grid_index string.
Tree models treat them as categorical/ordinal features—no explicit notion of neighbour influence.
Papers
Graph Convolutional Network (GCN): nodes = sensors, edges = physical proximity; convolution aggregates neighbour heat, imitating real conduction.
Graph approach reduced RMSE ≈ 12 % vs models that ignore topology.
Benefit → captures cross-probe heat diffusion that trees miss.
3. Temporal modelling capability
Current SiloFlow
Relies on trees (RF / HistGB / LightGBM) fed with lagged features (month_sin, hour_cos, etc.).
Trees see only the lags you explicitly supply; long-range patterns require many manual lags and still dilute.
Papers
Transformer (self-attention) or BLN learn long- and short-term dependencies automatically across 30 + day windows.
Attention attends to any earlier time-step without vanishing-gradient issues of RNN/LSTM.
Benefit → noticeably lower MAE on 7--30-day horizons.
4. Accuracy vs speed
Current
Training: seconds (CPU). Inference: milliseconds / row.
MAE (real datasets) ~0.7–1.2 °C at 5-day horizon.
GCN + Transformer
Training: minutes on GPU (1 M params). Inference: < 50 ms for a whole silo sequence.
Reported MAE 0.38 °C (7 days) — roughly half the error.
Improved BLN
Training: seconds on CPU; inference: sub-ms.
Accuracy between LightGBM and CGTN—cheaper way to gain ≈ 20 % improvement over RF.
5. Interpretability
Current
Feature importance from trees, easy to explain to operators.
GCN + Transformer
Harder but still possible: attention heat-maps + GCN gradient attribution show which probes/time-steps mattered.
6. Implementation overhead
Current
Pure scikit-learn / LightGBM — already running in Streamlit.
GCN + Transformer
Requires PyTorch + PyTorch Geometric; add adjacency-matrix pre-computation; GPU recommended.
BLN
Only an extra pip wheel (scikit-bln) and a small training wrapper.
Summary
| Aspect | SiloFlow RF/HistGB/LGBM | GCN + Transformer (papers) | Broad Learning Net (papers) |
|-----------------------|-------------------------|---------------------------|-----------------------------|
Spatial reasoning	Implicit, weak	Explicit graph	None
Output	One probe per row	All probes at once	Configurable
Typical MAE (7 d)	0.7–1.2 °C	~0.4 °C	~0.55 °C
Train time (CPU)	seconds	minutes (GPU)	seconds
Integration effort	None (existing)	High (PyTorch GNN)	Low-medium
So, the research points to two upgrades:
Quick win: add a Broad-Learning-Network trainer—CPU friendly, easy code, ≈ 20 % accuracy gain.
Long-term: implement the GCN + Transformer hybrid for best accuracy and full-silo temperature field visualisation.
Both approaches can coexist alongside the current tree-ensembles, letting the operator pick the right trade-off.