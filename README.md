# WattWise — Smart Home Energy Consumption Prediction

A Streamlit dashboard for the Predictive Analytics project **Smart Home Energy Consumption Prediction**, using Linear Regression as its only predictive model.

## Features
- Overview dashboard with dataset statistics and test metrics
- EDA: histogram/box plot, correlation heatmap, scatter plots and descriptive statistics
- Interactive energy consumption estimate form
- Chronological 80/20 train/test split
- Evaluation with MAE, RMSE and R²
- Coefficient visualization and CSV downloads
- Dataset, method and limitations documented in the app

## Dataset
Place `energydata_complete.csv` in the repository root. The target is `Appliances` (Wh); features include lights, indoor temperature/humidity, outdoor weather, and derived time features. `rv1` and `rv2` are excluded as uninterpretable random columns.

## Run locally
Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Open https://share.streamlit.io/ and choose New app.
2. Select `tannishka07/PA-project`, branch `main`, entrypoint `app.py`.
3. Deploy; packages install from `requirements.txt`.

The model sorts records by time, trains on the first 80%, and evaluates on the final 20%. Predictions are educational estimates from one household's historical data and may not generalize to other homes.