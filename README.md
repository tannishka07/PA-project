# WattWise — Smart Home Energy Consumption Prediction

A Streamlit dashboard for the Predictive Analytics project **Smart Home Energy Consumption Prediction**. The only predictive model used is **ordinary Linear Regression**, aligned with the course syllabus.

## Dashboard features

- Overview with dataset and test-performance metrics
- Exploratory data analysis: histograms, box plots, summary statistics, Z-scores, scatter plots, and correlation heatmap
- Interactive what-if estimator for appliance energy use
- Chronological 80/20 train/test split (no random mixing of future rows into training)
- MAE, RMSE, and R² evaluation on the held-out final time period
- Actual-vs-predicted time series and coefficient visualization
- CSV downloads for coefficients, descriptive statistics, and test predictions
- Syllabus coverage and model limitations

## Dataset

Dataset: [UCI Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction), collected at approximately 10-minute intervals. It has 19,735 observations and the prediction target is Appliances energy use (Wh).

The app uses `energydata_complete.csv` beside `app.py` when the CSV is present. If it is absent, the app attempts to load the public UCI copy automatically. You can also upload the CSV using the sidebar file uploader. The random columns `rv1` and `rv2` are excluded as predictors.

## Run locally

Python 3.10+ is recommended.

```bash
git clone https://github.com/tannishka07/PA-project.git
cd PA-project
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Open [Streamlit Community Cloud](https://share.streamlit.io/).
2. Choose **New app** and connect your GitHub account if prompted.
3. Select repository `tannishka07/PA-project`, branch `main`, entrypoint `app.py`.
4. Deploy. Streamlit installs packages from `requirements.txt`.

The repository intentionally does not store the 11+ MB CSV; the app uses the public UCI data URL by default, or you can upload your own compatible CSV in the app sidebar.

## Methodology

- Parse and sort timestamps chronologically.
- Derive hour, day of week, month, and weekend features.
- Train Linear Regression on the first 80% of records and evaluate on the final 20%.
- Report MAE, RMSE, and R² without altering the model's raw predictions.

On the attached dataset, this implementation produced approximately **MAE 50.46 Wh, RMSE 86.07 Wh, and R² 0.106** on the chronological holdout. Treat these as baseline results; the score indicates that a simple linear model captures only a limited part of the variation in appliance use.

## Limitations and responsible use

The data represents one home over about 4.5 months. Results may not generalize to other households or seasons. The form is a what-if estimator that needs sensor readings for the conditions being evaluated; it is not a forecast based only on future date/time. Fine-grained household usage can reveal occupancy patterns, so handle raw data responsibly.
