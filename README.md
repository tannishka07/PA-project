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
APP LINK : https://pa-project-nz4stgu726ry2bq8rx4qym.streamlit.app 
