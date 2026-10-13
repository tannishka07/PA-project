from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(page_title="WattWise | Home Energy Analytics", page_icon="⚡", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}.stApp{background:#f5f7fb}
.block-container{padding-top:1.5rem;max-width:1440px}h1,h2,h3{font-family:'Space Grotesk',sans-serif;color:#13294b}
.hero{background:linear-gradient(120deg,#102b50,#194e80 60%,#277ba8);padding:2rem 2.2rem;border-radius:22px;color:white;margin-bottom:1.3rem}
.hero h1{color:white;margin:0 0 .4rem;font-size:2.35rem}.hero p{color:#dbeafe;margin:0;max-width:780px}
.eyebrow{text-transform:uppercase;font-size:.72rem;letter-spacing:.13em;color:#a9d7ff;font-weight:700;margin-bottom:.5rem}
.metric-card{background:white;padding:1.1rem 1.2rem;border:1px solid #e3eaf3;border-radius:16px;min-height:105px}
.metric-label{color:#63748b;font-size:.83rem;margin-bottom:.35rem}.metric-value{color:#142d50;font-family:'Space Grotesk',sans-serif;font-size:1.8rem;font-weight:700}
.metric-note,.muted{color:#718198;font-size:.8rem}.section-note{color:#60718a;margin-top:-.4rem;margin-bottom:1rem}
div[data-testid="stSidebar"]{background:#eef3f9}
</style>""", unsafe_allow_html=True)

DATA_PATH = Path(__file__).parent / "energydata_complete.csv"
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv"
TARGET = "Appliances"
DROP = ["date", "rv1", "rv2"]

@st.cache_data(show_spinner=False)
def load_data(uploaded=None):
    if uploaded is not None:
        uploaded.seek(0)
        data = pd.read_csv(uploaded)
    elif DATA_PATH.exists():
        data = pd.read_csv(DATA_PATH)
    else:
        data = pd.read_csv(DATA_URL)
    if not {"date", TARGET}.issubset(data.columns):
        raise ValueError("CSV must contain date and Appliances columns.")
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    return data.dropna(subset=["date", TARGET]).sort_values("date").reset_index(drop=True)

def build_features(data):
    x = data.drop(columns=[c for c in DROP if c in data.columns], errors="ignore").drop(columns=[TARGET], errors="ignore").copy()
    x["hour"] = data["date"].dt.hour
    x["day_of_week"] = data["date"].dt.dayofweek
    x["month"] = data["date"].dt.month
    x["is_weekend"] = (data["date"].dt.dayofweek >= 5).astype(int)
    return x.select_dtypes(include=np.number).replace([np.inf, -np.inf], np.nan).fillna(0)

@st.cache_resource(show_spinner="Training Linear Regression…")
def train_model(fingerprint, data):
    x, y = build_features(data), data[TARGET].astype(float)
    split = int(len(data) * .8)
    if split < 2 or split >= len(data):
        raise ValueError("Dataset is too small for an 80/20 split.")
    model = LinearRegression().fit(x.iloc[:split], y.iloc[:split])
    pred = model.predict(x.iloc[split:])
    return {"model":model, "features":list(x.columns), "split":split,
      "actual":y.iloc[split:].reset_index(drop=True), "pred":pd.Series(pred),
      "dates":data["date"].iloc[split:].reset_index(drop=True),
      "mae":mean_absolute_error(y.iloc[split:],pred),
      "rmse":np.sqrt(mean_squared_error(y.iloc[split:],pred)),
      "r2":r2_score(y.iloc[split:],pred),
      "coef":pd.DataFrame({"Feature":x.columns,"Coefficient":model.coef_}).assign(
          Abs=lambda d:d["Coefficient"].abs()).sort_values("Abs",ascending=False)}

def card(label,value,note):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',unsafe_allow_html=True)

def header(title,subtitle):
    st.markdown(f"## {title}")
    st.markdown(f'<div class="section-note">{subtitle}</div>',unsafe_allow_html=True)

st.sidebar.markdown("## ⚡ WattWise")
st.sidebar.caption("Smart Home Energy Analytics")
page = st.sidebar.radio("NAVIGATION",["Overview","Explore the Data","Predict Consumption","Model Insights","About the Project"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Main algorithm:** Linear Regression")
st.sidebar.caption("Chronological 80/20 split · MAE · RMSE · R²")
uploaded = st.sidebar.file_uploader("Upload dataset (optional)",type=["csv"],help="Uses the local CSV if present, otherwise downloads the public UCI dataset.")
try:
    df = load_data(uploaded)
    fingerprint = f"{len(df)}-{df['date'].min()}-{df['date'].max()}-{int(df[TARGET].sum())}"
    result = train_model(fingerprint,df)
except Exception as exc:
    st.markdown('<div class="hero"><div class="eyebrow">PREDICTIVE ANALYTICS PROJECT</div><h1>WattWise</h1><p>Smart Home Energy Consumption Prediction with Linear Regression.</p></div>',unsafe_allow_html=True)
    st.error(f"Could not load dataset or train model: {exc}")
    st.info("Place energydata_complete.csv beside app.py, upload it from the sidebar, or check internet access for the UCI dataset fallback.")
    st.stop()

st.markdown('<div class="hero"><div class="eyebrow">PREDICTIVE ANALYTICS PROJECT</div><h1>WattWise</h1><p>Explore household energy patterns and estimate appliance consumption with transparent Linear Regression.</p></div>',unsafe_allow_html=True)

if page == "Overview":
    header("Energy at a glance","Dataset overview and performance on the final 20% of records, held out by time.")
    a,b,c,d=st.columns(4)
    with a: card("OBSERVATIONS",f"{len(df):,}","About 10-minute measurements")
    with b: card("AVERAGE USE",f"{df[TARGET].mean():.1f} Wh","Across all observations")
    with c: card("TEST MAE",f"{result['mae']:.1f} Wh","Lower is better")
    with d: card("TEST R²",f"{result['r2']:.3f}","Held-out time period")
    left,right=st.columns([1.4,1])
    with left:
        st.markdown("### Appliance energy over time")
        hourly=df.assign(Hour=df["date"].dt.floor("h")).groupby("Hour",as_index=False)[TARGET].mean()
        fig=px.line(hourly,x="Hour",y=TARGET,labels={"Hour":"Time",TARGET:"Appliance energy (Wh)"})
        fig.update_traces(line=dict(color="#2878b5",width=2))
        fig.update_layout(plot_bgcolor="white",paper_bgcolor="white")
        st.plotly_chart(fig,use_container_width=True)
    with right:
        st.markdown("### Actual vs predicted")
        comp=pd.DataFrame({"Actual":result["actual"],"Predicted":result["pred"]})
        fig=px.scatter(comp,x="Actual",y="Predicted",opacity=.45,labels={"Actual":"Actual (Wh)","Predicted":"Predicted (Wh)"})
        lo,hi=float(comp.min().min()),float(comp.max().max())
        fig.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode="lines",name="Perfect prediction",line=dict(color="#e58a38",dash="dash")))
        fig.update_layout(plot_bgcolor="white",paper_bgcolor="white")
        st.plotly_chart(fig,use_container_width=True)
    st.markdown("### Project workflow")
    cols=st.columns(5)
    for col,item in zip(cols,[("01","Load","Validate dataset"),("02","Prepare","Time features"),("03","Explore","Stats and charts"),("04","Train","Linear Regression"),("05","Evaluate","MAE, RMSE, R²")]):
        with col:
            st.markdown(f"**{item[0]} · {item[1]}**")
            st.caption(item[2])

elif page == "Explore the Data":
    header("Explore the dataset","Descriptive statistics and plots help explain patterns before interpreting the model.")
    with st.expander("Dataset preview",expanded=True): st.dataframe(df.head(100),use_container_width=True,height=280)
    a,b,c=st.columns(3)
    a.metric("Rows",f"{len(df):,}"); b.metric("Columns",str(df.shape[1])); c.metric("Missing values",f"{int(df.isna().sum().sum()):,}")
    t1,t2,t3=st.tabs(["Distribution","Relationships","Summary statistics"])
    with t1:
        nums=df.select_dtypes(include=np.number).columns.tolist()
        col=st.selectbox("Numeric variable",nums,index=nums.index(TARGET))
        fig=px.histogram(df,x=col,nbins=45,marginal="box",color_discrete_sequence=["#2878b5"])
        fig.update_layout(plot_bgcolor="white",paper_bgcolor="white")
        st.plotly_chart(fig,use_container_width=True)
        sd=df[col].std(ddof=0) or 1
        z=(df[col]-df[col].mean())/sd
        st.caption(f"Mean {df[col].mean():.2f} · Median {df[col].median():.2f} · Std dev {df[col].std():.2f} · |Z-score| > 3: {(z.abs()>3).sum():,}")
    with t2:
        candidates=[x for x in df.select_dtypes(include=np.number).columns if x not in ["rv1","rv2"]]
        defaults=[x for x in ["Appliances","lights","T1","RH_1","T_out","Windspeed"] if x in candidates]
        selected=st.multiselect("Heatmap variables",candidates,default=defaults)
        if len(selected)>=2:
            fig=px.imshow(df[selected].corr(),text_auto=".2f",color_continuous_scale="RdBu_r",zmin=-1,zmax=1,aspect="auto")
            st.plotly_chart(fig,use_container_width=True)
        else: st.info("Choose at least two variables for a correlation heatmap.")
        feature=st.selectbox("Compare against appliance energy",[x for x in candidates if x!=TARGET])
        fig=px.scatter(df.sample(min(2500,len(df)),random_state=7),x=feature,y=TARGET,opacity=.45,labels={TARGET:"Appliance energy (Wh)"})
        fig.update_layout(plot_bgcolor="white",paper_bgcolor="white")
        st.plotly_chart(fig,use_container_width=True)
    with t3:
        summary=df.select_dtypes(include=np.number).describe().T
        st.dataframe(summary,use_container_width=True)
        st.download_button("Download descriptive statistics",summary.to_csv().encode(),file_name="energy_descriptive_statistics.csv",mime="text/csv")

elif page == "Predict Consumption":
    header("Try a consumption estimate","Enter familiar sensor readings and weather conditions to estimate appliance energy use.")
    inputs=[x for x in df.columns if x not in ["date",TARGET,"rv1","rv2"]]
    defaults=df[inputs].median(numeric_only=True)
    values={}
    # Friendly labels are for display only; raw dataset column names remain the model inputs.
    field_labels = {
        "T1": "Temperature sensor 1 (°C)", "RH_1": "Humidity sensor 1 (%)",
        "T2": "Temperature sensor 2 (°C)", "RH_2": "Humidity sensor 2 (%)",
        "T3": "Temperature sensor 3 (°C)", "RH_3": "Humidity sensor 3 (%)",
        "T4": "Temperature sensor 4 (°C)", "RH_4": "Humidity sensor 4 (%)",
        "T5": "Temperature sensor 5 (°C)", "RH_5": "Humidity sensor 5 (%)",
        "T6": "Temperature sensor 6 (°C)", "RH_6": "Humidity sensor 6 (%)",
        "T7": "Temperature sensor 7 (°C)", "RH_7": "Humidity sensor 7 (%)",
        "T8": "Temperature sensor 8 (°C)", "RH_8": "Humidity sensor 8 (%)",
        "T9": "Temperature sensor 9 (°C)", "RH_9": "Humidity sensor 9 (%)",
        "T_out": "Outdoor temperature (°C)", "RH_out": "Outdoor humidity (%)",
        "lights": "Lighting energy use (Wh)", "Press_mm_hg": "Atmospheric pressure (mmHg)",
        "Windspeed": "Wind speed", "Visibility": "Visibility", "Tdewpoint": "Dew point temperature (°C)",
    }
    left,mid,right=st.columns(3)
    with left:
        st.markdown("**Indoor sensor readings**")
        for n in ["T1","RH_1","T2","RH_2","T3","RH_3","T4","RH_4"]:
            if n in inputs: values[n]=st.number_input(field_labels.get(n,n),value=float(defaults[n]),format="%.2f",help="Friendly label for dataset field " + n)
    with mid:
        st.markdown("**Additional sensors and outdoor conditions**")
        for n in ["T5","RH_5","T6","RH_6","T7","RH_7","T8","RH_8","T9","RH_9","T_out","RH_out"]:
            if n in inputs: values[n]=st.number_input(field_labels.get(n,n),value=float(defaults[n]),format="%.2f",help="Friendly label for dataset field " + n)
    with right:
        st.markdown("**Weather and lighting**")
        for n in ["lights","Press_mm_hg","Windspeed","Visibility","Tdewpoint"]:
            if n in inputs: values[n]=st.number_input(field_labels.get(n,n),value=float(defaults[n]),format="%.2f",help="Friendly label for dataset field " + n)
        day=st.date_input("Date",value=df["date"].iloc[-1].date())
        clock=st.time_input("Time",value=df["date"].iloc[-1].time().replace(second=0,microsecond=0))
    if st.button("⚡ Estimate appliance energy",type="primary",use_container_width=True):
        dt=pd.Timestamp.combine(day,clock)
        values.update({"hour":dt.hour,"day_of_week":dt.dayofweek,"month":dt.month,"is_weekend":int(dt.dayofweek>=5)})
        row={n:values.get(n,float(defaults[n]) if n in defaults.index else 0.0) for n in result["features"]}
        estimate=float(result["model"].predict(pd.DataFrame([row],columns=result["features"]))[0])
        threshold=float(df[TARGET].quantile(.75))
        x,y=st.columns([1,2])
        with x: card("PREDICTED CONSUMPTION",f"{max(0,estimate):.1f} Wh","Displayed estimate is floored at zero")
        with y:
            st.markdown("**Higher-than-usual usage**" if estimate>threshold else "**Within the usual range**")
            st.caption(f"Reference threshold (75th percentile): {threshold:.1f} Wh. This is illustrative, not an energy-efficiency diagnosis.")
            st.progress(float(np.clip(estimate/max(float(df[TARGET].quantile(.99)),1),0,1)))
        st.caption(f"Raw model output: {estimate:.2f} Wh. This is a what-if estimate using entered sensor readings, not a forecast from unknown future sensor data.")

elif page == "Model Insights":
    header("Inside the model","Held-out evaluation and fitted Linear Regression coefficients.")
    a,b,c=st.columns(3)
    with a: card("MAE",f"{result['mae']:.2f} Wh","Average absolute error")
    with b: card("RMSE",f"{result['rmse']:.2f} Wh","Penalizes larger errors")
    with c: card("R²",f"{result['r2']:.3f}","Held-out coefficient of determination")
    splitdate=df["date"].iloc[result["split"]]
    st.info(f"Training uses first {result['split']:,} rows. Testing uses final {len(df)-result['split']:,} rows from {splitdate:%d %b %Y, %H:%M}.")
    test=pd.DataFrame({"Date":result["dates"],"Actual (Wh)":result["actual"],"Predicted (Wh)":result["pred"]})
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=test["Date"],y=test["Actual (Wh)"],name="Actual",mode="lines",line=dict(color="#163b63",width=1.4)))
    fig.add_trace(go.Scatter(x=test["Date"],y=test["Predicted (Wh)"],name="Predicted",mode="lines",line=dict(color="#e58a38",width=1.4)))
    fig.update_layout(xaxis_title="Time",yaxis_title="Appliance energy (Wh)",plot_bgcolor="white",paper_bgcolor="white",legend=dict(orientation="h",y=1.1))
    st.plotly_chart(fig,use_container_width=True)
    st.markdown("### Largest coefficients by absolute magnitude")
    top=result["coef"].head(15).sort_values("Coefficient")
    fig=px.bar(top,x="Coefficient",y="Feature",orientation="h",color="Coefficient",color_continuous_scale="RdBu_r",color_continuous_midpoint=0)
    fig.update_layout(plot_bgcolor="white",paper_bgcolor="white",coloraxis_showscale=False)
    st.plotly_chart(fig,use_container_width=True)
    st.caption("Coefficient magnitudes depend on feature units. Association does not establish causation.")
    with st.expander("View all coefficients"):
        full=result["coef"][["Feature","Coefficient"]]
        st.dataframe(full,use_container_width=True)
        st.download_button("Download coefficients",full.to_csv(index=False).encode(),file_name="linear_regression_coefficients.csv",mime="text/csv")
    st.dataframe(test.head(300),use_container_width=True)
    st.download_button("Download held-out predictions",test.to_csv(index=False).encode(),file_name="held_out_energy_predictions.csv",mime="text/csv")

else:
    header("About the project","Syllabus-aligned predictive analytics with one main regression algorithm.")
    st.markdown("""
### Objective
Estimate appliance energy consumption, measured in Wh, from indoor temperature and humidity sensors, lights, outdoor weather, and time-derived features.

### Algorithm and validation
Ordinary **Linear Regression** is the only predictive model. Records are sorted by timestamp; the first 80% trains the model and the final 20% is held out for evaluation.

### Preprocessing and feature engineering
- Parse timestamps and remove rows with invalid timestamps or targets.
- Derive hour, day of week, month, and weekend.
- Exclude rv1 and rv2, which are random values and not meaningful sensor inputs.
- The UCI dataset has no missing values, but required fields are validated.

### Metrics
- **MAE:** average absolute error in Wh.
- **RMSE:** root mean squared error; larger misses count more.
- **R²:** performance relative to predicting the held-out mean; can be negative.

### Syllabus coverage
Preprocessing, descriptive statistics, outlier exploration, Z-scores, histograms and box plots, scatter plots, correlation heatmaps, feature engineering, Linear Regression, and model evaluation.

### Limitations and responsible use
The data represents one home over about 4.5 months and may not generalize to other houses or seasons. The form is a what-if estimator requiring sensor readings for the conditions being evaluated, not a standalone forecast. Fine-grained energy usage can reveal occupancy routines; handle raw records responsibly.

Dataset source: [UCI Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction).
""")
    st.dataframe(pd.DataFrame([
        ("Preprocessing","Timestamp validation and data checks"),
        ("EDA","Summary statistics, histogram, box plot, Z-score"),
        ("Visualization","Time series, scatter plots, correlation heatmap"),
        ("Feature engineering","Hour, weekday, month, weekend"),
        ("Predictive model","Ordinary Linear Regression"),
        ("Evaluation","MAE, RMSE, R² on chronological holdout"),
        ("Ethics","Generalization and household privacy"),
    ],columns=["Syllabus topic","Implementation"]),use_container_width=True)

st.markdown("---")
st.markdown('<div class="muted">WattWise · Predictive Analytics · Streamlit + pandas + scikit-learn + Plotly</div>',unsafe_allow_html=True)
