"""
EV Charging Demand Prediction — Streamlit Dashboard
Milestone 1 UI requirement

Run with:
    streamlit run app.py
"""
\
import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
\
warnings.filterwarnings("ignore")
\
\
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CLEAN_DIR   = os.path.join(BASE_DIR, "processed")
ZONE_CSV    = os.path.join(CLEAN_DIR, "zone_hourly_volume_long.csv")
RESULTS_CSV = os.path.join(CLEAN_DIR, "zone_model_results.csv")
RAW_DIR     = os.path.join(BASE_DIR, "20220901-20230228_station-raw")
STATION_INFO_PATH = os.path.join(RAW_DIR, "station_information.csv")
\
\
FEATURES = [\
    "hour", "dayofweek", "month", "is_weekend", "season",\
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",\
    "lag_1h", "lag_24h", "lag_168h", "roll_24h_mean",\
]
\
\
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"]       = df.index.hour
    df["dayofweek"]  = df.index.dayofweek
    df["month"]      = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["season"]     = df["month"].map(\
        {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}\
    )
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["lag_1h"]    = df["volume"].shift(1)
    df["lag_24h"]   = df["volume"].shift(24)
    df["lag_168h"]  = df["volume"].shift(168)
    df["roll_24h_mean"] = df["volume"].shift(1).rolling(24).mean()
    return df.dropna()
@st.cache_data
def load_zone_data():
    if not os.path.exists(ZONE_CSV):
        return None
    df = pd.read_csv(ZONE_CSV)
    df["time"] = pd.to_datetime(df["time"])
    df["hour"]      = df["time"].dt.hour
    df["dayofweek"] = df["time"].dt.dayofweek
    df["month"]     = df["time"].dt.month
    return df
@st.cache_data
def load_results():
    if not os.path.exists(RESULTS_CSV):
        return None
    return pd.read_csv(RESULTS_CSV)
@st.cache_data
def load_station_info():
    if not os.path.exists(STATION_INFO_PATH):
        return None
    return pd.read_csv(STATION_INFO_PATH)
@st.cache_data
def train_zone_model(zone_id: int):
    df = load_zone_data()
    if df is None:
        return None, None, None, None
    zdf = (\
        df[df["TAZID"] == zone_id]\
        .copy()\
        .sort_values("time")\
        .set_index("time")\
    )
    zdf = add_features(zdf)
    if len(zdf) < 200:
        return None, None, None, None
    X, y = zdf[FEATURES], zdf["volume"]
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = np.maximum(model.predict(X_test), 0)
    return y_test, y_pred, model, zdf
st.set_page_config(\
    page_title="EV Charging Demand Predictor",\
    page_icon="",\
    layout="wide",\
)
\
\
st.markdown(\
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .metric-card {
        background: linear-gradient(135deg,#1e3c72,#2a5298);
        padding: 20px 24px; border-radius: 12px; color: white;
        text-align: center; margin-bottom: 8px;
    }
    .metric-card h3 { font-size: 2rem; margin: 0; }
    .metric-card p  { margin: 0; font-size: 0.85rem; opacity: .8; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
    </style>
    """,\
    unsafe_allow_html=True,\
)
\
st.title(" EV Charging Demand Prediction Dashboard")
st.caption("Milestone 1 · UrbanEVDataset (Shenzhen) · Sep 2022 – Feb 2023")
\
\
with st.sidebar:
    st.header(" Settings")
    zone_data = load_zone_data()
    data_ready = zone_data is not None
    \
    if data_ready:
        zones = sorted(zone_data["TAZID"].unique().tolist())
        sel_zone = st.selectbox("Select Zone (TAZID)", zones, index=min(5, len(zones)-1))
        st.caption(" Model: Linear Regression")
        st.markdown("---")
        st.markdown(\
            "**Pipeline**\n" \
            "1. Raw 5-min CSVs (per station)\n" \
            "2. Gap-fill (ffill/bfill)\n" \
            "3. Hourly resampling\n" \
            "4. Zone aggregation\n" \
            "5. Feature engineering\n" \
            "6. Train / Predict (80/20)\n" \
            "7. MAE & RMSE evaluation"\
        )
    else:
        st.warning(\
            "Processed data not found. " \
            "Please run preprocessing.ipynb first to generate `processed/zone_hourly_volume_long.csv`."\
        )
        sel_zone = None
tab1, tab2, tab3, tab4 = st.tabs(\
    [" Overview", " Zone Prediction", " Peak Demand", " Model Results"]\
)
\
\
\
\
with tab1:
    if not data_ready:
        st.info("Run preprocessing.ipynb then restart the app.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        total_kwh   = zone_data["volume"].sum()
        num_zones   = zone_data["TAZID"].nunique()
        date_range  = f"{zone_data['time'].min().date()} → {zone_data['time'].max().date()}"
        avg_hourly  = zone_data.groupby("time")["volume"].sum().mean()
        \
        for col, val, label in [\
            (c1, f"{total_kwh:,.0f} kWh", "Total Energy (all zones)"),\
            (c2, f"{num_zones}", "Unique Zones"),\
            (c3, f"{avg_hourly:.1f} kWh", "Avg Hourly Demand"),\
            (c4, date_range, "Dataset Period"),\
        ]:
            col.markdown(\
                f'<div class="metric-card"><h3>{val}</h3><p>{label}</p></div>',\
                unsafe_allow_html=True,\
            )
        st.markdown("###  Total Demand Over Time (All Zones)")
        total_over_time = zone_data.groupby("time")["volume"].sum()
        fig, ax = plt.subplots(figsize=(14, 3))
        ax.plot(total_over_time.index, total_over_time.values, color="#2a5298", lw=0.8)
        ax.fill_between(total_over_time.index, total_over_time.values, alpha=0.15, color="#2a5298")
        ax.set_ylabel("kWh"); ax.set_xlabel("Time"); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        \
        st.markdown("###  Demand Patterns")
        col1, col2, col3 = st.columns(3)
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        \
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        zone_data.groupby("hour")["volume"].mean().plot(kind="bar", ax=ax1, color="steelblue", edgecolor="white")
        ax1.set_title("Avg Demand by Hour"); ax1.set_xlabel("Hour"); ax1.set_ylabel("kWh")
        fig1.tight_layout(); col1.pyplot(fig1)
        \
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        zone_data.groupby("dayofweek")["volume"].mean().plot(kind="bar", ax=ax2, color="darkorange", edgecolor="white")
        ax2.set_xticklabels(days, rotation=45)
        ax2.set_title("Avg Demand by Day"); ax2.set_xlabel("Day"); ax2.set_ylabel("kWh")
        fig2.tight_layout(); col2.pyplot(fig2)
        \
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        zone_data.groupby("month")["volume"].mean().plot(kind="bar", ax=ax3, color="seagreen", edgecolor="white")
        ax3.set_title("Avg Demand by Month"); ax3.set_xlabel("Month"); ax3.set_ylabel("kWh")
        fig3.tight_layout(); col3.pyplot(fig3)
with tab2:
    if not data_ready:
        st.info("Run preprocessing.ipynb then restart the app.")
    else:
        st.subheader(f"Zone {sel_zone} — Demand Forecast (Linear Regression)")
        with st.spinner("Training model …"):
            y_test, y_pred, model, zdf = train_zone_model(sel_zone)
        if y_test is None:
            st.warning("Not enough data for this zone.")
        else:
            mae  = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            \
            m1, m2, m3 = st.columns(3)
            m1.metric("MAE",  f"{mae:.2f} kWh")
            m2.metric("RMSE", f"{rmse:.2f} kWh")
            m3.metric("Test samples", f"{len(y_test):,}")
            \
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(y_test.index, y_test.values, label="Actual", color="royalblue", lw=1.5)
            ax.plot(y_test.index, y_pred,        label="Predicted", color="tomato",\
                    linestyle="--", lw=1.2)
            ax.set_title(f"Zone {sel_zone} — Actual vs Predicted (Test Set)")
            ax.set_ylabel("kWh"); ax.legend(); ax.grid(True, alpha=0.3)
            fig.tight_layout(); st.pyplot(fig)
            \
\
            residuals = y_test.values - y_pred
            fig2, axes = plt.subplots(1, 2, figsize=(12, 3))
            axes[0].hist(residuals, bins=40, color="slategray", edgecolor="white")
            axes[0].set_title("Residual Distribution"); axes[0].set_xlabel("Error (kWh)")
            axes[1].scatter(y_pred, residuals, alpha=0.3, s=10, color="slategray")
            axes[1].axhline(0, color="red", lw=1)
            axes[1].set_title("Residuals vs Predicted"); axes[1].set_xlabel("Predicted kWh")
            axes[1].set_ylabel("Residual")
            fig2.tight_layout(); st.pyplot(fig2)
with tab3:
    if not data_ready:
        st.info("Run preprocessing.ipynb then restart the app.")
    else:
        st.subheader(f" Peak Demand Analysis — Zone {sel_zone}")
        _, _, _, zdf = train_zone_model(sel_zone)
        \
        if zdf is None:
            st.warning("Not enough data.")
        else:
            \
            pivot = zdf.pivot_table(\
                values="volume", index="hour", columns="dayofweek", aggfunc="mean"\
            )
            pivot.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            \
            st.markdown("#### Demand Heatmap (Hour × Day)")
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(pivot, cmap="YlOrRd", ax=ax, linewidths=0.3, annot=True, fmt=".0f")
            ax.set_title(f"Zone {sel_zone} — Avg kWh")
            ax.set_ylabel("Hour of Day"); ax.set_xlabel("Day")
            fig.tight_layout(); st.pyplot(fig)
            \
            st.markdown("#### Top 10 Peak Hours")
            top10 = zdf["volume"].nlargest(10).reset_index()
            top10.columns = ["Timestamp", "Volume (kWh)"]
            st.dataframe(top10, use_container_width=True)
            \
\
            zdf_copy = zdf.copy()
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            for label, grp in zdf_copy.groupby("is_weekend"):
                name = "Weekend" if label else "Weekday"
                grp.groupby("hour")["volume"].mean().plot(\
                    ax=ax2, label=name, linewidth=2, marker="o", markersize=4\
                )
            ax2.set_title(f"Zone {sel_zone} — Avg Hourly Demand: Weekday vs Weekend")
            ax2.set_xlabel("Hour"); ax2.set_ylabel("Avg kWh")
            ax2.legend(); ax2.grid(True, alpha=0.3)
            fig2.tight_layout(); st.pyplot(fig2)
with tab4:
    st.subheader(" Prediction Accuracy — All Zones")
    results_df = load_results()
    if results_df is None:
        st.info("Run preprocessing.ipynb (cells 10–11) to generate `zone_model_results.csv`.")
    else:
        lr_df = results_df[results_df["model"] == "LinearRegression"]
        summary = lr_df[["MAE", "RMSE"]].agg(["mean", "median", "std"])
        st.markdown("#### Summary Statistics (Linear Regression)")
        st.dataframe(summary.round(3), use_container_width=True)
        \
        col1, col2 = st.columns(2)
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        lr_df["MAE"].hist(bins=30, ax=ax1, color="steelblue", edgecolor="white")
        ax1.set_title("MAE Distribution"); ax1.set_xlabel("MAE (kWh)")
        fig1.tight_layout(); col1.pyplot(fig1)
        \
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        lr_df["RMSE"].hist(bins=30, ax=ax2, color="darkorange", edgecolor="white")
        ax2.set_title("RMSE Distribution"); ax2.set_xlabel("RMSE (kWh)")
        fig2.tight_layout(); col2.pyplot(fig2)
        \
        st.markdown("#### All Zone Results (Linear Regression)")
        st.dataframe(\
            lr_df.sort_values("RMSE").round(3),\
            use_container_width=True,\
            height=400,\
        )
        \
\
        st.download_button(\
            " Download Results CSV",\
            data=lr_df.to_csv(index=False),\
            file_name="zone_model_results.csv",\
            mime="text/csv",\
        )
    st.markdown("---")
    st.subheader(" Upload EV Charging Data (Optional)")
    uploaded = st.file_uploader(\
        "Upload a station CSV (must have columns: time, volume)", type="csv"\
    )
    if uploaded:
        udf = pd.read_csv(uploaded)
        if "time" not in udf.columns or "volume" not in udf.columns:
            st.error("CSV must have 'time' and 'volume' columns.")
        else:
            udf["time"] = pd.to_datetime(udf["time"])
            udf = udf.sort_values("time").set_index("time")
            udf = add_features(udf)
            if len(udf) < 50:
                st.warning("Not enough rows after feature engineering (need ≥200).")
            else:
                X, y = udf[FEATURES], udf["volume"]
                split = int(len(X) * 0.8)
                X_tr, X_te = X.iloc[:split], X.iloc[split:]
                y_tr, y_te = y.iloc[:split], y.iloc[split:]
                mdl = LinearRegression()
                mdl.fit(X_tr, y_tr)
                y_pr = np.maximum(mdl.predict(X_te), 0)
                mae_u  = mean_absolute_error(y_te, y_pr)
                rmse_u = np.sqrt(mean_squared_error(y_te, y_pr))
                \
                st.success(f" Model trained  |  MAE: {mae_u:.2f} kWh  |  RMSE: {rmse_u:.2f} kWh")
                \
                fig_u, ax_u = plt.subplots(figsize=(14, 4))
                ax_u.plot(y_te.index, y_te.values, label="Actual", color="royalblue")
                ax_u.plot(y_te.index, y_pr, label="Predicted", color="tomato", linestyle="--")
                ax_u.set_title("Uploaded Data — Actual vs Predicted")
                ax_u.legend(); ax_u.grid(True, alpha=0.3)
                fig_u.tight_layout(); st.pyplot(fig_u)
