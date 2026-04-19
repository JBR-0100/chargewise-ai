# ── Imports ─────────────────────────────────────────────────────────────────
import os
import sys
import warnings
from datetime import datetime, timedelta, time

from dotenv import load_dotenv
load_dotenv()  # Load .env file (GEMINI_API_KEY, etc.)

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Force non-interactive backend for matplotlib
matplotlib.use("Agg")

# Custom modules
from agent.data_agent import compute_stats, load_zone_df, add_features
from agent.ml_pipeline import CVDemandModel, FEATURES

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CLEAN_DIR   = os.path.join(BASE_DIR, "processed")
ZONE_CSV    = os.path.join(CLEAN_DIR, "zone_hourly_volume_long.csv")
RESULTS_CSV = os.path.join(CLEAN_DIR, "zone_model_results.csv")
KB_PATH     = os.path.join(BASE_DIR, "knowledge", "ev_planning_guidelines.txt")


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_zone_data():
    """Load the main zone-hourly demand dataset."""
    if not os.path.exists(ZONE_CSV):
        return None
    try:
        df = pd.read_csv(ZONE_CSV)
        df["time"] = pd.to_datetime(df["time"])
        # Add basic time components for filtering/display
        df["hour"]      = df["time"].dt.hour
        df["dayofweek"] = df["time"].dt.dayofweek
        df["month"]     = df["time"].dt.month
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


@st.cache_data
def load_results():
    """Load pre-computed model metrics."""
    if not os.path.exists(RESULTS_CSV):
        return None
    return pd.read_csv(RESULTS_CSV)


@st.cache_resource
def get_trained_model(zone_id: int, model_type: str = "Ridge"):
    """
    Train a CVDemandModel for a specific zone.
    Cached as a resource to avoid re-training on every interaction.
    """
    df = load_zone_data()
    if df is None:
        return None, None
        
    zdf = (
        df[df["TAZID"] == zone_id]
        .copy().sort_values("time").set_index("time")
    )
    
    if len(zdf) < 200:
        return None, None
        
    model = CVDemandModel(model_type=model_type)
    metrics = model.train_with_cv(zdf)
    return model, zdf


# ═══════════════════════════════════════════════════════════════════════════
# Page setup
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ChargeWise AI",
    page_icon="CW",
    layout="wide",
)

# Premium CSS Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

.metric-card {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    padding: 24px; border-radius: 16px; color: white;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    text-align: center; margin-bottom: 12px;
}
.metric-card h3 { font-size: 2.2rem; margin: 0; font-weight: 700; }
.metric-card p  { margin: 5px 0 0 0; font-size: 0.9rem; opacity: 0.85; text-transform: uppercase; letter-spacing: 1px; }

.report-card {
    background: #ffffff;
    border-left: 5px solid #2a5298;
    padding: 20px; border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    margin-bottom: 16px;
}
.report-card h4 { color: #1e3c72; margin-top: 0; font-weight: 600; }

.state-badge {
    display: inline-block;
    padding: 6px 16px; border-radius: 30px;
    font-size: 0.85rem; font-weight: 600;
    background: #e8f0fe; color: #1e3c72;
    border: 1px solid #d2e3fc;
}
.warning-box {
    background: #fff9e6; border-left: 5px solid #ffcc00;
    padding: 12px 16px; border-radius: 8px; margin-bottom: 12px;
    font-size: 0.9rem; color: #856404;
}
.stTabs [data-baseweb="tab"] { font-size: 1.1rem; font-weight: 600; padding-top: 10px; padding-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("ChargeWise AI")
st.caption("Advanced EV Charging Demand Analytics & Agentic Infrastructure Planning")

# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        '<div style="background:linear-gradient(135deg,#1e3c72,#2a5298);color:white;'
        'padding:12px 16px;border-radius:12px;text-align:center;font-weight:700;'
        'font-size:1.1rem;margin-bottom:12px;">CW &mdash; ChargeWise AI</div>',
        unsafe_allow_html=True
    )
    st.header("Control Panel")
    
    zone_data = load_zone_data()
    data_ready = zone_data is not None

    if data_ready:
        zones = sorted(zone_data["TAZID"].unique().tolist())
        sel_zone = st.selectbox("Current Zone Focus (TAZID)", zones, index=min(5, len(zones) - 1))
        st.success(f"Data Loaded: {len(zone_data):,} records")
    else:
        st.warning("Processed data not found. Please run preprocessing first.")
        sel_zone = None

    # API key loaded silently from .env or environment
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_api_key:
        try: gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")
        except: pass

    st.divider()
    st.info(
        "**ChargeWise Pipeline**\n\n"
        "1.  **Data Ingestion**: Raw station-level telemetry\n"
        "2.  **Gap Analysis**: Auto-fill missing sensor data\n"
        "3.  **Featurization**: Cyclical time series encoding\n"
        "4.  **Modeling**: Time-Series Aware Ridge Regression\n"
        "5.  **Validation**: 5-Fold Temporal Cross-Validation\n"
        "6.  **Agentic Planning**: RAG-based expansion strategy"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main Content Tabs
# ═══════════════════════════════════════════════════════════════════════════

tabs = st.tabs([
    "Overview",
    "Zone Forecasting",
    "Infrastructure Planning",
    "Load Hotspots",
    "Model Performance",
    "AI Planning Agent",
])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════

with tabs[0]:
    if not data_ready:
        st.info("Welcome! Please ensure the processed data is available in the `./processed/` directory.")
    else:
        # High-level Metrics
        c1, c2, c3, c4 = st.columns(4)
        total_kwh  = zone_data["volume"].sum()
        num_zones  = zone_data["TAZID"].nunique()
        date_range = f"{zone_data['time'].min().date()} to {zone_data['time'].max().date()}"
        avg_hourly = zone_data.groupby("time")["volume"].sum().mean()

        metrics = [
            (c1, f"{total_kwh:,.0f} kWh", "Total Energy Delivered"),
            (c2, f"{num_zones}", "Operational Zones"),
            (c3, f"{avg_hourly:.1f} kWh", "Avg Sysem-wide Hourly Demand"),
            (c4, date_range, "Observation Period"),
        ]
        for col, val, label in metrics:
            col.markdown(f'<div class="metric-card"><h3>{val}</h3><p>{label}</p></div>', unsafe_allow_html=True)

        st.markdown("### Aggregated System Demand")
        tot = zone_data.groupby("time")["volume"].sum()
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(tot.index, tot.values, color="#1e3c72", lw=1, alpha=0.8)
        ax.fill_between(tot.index, tot.values, alpha=0.1, color="#1e3c72")
        ax.set_ylabel("Demand (kWh)"); ax.grid(True, alpha=0.2); sns.despine()
        st.pyplot(fig)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Demand by Hour of Day")
            fig_h, ax_h = plt.subplots(figsize=(8, 5))
            sns.barplot(data=zone_data, x="hour", y="volume", ax=ax_h, palette="viridis", errorbar=None)
            ax_h.set_ylabel("Avg kWh"); sns.despine(); st.pyplot(fig_h)

        with col2:
            st.markdown("#### Demand by Day of Week")
            fig_d, ax_d = plt.subplots(figsize=(8, 5))
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            sns.barplot(data=zone_data, x="dayofweek", y="volume", ax=ax_d, palette="magma", errorbar=None)
            ax_d.set_xticklabels(days); ax_d.set_ylabel("Avg kWh"); sns.despine(); st.pyplot(fig_d)


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — ZONE FORECASTING
# ══════════════════════════════════════════════════════════════════════════

with tabs[1]:
    if not data_ready:
        st.stop()
    
    st.subheader(f"Demand Forecasting - Zone {sel_zone}")
    
    model, zdf = get_trained_model(sel_zone)
    
    if model is None:
        st.warning("Insufficient historical data for this zone (minimum 200 hours required).")
    else:
        # Forecast Visuals
        col_main, col_side = st.columns([3, 1])
        
        with col_main:
            # Show actual vs predicted on hold-out set
            X, y = model.prepare_data(zdf)
            split = int(len(X) * 0.8)
            X_test, y_test = X.iloc[split:], y.iloc[split:]
            
            preds_df = model.predict_with_interval(X_test)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            # Just show the last 168 hours (one week) for clarity
            display_slice = -168
            t_idx = y_test.index[display_slice:]
            
            ax.plot(t_idx, y_test.values[display_slice:], label="Actual", color="#1e3c72", lw=2)
            ax.plot(t_idx, preds_df["prediction"].values[display_slice:], label="Predicted", color="#e74c3c", linestyle="--")
            ax.fill_between(
                t_idx, 
                preds_df["lower_bound"].values[display_slice:], 
                preds_df["upper_bound"].values[display_slice:], 
                color="#e74c3c", alpha=0.15, label="95% Confidence Interval"
            )
            ax.legend(); ax.grid(True, alpha=0.2); ax.set_ylabel("kWh")
            st.pyplot(fig)
            
        with col_side:
            st.markdown("#### Future Prediction")
            st.write("Input a future date to predict demand.")
            target_date = st.date_input("Target Date", value=datetime.now() + timedelta(days=1))
            target_hour = st.slider("Target Hour", 0, 23, 12)
            
            # Construct a dummy timestamp for feature generation
            target_ts = pd.Timestamp(datetime.combine(target_date, time(target_hour)))
            
            # For purely future prediction without lags, we'll use the last known values as proxies
            # (In a production system, we'd use recursive forecasting)
            future_df = pd.DataFrame(index=[target_ts])
            # Add features (temporal only)
            future_df["hour"] = future_df.index.hour
            future_df["dayofweek"] = future_df.index.dayofweek
            future_df["month"] = future_df.index.month
            future_df["is_weekend"] = (future_df.index.dayofweek >= 5).astype(int)
            future_df["season"] = future_df["month"].map(
                {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
            )
            future_df["hour_sin"] = np.sin(2 * np.pi * future_df["hour"] / 24)
            future_df["hour_cos"] = np.cos(2 * np.pi * future_df["hour"] / 24)
            future_df["dow_sin"] = np.sin(2 * np.pi * future_df["dayofweek"] / 7)
            future_df["dow_cos"] = np.cos(2 * np.pi * future_df["dayofweek"] / 7)
            
            # For simplicity in this demo, we'll fill lags with mean historical values
            future_df["lag_1h"] = y.mean()
            future_df["lag_24h"] = y.mean()
            future_df["lag_168h"] = y.mean()
            future_df["roll_24h_mean"] = y.mean()
            
            future_pred = model.predict_with_interval(future_df)
            
            st.metric("Predicted Demand", f"{future_pred['prediction'][0]:.2f} kWh")
            st.caption(f"Range: {future_pred['lower_bound'][0]:.1f} - {future_pred['upper_bound'][0]:.1f} kWh")
            st.info("Note: Future predictions use historical means for lag features.")


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — INFRASTRUCTURE PLANNING
# ══════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.subheader("Infrastructure Planning Visualizations")
    st.markdown("Use these maps to identify priority zones and peak-hour requirements.")
    
    if data_ready:
        col1, col2 = st.columns([1, 2])
        with col1:
            sel_zones_multi = st.multiselect(
                "Filter Zones for Heatmap", 
                options=zones, 
                default=zones[:min(10, len(zones))]
            )
            agg_type = st.radio("Aggregation", ["Mean Hourly Demand", "Peak Hourly Demand"])
        
        with col2:
            subset = zone_data[zone_data["TAZID"].isin(sel_zones_multi)]
            if not subset.empty:
                agg_func = "mean" if "Mean" in agg_type else "max"
                pivot = subset.pivot_table(
                    values="volume", 
                    index="TAZID", 
                    columns="hour", 
                    aggfunc=agg_func
                )
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(pivot, cmap="YlOrRd", annot=False, cbar_kws={'label': 'kWh'})
                ax.set_title(f"{agg_type} by Zone and Hour")
                ax.set_xlabel("Hour of Day"); ax.set_ylabel("Zone (TAZID)")
                st.pyplot(fig)
            else:
                st.info("Select zones to generate heatmap.")


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — LOAD HOTSPOTS
# ══════════════════════════════════════════════════════════════════════════

with tabs[3]:
    if not data_ready:
        st.stop()
        
    st.subheader(f"Peak Load Analysis - Zone {sel_zone}")
    _, zdf = get_trained_model(sel_zone)
    
    if zdf is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Demand Intensity (Hour × Day)")
            pivot = zdf.pivot_table(values="volume", index="hour", columns="dayofweek", aggfunc="mean")
            pivot.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(pivot, cmap="inferno", ax=ax, annot=True, fmt=".1f")
            st.pyplot(fig)
            
        with c2:
            st.markdown("#### Weekday vs Weekend Profile")
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            zdf["is_weekend"] = (zdf.index.dayofweek >= 5)
            sns.lineplot(data=zdf, x="hour", y="volume", hue="is_weekend", ax=ax2, palette="cool")
            ax2.legend(["Weekday", "Weekend"]); ax2.grid(True, alpha=0.2)
            st.pyplot(fig2)


# ══════════════════════════════════════════════════════════════════════════
# TAB 5 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.subheader("Model Robustness & Accuracy")
    results_df = load_results()
    
    if results_df is None:
        st.warning("Please run `preprocess_run.py` to generate the global performance data.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Average MAE", f"{results_df['MAE'].mean():.2f} kWh")
        m2.metric("Average R²", f"{results_df['R2'].mean():.2f}")
        m3.metric("Zones Evaluated", len(results_df))
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Error Distribution (MAE)")
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            sns.histplot(results_df["MAE"], bins=20, kde=True, color="steelblue", ax=ax1)
            st.pyplot(fig1)
            
        with col2:
            st.markdown("#### R² Distribution (Robustness)")
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            sns.histplot(results_df["R2"], bins=20, kde=True, color="seagreen", ax=ax2)
            st.pyplot(fig2)
            
        st.markdown("#### Full Benchmark Results")
        st.dataframe(results_df.sort_values("R2", ascending=False), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 6 — AI PLANNING AGENT
# ══════════════════════════════════════════════════════════════════════════

with tabs[5]:
    st.subheader("AI Planning Assistant")
    st.caption("Strategic expansion recommendations based on demand patterns and local constraints.")

    if not data_ready:
        st.stop()

    # Reuse existing agent logic from previous version
    st.markdown("#### Select Zones for Site Report")
    agent_zones = st.multiselect(
        "TAZID(s)", options=zones, default=[sel_zone] if sel_zone else []
    )

    if not gemini_api_key:
        st.error("API Key Required. Please provide your Gemini API key in the sidebar.")
    elif st.button("Generate Strategic Report", type="primary"):
        # Import agent modules lazily
        try:
            from agent.rag import get_knowledge_base
            from agent.planning_agent import PlanningAgent
            from agent.pdf_export import generate_pdf
        except ImportError as e:
            st.error(f"Missing modules: {e}")
            st.stop()
            
        with st.spinner("Analyzing demand architecture..."):
            # Compute stats for selected cluster
            stats = compute_stats(agent_zones, ZONE_CSV)
            kb = get_knowledge_base(KB_PATH)
            chunks = kb.retrieve_for_stats(stats)
            
            agent = PlanningAgent(api_key=gemini_api_key)
            report = agent.run(stats, chunks)
            
            st.success("Analysis Complete!")
            
            # Display results
            st.markdown(f'<div class="report-card"><h4>Demand Summary</h4>{report.demand_summary}</div>', unsafe_allow_html=True)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### Recommended Expansion")
                for rec in report.expansion_recommendations:
                    st.write(f"- {rec}")
            with col_b:
                st.markdown("#### Grid Balancing Insights")
                for ins in report.scheduling_insights:
                    st.write(f"- {ins}")
            
            # PDF Download
            st.divider()
            stats_dict = {
                "Target Zones": ", ".join(str(z) for z in agent_zones),
                "Total Energy": f"{stats.total_kwh:,.0f} kWh",
                "Peak Load Hour": f"{stats.peak_hour}:00",
                "Data Quality": stats.data_quality.upper()
            }
            pdf_bytes = generate_pdf(report, zone_ids=agent_zones, stats_dict=stats_dict)
            st.download_button("Download Planning Document (PDF)", data=pdf_bytes, file_name="chargewise_report.pdf", mime="application/pdf")
