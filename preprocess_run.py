"""
preprocess_run.py — Run the preprocessing pipeline locally
Equivalent to executing the preprocessing notebook cells 3–10.

Usage:
    python3 preprocess_run.py
"""
\
import os
import glob
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
\
warnings.filterwarnings("ignore")
\
\
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
RAW_DIR   = os.path.join(BASE_DIR, "20220901-20230228_station-raw")
CLEAN_DIR = os.path.join(BASE_DIR, "processed")
\
STATION_INFO_PATH = os.path.join(RAW_DIR, "station_information.csv")
RAW_5MIN_GLOB     = os.path.join(RAW_DIR, "charge_5min", "*.csv")
\
OUT_5MIN  = os.path.join(CLEAN_DIR, "charge_5min")
OUT_1HOUR = os.path.join(CLEAN_DIR, "charge_1hour")
OUT_ZONE  = os.path.join(CLEAN_DIR, "zone_hourly_volume_long.csv")
OUT_RESULTS = os.path.join(CLEAN_DIR, "zone_model_results.csv")
\
os.makedirs(OUT_5MIN,  exist_ok=True)
os.makedirs(OUT_1HOUR, exist_ok=True)
\
\
BAD_STATIONS = {\
    2129, 1663, 1478, 1082, 1055, 1722, 1039, 1036, 1681, 2125,\
    1487, 1113, 2138, 1034, 1337, 1497, 2337, 1501, 1101, 2291\
}
\
AGG_RULES = {\
    "busy":      "mean",\
    "idle":      "mean",\
    "fast_busy": "mean",\
    "fast_idle": "mean",\
    "slow_busy": "mean",\
    "slow_idle": "mean",\
    "duration":  "sum",\
    "volume":    "sum",\
    "s_price":   "mean",\
    "e_price":   "mean",\
    "TAZID":     "first",\
}
\
FEATURES = [\
    "hour", "dayofweek", "month", "is_weekend", "season",\
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",\
    "lag_1h", "lag_24h", "lag_168h", "roll_24h_mean",\
]
\
\
from agent.data_agent import add_features
from agent.ml_pipeline import CVDemandModel, FEATURES

print("=" * 60)
if not os.path.exists(RAW_DIR):
    print(f"WARNING: Raw data directory not found at {RAW_DIR}")
    print("Skipping raw data processing steps (1-4).")
    if os.path.exists(OUT_ZONE):
        print(f"Found existing processed data at {OUT_ZONE}. Proceeding to Step 5.")
        run_ml_step = True
    else:
        print(f"ERROR: Processed data not found at {OUT_ZONE}. Cannot proceed.")
        exit(1)
else:
    run_ml_step = True
    print("STEP 1: Loading station information …")
    station_info = pd.read_csv(STATION_INFO_PATH)
    station_to_zone = dict(zip(station_info["station_id"], station_info["TAZID"]))
    print(f"  Stations in info: {len(station_info)}")
    print(f"  Zones:            {station_info['TAZID'].nunique()}")


    print("\nSTEP 2: Cleaning & gap-filling 5-min station files …")
    files = sorted(glob.glob(RAW_5MIN_GLOB))
    print(f"  Raw files found: {len(files)}")

    success, skipped, fail = [], [], []

    for i, file_path in enumerate(files):
        try:
            station_id_str = os.path.basename(file_path).replace(".csv", "")
            station_id = int(station_id_str)
        except ValueError:
            continue
        
        if station_id in BAD_STATIONS or station_id not in station_to_zone:
            skipped.append(station_id)
            continue

        zone_id = station_to_zone[station_id]
        try:
            df = pd.read_csv(file_path)
            if len(df) < 10:
                skipped.append(station_id)
                continue
            df["time"]  = pd.to_datetime(df["time"])
            df["TAZID"] = zone_id
            df = df.sort_values("time").set_index("time")
            
            # Gap-filling logic: ensure a continuous 5-min index
            full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="5min")
            df = df.reindex(full_index).ffill().bfill()
            df = df.reset_index().rename(columns={"index": "time"})
            
            if df.isnull().sum().sum() > 0:
                fail.append(station_id)
                continue
            df.to_csv(os.path.join(OUT_5MIN, f"{station_id}.csv"), index=False)
            success.append(station_id)
        except Exception as e:
            print(f"   Station {station_id}: {e}")
            fail.append(station_id)
        if (i + 1) % 200 == 0:
            print(f"  … processed {i+1}/{len(files)} files")

    print(f"   Cleaned: {len(success)} | Skipped: {len(skipped)} | Failed: {len(fail)}")


    print("\nSTEP 3: Resampling to hourly …")
    cleaned_files = sorted(glob.glob(os.path.join(OUT_5MIN, "*.csv")))
    for file_path in cleaned_files:
        station_id = int(os.path.basename(file_path).replace(".csv", ""))
        df = pd.read_csv(file_path)
        df["time"] = pd.to_datetime(df["time"])
        # Aggregate 5-min intervals into hourly buckets
        df_hourly = df.set_index("time").resample("h").agg(AGG_RULES).reset_index()
        df_hourly.to_csv(os.path.join(OUT_1HOUR, f"{station_id}.csv"), index=False)
    hourly_files = glob.glob(os.path.join(OUT_1HOUR, "*.csv"))
    print(f"   Hourly files: {len(hourly_files)}")


    print("\nSTEP 4: Aggregating to zone-level …")
    all_data = []
    for file_path in hourly_files:
        df = pd.read_csv(file_path)
        df["time"] = pd.to_datetime(df["time"])
        all_data.append(df[["time", "TAZID", "volume"]])

    if all_data:
        all_data = pd.concat(all_data, ignore_index=True)
        zone_hourly = all_data.groupby(["time", "TAZID"], as_index=False).agg({"volume": "sum"})
        zone_hourly.to_csv(OUT_ZONE, index=False)
        print(f"   zone_hourly_volume_long.csv: {zone_hourly.shape}")
        print(f"     Zones: {zone_hourly['TAZID'].nunique()}")
    else:
        print("   ERROR: No data to aggregate!")
        exit(1)


if run_ml_step:
    print("\nSTEP 5: Training advanced models with Cross-Validation …")

zone_hourly_df = pd.read_csv(OUT_ZONE)
zone_hourly_df["time"] = pd.to_datetime(zone_hourly_df["time"])
zones = sorted(zone_hourly_df["TAZID"].unique())

results = []
evaluated = 0

for zone_id in zones:
    zdf = (
        zone_hourly_df[zone_hourly_df["TAZID"] == zone_id]
        .copy().sort_values("time").set_index("time")
    )
    
    # Using the new ML pipeline
    try:
        pipeline = CVDemandModel(model_type="Ridge")
        metrics = pipeline.train_with_cv(zdf, n_splits=5)
        
        results.append({
            "zone": zone_id,
            "model": "Ridge",
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"],
            "CV_MAE": metrics["CV_MAE_mean"],
            "CV_RMSE": metrics["CV_RMSE_mean"],
            "CV_R2": metrics["CV_R2_mean"],
        })
        evaluated += 1
    except Exception:
        # Skip zones with insufficient data for CV
        continue

if results:
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_RESULTS, index=False)
    print(f"   Evaluated {evaluated}/{len(zones)} zones")
    print(f"\n   Mean performance across all zones (Hold-out Test Set):")
    print(results_df[["MAE", "RMSE", "R2"]].mean().round(4).to_string())
    print(f"\n   Mean performance across all zones (Cross-Validation Mean):")
    print(results_df[["CV_MAE", "CV_RMSE", "CV_R2"]].mean().round(4).to_string())
else:
    print("   ERROR: No models were successfully trained.")

print("\n" + "=" * 60)
print(" Preprocessing complete! Processed files are in: ./processed/")
print("   Run the dashboard:  streamlit run app.py")
print("=" * 60)

