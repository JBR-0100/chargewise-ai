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
def add_features(df):
    df = df.copy()
    df["hour"]       = df.index.hour
    df["dayofweek"]  = df.index.dayofweek
    df["month"]      = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["season"]     = df["month"].map(\
        {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}\
    )
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]    = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["lag_1h"]     = df["volume"].shift(1)
    df["lag_24h"]    = df["volume"].shift(24)
    df["lag_168h"]   = df["volume"].shift(168)
    df["roll_24h_mean"] = df["volume"].shift(1).rolling(24).mean()
    return df.dropna()
print("=" * 60)
print("STEP 1: Loading station information …")
station_info = pd.read_csv(STATION_INFO_PATH)
station_to_zone = dict(zip(station_info["station_id"], station_info["TAZID"]))
print(f"  Stations in info: {len(station_info)}")
print(f"  Zones:            {station_info['TAZID'].nunique()}")
\
\
\
\
print("\nSTEP 2: Cleaning & gap-filling 5-min station files …")
files = sorted(glob.glob(RAW_5MIN_GLOB))
print(f"  Raw files found: {len(files)}")
\
success, skipped, fail = [], [], []
\
for i, file_path in enumerate(files):
    station_id = int(os.path.basename(file_path).replace(".csv", ""))
    \
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
        \
        full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="5min")
        df = df.reindex(full_index).ffill().bfill()
        df = df.reset_index().rename(columns={"index": "time"})
        \
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
\
\
\
\
print("\nSTEP 3: Resampling to hourly …")
cleaned_files = sorted(glob.glob(os.path.join(OUT_5MIN, "*.csv")))
for file_path in cleaned_files:
    station_id = int(os.path.basename(file_path).replace(".csv", ""))
    df = pd.read_csv(file_path)
    df["time"] = pd.to_datetime(df["time"])
    df_hourly = df.set_index("time").resample("h").agg(AGG_RULES).reset_index()
    df_hourly.to_csv(os.path.join(OUT_1HOUR, f"{station_id}.csv"), index=False)
hourly_files = glob.glob(os.path.join(OUT_1HOUR, "*.csv"))
print(f"   Hourly files: {len(hourly_files)}")
\
\
\
\
print("\nSTEP 4: Aggregating to zone-level …")
all_data = []
for file_path in hourly_files:
    df = pd.read_csv(file_path)
    df["time"] = pd.to_datetime(df["time"])
    all_data.append(df[["time", "TAZID", "volume"]])
all_data = pd.concat(all_data, ignore_index=True)
zone_hourly = all_data.groupby(["time", "TAZID"], as_index=False).agg({"volume": "sum"})
zone_hourly.to_csv(OUT_ZONE, index=False)
print(f"   zone_hourly_volume_long.csv: {zone_hourly.shape}")
print(f"     Zones: {zone_hourly['TAZID'].nunique()}")
\
\
\
\
print("\nSTEP 5: Training models per zone …")
zone_hourly_df = pd.read_csv(OUT_ZONE)
zone_hourly_df["time"] = pd.to_datetime(zone_hourly_df["time"])
zones = zone_hourly_df["TAZID"].unique()
\
results = []
evaluated = 0
\
for zone_id in zones:
    zdf = (\
        zone_hourly_df[zone_hourly_df["TAZID"] == zone_id]\
        .copy().sort_values("time").set_index("time")\
    )
    zdf = add_features(zdf)
    if len(zdf) < 200:
        continue
    X, y = zdf[FEATURES], zdf["volume"]
    split = int(len(X) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]
    \
    model = LinearRegression()
    model.fit(X_tr, y_tr)
    y_pred = np.maximum(model.predict(X_te), 0)
    mae  = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    results.append({"zone": zone_id, "model": "LinearRegression", "MAE": mae, "RMSE": rmse})
    evaluated += 1
results_df = pd.DataFrame(results)
results_df.to_csv(OUT_RESULTS, index=False)
print(f"   Evaluated {evaluated} zones")
\
print(f"\n   Mean performance across all zones ")
print(results_df[["MAE", "RMSE"]].mean().round(3).to_string())
\
print("\n" + "=" * 60)
print(" Preprocessing complete! Processed files are in: ./processed/")
print("   Run the dashboard:  streamlit run app.py")
print("=" * 60)
