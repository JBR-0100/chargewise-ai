"""
agent/data_agent.py
Data Agent — Step 1 of the agentic pipeline.
Computes zone-level demand statistics from the processed CSV.
Handles missing/noisy data gracefully via IQR-based anomaly detection.
Provides feature engineering utilities for ML model training.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

# ── Default paths ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ZONE_CSV = os.path.join(BASE_DIR, "processed", "zone_hourly_volume_long.csv")


@dataclass
class ZoneDemandStats:
    """
    Typed demand statistics for one or more zones.
    
    Attributes:
        zone_ids: List of TAZIDs analyzed.
        total_kwh: Cumulative volume across all records.
        avg_hourly_kwh: Mean volume per hour.
        peak_hour: Hour of the day (0-23) with highest average demand.
        peak_day: Day of the week with highest average demand.
        top_peak_timestamps: List of ISO-formatted timestamps for the top 5 peaks.
        is_high_load: Flag indicating if this zone is in the top 25% of all zones.
        anomaly_count: Count of values outside the 1.5x IQR fence.
        weekday_avg: Mean volume on weekdays (Mon-Fri).
        weekend_avg: Mean volume on weekends (Sat-Sun).
        monthly_trend: Mapping of month names to their average hourly demand.
        data_quality: Assessment of data reliability: "good", "sparse", or "noisy".
        warnings: List of specific data quality concerns discovered.
    """
    zone_ids: list[int]
    total_kwh: float
    avg_hourly_kwh: float
    peak_hour: int
    peak_day: str
    top_peak_timestamps: list[str]
    is_high_load: bool
    anomaly_count: int
    weekday_avg: float
    weekend_avg: float
    monthly_trend: dict[str, float]
    data_quality: str
    warnings: list[str] = field(default_factory=list)


_DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}


def add_features(df: pd.DataFrame, target_col: str = "volume") -> pd.DataFrame:
    """
    Apply feature engineering to a time-indexed DataFrame.
    
    Parameters:
        df: DataFrame with a DatetimeIndex.
        target_col: Name of the column containing the demand volume.
        
    Returns:
        DataFrame with added temporal and lag features. Columns with NaNs are dropped.
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    # Temporal features
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    
    # Seasonality (rough approximation for Northern Hemisphere)
    # 0: Winter, 1: Spring, 2: Summer, 3: Autumn
    df["season"] = df["month"].map(
        {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    )

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    # Lag features (if target column exists)
    if target_col in df.columns:
        df["lag_1h"] = df[target_col].shift(1)
        df["lag_24h"] = df[target_col].shift(24)
        df["lag_168h"] = df[target_col].shift(168)
        df["roll_24h_mean"] = df[target_col].shift(1).rolling(24).mean()

    return df.dropna()


def load_zone_df(zone_ids: list[int], csv_path: str = DEFAULT_ZONE_CSV) -> pd.DataFrame:
    """
    Load and filter zone-hourly CSV for the requested zones.
    
    Parameters:
        zone_ids: List of TAZIDs to include.
        csv_path: Path to the hourly volume CSV.
        
    Returns:
        Filtered DataFrame with a 'time' column and 'TAZID' column.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Zone CSV not found at: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
        if "time" not in df.columns or "TAZID" not in df.columns:
            raise KeyError(f"Expected columns 'time' and 'TAZID' not found in {csv_path}")
            
        df["time"] = pd.to_datetime(df["time"])
        df = df[df["TAZID"].isin(zone_ids)].copy()

        if df.empty:
            raise ValueError(f"No data found for zones: {zone_ids}")

        return df
    except Exception as e:
        raise RuntimeError(f"Error reading CSV File {csv_path}: {e}")


def _detect_anomalies(series: pd.Series) -> int:
    """Return count of values outside 1.5×IQR fence."""
    if series.empty:
        return 0
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((series < lower) | (series > upper)).sum())


def _assess_data_quality(df: pd.DataFrame, anomaly_count: int) -> tuple[str, list[str]]:
    """Determine reliability based on record density and variance."""
    warnings: list[str] = []
    n = len(df)

    if n < 100:
        warnings.append(f"Sparse data: only {n} hourly records available.")
        return "sparse", warnings

    zero_frac = (df["volume"] == 0).mean()
    if zero_frac > 0.5:
        warnings.append(f"{zero_frac:.0%} of volume values are zero — possible sensor gap.")
        return "sparse", warnings

    if anomaly_count > n * 0.05:
        warnings.append(f"{anomaly_count} spike anomalies detected (>{n * 0.05:.0f} records).")
        return "noisy", warnings

    return "good", warnings


def compute_stats(
    zone_ids: list[int],
    csv_path: str = DEFAULT_ZONE_CSV,
    all_zones_total: Optional[float] = None,
) -> ZoneDemandStats:
    """
    Compute comprehensive demand statistics for the given zone(s).

    Parameters:
        zone_ids: List of zone TAZIDs to analyze.
        csv_path: Path to the processed hourly volume data.
        all_zones_total: Precomputed total kWh across all zones for thresholding.
        
    Returns:
        ZoneDemandStats object populated with results.
    """
    df = load_zone_df(zone_ids, csv_path)

    # Aggregate across zones if multiple selected
    hourly = df.groupby("time")["volume"].sum().sort_index()

    # Basic stats
    total_kwh = float(hourly.sum())
    avg_hourly = float(hourly.mean())
    anomaly_count = _detect_anomalies(hourly)

    # Time features
    hourly_df = hourly.to_frame("volume")
    hourly_df["hour"] = hourly_df.index.hour
    hourly_df["dow"] = hourly_df.index.dayofweek
    hourly_df["month"] = hourly_df.index.month

    # Peak analysis
    peak_hour_series = hourly_df.groupby("hour")["volume"].mean()
    peak_hour = int(peak_hour_series.idxmax()) if not peak_hour_series.empty else 0
    
    peak_dow_series = hourly_df.groupby("dow")["volume"].mean()
    peak_dow = int(peak_dow_series.idxmax()) if not peak_dow_series.empty else 0
    peak_day = _DAY_NAMES[peak_dow]

    top_peaks = hourly.nlargest(5).index.strftime("%Y-%m-%d %H:%M").tolist()

    # Averages
    weekday_avg = float(hourly_df[hourly_df["dow"] < 5]["volume"].mean())
    weekend_avg = float(hourly_df[hourly_df["dow"] >= 5]["volume"].mean())

    # Ensure monthly names lookup exists
    monthly_trend = {
        _MONTH_NAMES.get(m, str(m)): round(float(v), 2)
        for m, v in hourly_df.groupby("month")["volume"].mean().items()
    }

    # High-load flag estimation
    if all_zones_total is None:
        try:
            full = pd.read_csv(csv_path)
            zone_totals = full.groupby("TAZID")["volume"].sum()
            threshold = zone_totals.quantile(0.75)
        except Exception:
            threshold = total_kwh
    else:
        threshold = all_zones_total * 0.05

    is_high_load = total_kwh >= float(threshold)

    # Data quality screening
    quality, warns = _assess_data_quality(hourly_df, anomaly_count)

    return ZoneDemandStats(
        zone_ids=zone_ids,
        total_kwh=round(total_kwh, 2),
        avg_hourly_kwh=round(avg_hourly, 4),
        peak_hour=peak_hour,
        peak_day=peak_day,
        top_peak_timestamps=top_peaks,
        is_high_load=is_high_load,
        anomaly_count=anomaly_count,
        weekday_avg=round(weekday_avg, 4),
        weekend_avg=round(weekend_avg, 4),
        monthly_trend=monthly_trend,
        data_quality=quality,
        warnings=warns,
    )


# ── Simple sanity check ───────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        stats = compute_stats([108, 502])
        print(f"Stats for zones 108, 502:\n{stats}")
    except Exception as e:
        print(f"Stats computation failed: {e}")

