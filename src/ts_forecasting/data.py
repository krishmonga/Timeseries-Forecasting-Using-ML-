from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller


@dataclass
class DatasetInfo:
    name: str
    source_path: str
    records: int


def load_dataset(project_root: Path) -> Tuple[pd.Series, DatasetInfo]:
    """Auto-load a real-world dataset and return a weekly univariate series."""
    walmart_train = project_root / "walmartseries" / "train.csv"
    if walmart_train.exists():
        frame = pd.read_csv(walmart_train, parse_dates=["Date"])
        series = (
            frame.groupby("Date", as_index=True)["Weekly_Sales"]
            .sum()
            .sort_index()
            .asfreq("W-FRI")
        )
        return series.rename("y"), DatasetInfo(
            name="Walmart Weekly Sales (aggregated)",
            source_path=str(walmart_train),
            records=len(frame),
        )

    # Fallback to an offline-friendly built-in real-world series.
    from statsmodels.datasets import sunspots

    fallback = sunspots.load_pandas().data
    fallback["Date"] = pd.to_datetime(fallback["YEAR"].astype(int).astype(str) + "-12-31")
    series = fallback.set_index("Date")["SUNACTIVITY"].rename("y").asfreq("YE")
    return series, DatasetInfo(
        name="Sunspots Activity (fallback)",
        source_path="statsmodels.datasets.sunspots",
        records=len(fallback),
    )


def clean_series(series: pd.Series) -> Tuple[pd.Series, Dict[str, float]]:
    """Fill missing values, cap outliers, and return a quality report."""
    report: Dict[str, float] = {
        "missing_before": float(series.isna().sum()),
    }

    cleaned = series.astype(float).interpolate(method="linear").bfill().ffill()
    q1, q3 = cleaned.quantile(0.25), cleaned.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    clipped = cleaned.clip(lower=lower, upper=upper)

    report["missing_after"] = float(clipped.isna().sum())
    report["outliers_capped"] = float(((cleaned < lower) | (cleaned > upper)).sum())
    return clipped.rename(series.name), report


def adf_check(series: pd.Series) -> Dict[str, float]:
    """Run Augmented Dickey-Fuller stationarity test."""
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "adf_statistic": float(result[0]),
        "p_value": float(result[1]),
        "used_lag": float(result[2]),
        "n_obs": float(result[3]),
    }


def plot_series(series: pd.Series, title: str, output_file: Path) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(series.index, series.values, label=series.name or "value")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150)
    plt.close()

