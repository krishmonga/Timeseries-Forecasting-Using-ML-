from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.ts_forecasting.data import load_dataset
from src.ts_forecasting.production_system import prepare_series, train_or_load_best_models


st.set_page_config(page_title="Sales Forecasting - Production", layout="wide")
st.title("Sales Forecasting (Production Ready)")
st.caption("Upload new data, retrain or reuse saved models, and forecast next 30/60/90 days.")

project_root = Path(__file__).resolve().parent
artifacts_dir = project_root / "outputs" / "production"
artifacts_dir.mkdir(parents=True, exist_ok=True)

mode = st.radio("Model mode", ["Use existing trained model", "Retrain model"], horizontal=True)
horizons = st.multiselect("Forecast horizons (days)", [30, 60, 90], default=[30, 60, 90])

uploaded_full = st.file_uploader("Upload full dataset CSV (optional)", type=["csv"], key="full")
uploaded_append = st.file_uploader("Upload append rows CSV (optional)", type=["csv"], key="append")

date_col = st.text_input("Date column name", value="Date")
target_col = st.text_input("Sales column name", value="Weekly_Sales")

if st.button("Run Forecasting"):
    with st.spinner("Running pipeline..."):
        if uploaded_full is not None:
            base_df = pd.read_csv(uploaded_full)
            series = prepare_series(base_df, date_col=date_col, target_col=target_col)
        else:
            base_series, _ = load_dataset(project_root)
            base_df = base_series.reset_index().rename(columns={"y": target_col, "Date": date_col})
            if date_col not in base_df.columns:
                base_df = base_df.rename(columns={base_df.columns[0]: date_col})
            if target_col not in base_df.columns:
                base_df = base_df.rename(columns={base_df.columns[1]: target_col})
            series = prepare_series(base_df, date_col=date_col, target_col=target_col)

        if uploaded_append is not None:
            append_df = pd.read_csv(uploaded_append)
            merged = pd.concat([base_df, append_df], ignore_index=True)
            series = prepare_series(merged, date_col=date_col, target_col=target_col)

        retrain = mode == "Retrain model"
        result = train_or_load_best_models(
            series=series,
            artifacts_dir=artifacts_dir,
            retrain=retrain,
            horizons_days=tuple(horizons) if horizons else (30, 60, 90),
        )

    st.success("Pipeline complete.")
    st.subheader("Model Comparison")
    comp_path = artifacts_dir / "model_comparison_production.csv"
    if comp_path.exists():
        st.dataframe(pd.read_csv(comp_path), use_container_width=True)

    st.subheader("Actual vs Predicted")
    actual_path = artifacts_dir / "actual_vs_predicted.csv"
    if actual_path.exists():
        actual_df = pd.read_csv(actual_path)
        st.dataframe(actual_df.head(30), use_container_width=True)

    st.subheader("Future Predictions (Best Model)")
    best_path = artifacts_dir / "future_predictions_best_model.csv"
    if best_path.exists():
        best_df = pd.read_csv(best_path)
        st.dataframe(best_df, use_container_width=True)
        st.line_chart(best_df.set_index("Date")["Predicted_Sales"])

    st.subheader("System Adaptation Summary")
    meta = result.get("metadata", {})
    st.json(
        {
            "best_model": meta.get("best_model"),
            "seasonal_period": meta.get("seasonal_period"),
            "retraining_strategy": meta.get("retraining_strategy"),
            "incremental_learning": meta.get("incremental_learning"),
        }
    )

