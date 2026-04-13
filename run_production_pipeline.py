from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.ts_forecasting.data import load_dataset
from src.ts_forecasting.production_system import prepare_series, train_or_load_best_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Production forecasting pipeline with retraining and persistence.")
    parser.add_argument("--input-csv", type=str, default="", help="Optional full dataset CSV (Date, Weekly_Sales).")
    parser.add_argument("--append-csv", type=str, default="", help="Optional CSV with new rows to append.")
    parser.add_argument("--date-col", type=str, default="Date")
    parser.add_argument("--target-col", type=str, default="Weekly_Sales")
    parser.add_argument("--retrain", action="store_true", help="Retrain models and overwrite artifacts.")
    parser.add_argument("--use-existing-model", action="store_true", help="Load saved models for prediction only.")
    parser.add_argument("--horizons", type=str, default="30,60,90", help="Comma-separated forecast horizons in days.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    artifacts_dir = project_root / "outputs" / "production"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if args.input_csv:
        base_df = pd.read_csv(args.input_csv)
        series = prepare_series(base_df, date_col=args.date_col, target_col=args.target_col)
    else:
        base_series, _ = load_dataset(project_root)
        base_df = base_series.reset_index().rename(columns={"y": args.target_col, "Date": args.date_col})
        if args.date_col not in base_df.columns:
            base_df = base_df.rename(columns={base_df.columns[0]: args.date_col})
        if args.target_col not in base_df.columns:
            base_df = base_df.rename(columns={base_df.columns[1]: args.target_col})
        series = prepare_series(base_df, date_col=args.date_col, target_col=args.target_col)

    if args.append_csv:
        append_df = pd.read_csv(args.append_csv)
        merged = pd.concat([base_df, append_df], ignore_index=True)
        series = prepare_series(merged, date_col=args.date_col, target_col=args.target_col)

    horizons_days = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    retrain = args.retrain or not args.use_existing_model
    result = train_or_load_best_models(
        series=series,
        artifacts_dir=artifacts_dir,
        retrain=retrain,
        horizons_days=horizons_days,
    )

    print("Production pipeline complete.")
    print(f"Artifacts directory: {artifacts_dir}")
    print(f"Best model: {result['metadata'].get('best_model')}")
    print("Saved CSVs:")
    print(f"- {artifacts_dir / 'actual_vs_predicted.csv'}")
    print(f"- {artifacts_dir / 'model_comparison_production.csv'}")
    print(f"- {artifacts_dir / 'future_predictions_all_models.csv'}")
    print(f"- {artifacts_dir / 'future_predictions_best_model.csv'}")


if __name__ == "__main__":
    main()

