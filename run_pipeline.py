from pathlib import Path

from src.ts_forecasting.pipeline import run_pipeline


if __name__ == "__main__":
    run_pipeline(Path(__file__).resolve().parent)

