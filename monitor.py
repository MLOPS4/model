import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

# Evidently for monitoring
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset, DataQualityPreset
from evidently.metrics import (
    RegressionQualityMetric,
    RegressionPredictedVsActualScatter,
    RegressionErrorDistribution,
)

# Constants
REPORTS_DIR = "./reports"


def compute_metrics(y_true, y_pred) -> dict:
    """Compute regression metrics."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def generate_data_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    version: str,
    output_dir: str = REPORTS_DIR,
) -> str:
    """Generate data drift report comparing reference and current data."""
    os.makedirs(output_dir, exist_ok=True)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)

    report_path = f"{output_dir}/data_drift_{version}.html"
    report.save_html(report_path)

    return report_path


def generate_data_quality_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    version: str,
    output_dir: str = REPORTS_DIR,
) -> str:
    """Generate data quality report."""
    os.makedirs(output_dir, exist_ok=True)

    report = Report(metrics=[DataQualityPreset()])
    report.run(reference_data=reference_data, current_data=current_data)

    report_path = f"{output_dir}/data_quality_{version}.html"
    report.save_html(report_path)

    return report_path


def generate_regression_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    version: str,
    output_dir: str = REPORTS_DIR,
) -> str:
    """Generate regression performance report."""
    os.makedirs(output_dir, exist_ok=True)

    report = Report(
        metrics=[
            RegressionPreset(),
            RegressionQualityMetric(),
            RegressionPredictedVsActualScatter(),
            RegressionErrorDistribution(),
        ]
    )
    report.run(reference_data=reference_data, current_data=current_data)

    report_path = f"{output_dir}/regression_performance_{version}.html"
    report.save_html(report_path)

    return report_path


def generate_all_reports(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    target_col: str,
    prediction_col: str,
    version: str,
    output_dir: str = REPORTS_DIR,
) -> dict:
    """Generate all Evidently monitoring reports."""
    report_paths = {}

    # Data Drift Report
    report_paths["data_drift"] = generate_data_drift_report(
        reference_data, current_data, version, output_dir
    )

    # Data Quality Report
    report_paths["data_quality"] = generate_data_quality_report(
        reference_data, current_data, version, output_dir
    )

    # Regression Performance Report (if predictions available)
    if prediction_col in current_data.columns and target_col in current_data.columns:
        report_paths["regression_performance"] = generate_regression_report(
            reference_data, current_data, version, output_dir
        )

    return report_paths


def monitor_model_performance(
    model,
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list,
    target_col: str,
    version: str,
    output_dir: str = REPORTS_DIR,
) -> dict:
    """
    Monitor model performance by generating predictions and Evidently reports.
    
    Args:
        model: Trained model with predict method
        reference_df: Reference/training data
        current_df: Current/production data to monitor
        features: List of feature column names
        target_col: Target column name
        version: Version string for report naming
        output_dir: Directory to save reports
    
    Returns:
        Dictionary with report paths and metrics
    """
    # Generate predictions
    ref_predictions = model.predict(reference_df[features])
    curr_predictions = model.predict(current_df[features])

    # Add predictions to dataframes
    ref_with_pred = reference_df.copy()
    curr_with_pred = current_df.copy()
    ref_with_pred['prediction'] = ref_predictions
    curr_with_pred['prediction'] = curr_predictions

    # Compute metrics if target is available
    metrics = {}
    if target_col in current_df.columns:
        metrics["reference"] = compute_metrics(reference_df[target_col], ref_predictions)
        metrics["current"] = compute_metrics(current_df[target_col], curr_predictions)

    # Generate reports
    report_columns = features + [target_col, 'prediction']
    report_paths = generate_all_reports(
        reference_data=ref_with_pred[report_columns],
        current_data=curr_with_pred[report_columns],
        target_col=target_col,
        prediction_col='prediction',
        version=version,
        output_dir=output_dir,
    )

    return {
        "report_paths": report_paths,
        "metrics": metrics,
    }


