import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# MLflow for experiment tracking
import mlflow
import mlflow.sklearn

# Local monitoring module
from monitor import compute_metrics, generate_all_reports, REPORTS_DIR

# Constants
MLFLOW_EXPERIMENT_NAME = "harga_pangan_model"


def setup_mlflow(experiment_name: str = MLFLOW_EXPERIMENT_NAME):
    """Setup MLflow tracking."""
    mlflow.set_experiment(experiment_name)


def retrain(data_csv, version):
    """Retrain the model with MLflow tracking and Evidently monitoring."""
    setup_mlflow()

    df_clean = pd.read_csv(data_csv)

    # Label Encoder
    le_prov = LabelEncoder()
    df_clean['Provinsi_ID'] = le_prov.fit_transform(df_clean['Nama Provinsi'])
    le_kom = LabelEncoder()
    df_clean['Komoditas_ID'] = le_kom.fit_transform(df_clean['Komoditas'])

    # Definisi Fitur (X) dan Target (y)
    features = ['Harga_Bulan_Lalu', 'Harga_Tahun_Lalu', 'Bulan_Angka', 'Provinsi_ID', 'Komoditas_ID']
    target = 'Harga'

    df_train = df_clean[df_clean['Tahun'] < 2024].copy()
    df_test = df_clean[df_clean['Tahun'] >= 2024].copy()

    X_train = df_train[features]
    y_train = df_train[target]
    X_test = df_test[features]
    y_test = df_test[target]

    # Model parameters
    model_params = {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "random_state": 42,
    }

    with mlflow.start_run(run_name=f"train_{version}"):
        # Log parameters
        mlflow.log_params(model_params)
        mlflow.log_param("version", version)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("features", features)

        # Initialize & Train Model
        model = RandomForestRegressor(**model_params)
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Compute metrics
        train_metrics = compute_metrics(y_train, y_train_pred)
        test_metrics = compute_metrics(y_test, y_test_pred)

        # Log metrics to MLflow
        for metric_name, value in train_metrics.items():
            mlflow.log_metric(f"train_{metric_name}", value)
        for metric_name, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)

        print(f"Training Metrics: MAE={train_metrics['mae']:.2f}, MAPE={train_metrics['mape']:.4f}, R2={train_metrics['r2']:.4f}")
        print(f"Test Metrics: MAE={test_metrics['mae']:.2f}, MAPE={test_metrics['mape']:.4f}, R2={test_metrics['r2']:.4f}")

        # Add predictions to dataframes for Evidently
        df_train['prediction'] = y_train_pred
        df_test['prediction'] = y_test_pred

        # Generate Evidently reports
        report_paths = generate_all_reports(
            reference_data=df_train[features + [target, 'prediction']],
            current_data=df_test[features + [target, 'prediction']],
            target_col=target,
            prediction_col='prediction',
            version=version,
        )

        # Log Evidently reports as artifacts
        for report_name, report_path in report_paths.items():
            mlflow.log_artifact(report_path, artifact_path="evidently_reports")
            print(f"Generated {report_name} report: {report_path}")

        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")

        # Save model package locally
        model_package = {
            'model_rf': model,
            'le_prov': le_prov,
            'le_kom': le_kom,
            'data_ref': df_clean,
            'version': version,
            'metrics': {
                'train': train_metrics,
                'test': test_metrics,
            },
        }
        model_path = f'./models/hargaPangan_{version}.pkl'
        joblib.dump(model_package, model_path)
        mlflow.log_artifact(model_path, artifact_path="model_package")

        # Log feature importances
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        importance_path = f"{REPORTS_DIR}/feature_importance_{version}.csv"
        feature_importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path, artifact_path="feature_importance")

        print(f"\nFeature Importances:\n{feature_importance.to_string(index=False)}")
        print(f"\nMLflow Run ID: {mlflow.active_run().info.run_id}")

    return model_package, model_path
