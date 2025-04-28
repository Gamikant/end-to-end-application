import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient
import os

def init_mlflow(experiment_name="Feature_Selection_Pipeline"):
    """Initialize MLflow experiment"""
    mlflow.set_tracking_uri("http://localhost:5001")  # Change if using remote server
    mlflow.autolog(disable=True)
    # Create experiment if it doesn't exist
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    
def log_metrics_and_artifacts(
    importance_scores_normal,
    importance_scores_abnormal,
    feature_names,
    features_dropped,
    autoencoder_F_history,  # History for fraud autoencoder
    autoencoder_NF_history, # History for non-fraud autoencoder
    figure_paths,
    run_id
):
    """Log metrics and artifacts to MLflow"""
    
    # Log feature importance metrics
    for feature, importance in zip(feature_names, importance_scores_normal):
        mlflow.log_metric(f"normal_importance_{feature}", importance)
    
    for feature, importance in zip(feature_names, importance_scores_abnormal):
        mlflow.log_metric(f"abnormal_importance_{feature}", importance)
    
    # Log fraud autoencoder training metrics
    mlflow.log_metric("fraud_autoencoder_final_loss", autoencoder_F_history.history['loss'][-1])
    mlflow.log_metric("fraud_autoencoder_final_val_loss", autoencoder_F_history.history['val_loss'][-1])
    
    # Log non-fraud autoencoder training metrics
    mlflow.log_metric("nonfraud_autoencoder_final_loss", autoencoder_NF_history.history['loss'][-1])
    mlflow.log_metric("nonfraud_autoencoder_final_val_loss", autoencoder_NF_history.history['val_loss'][-1])
    
    # Log training curves for both autoencoders
    import matplotlib.pyplot as plt
    
    # Plot fraud autoencoder loss
    plt.figure(figsize=(10, 6))
    plt.plot(autoencoder_F_history.history['loss'], label='Training Loss')
    plt.plot(autoencoder_F_history.history['val_loss'], label='Validation Loss')
    plt.title('Fraud Autoencoder Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'artifacts/{run_id}/figures/fraud_autoencoder_loss.png')
    plt.close()
    
    # Plot non-fraud autoencoder loss
    plt.figure(figsize=(10, 6))
    plt.plot(autoencoder_NF_history.history['loss'], label='Training Loss')
    plt.plot(autoencoder_NF_history.history['val_loss'], label='Validation Loss')
    plt.title('Non-Fraud Autoencoder Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'artifacts/{run_id}/figures/nonfraud_autoencoder_loss.png')
    plt.close()
    
    # Log number of features dropped
    mlflow.log_metric("num_features_dropped", len(features_dropped))
    
    # Log all artifacts
    figure_paths.extend([
        f'artifacts/{run_id}/figures/fraud_autoencoder_loss.png',
        f'artifacts/{run_id}/figures/nonfraud_autoencoder_loss.png'
    ])
    
    for fig_path in figure_paths:
        mlflow.log_artifact(fig_path)

    with open(f"artifacts/{run_id}/feature selection/features_dropped.txt", "w") as f:
        for feature in features_dropped:
            f.write(f"{feature}\n")
    
    mlflow.log_artifact(f"artifacts/{run_id}/feature selection/features_dropped.txt")