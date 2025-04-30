import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient
import os
import pandas as pd

def init_mlflow(experiment_name="Feature_Selection_Pipeline"):
    """Initialize MLflow experiment"""
    # mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_tracking_uri("http://mlflow:5001")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

def log_metrics_and_artifacts(
    importance_scores_normal,
    importance_scores_abnormal,
    feature_names,
    method,
    features_dropped,
    autoencoder_F_history,
    autoencoder_NF_history,
    figure_paths,
    autoencoder_F,  # Add autoencoder models
    autoencoder_NF,
    dev_data,       # Add datasets
    oos_data,
    oot_data,
    run_id
):
    """Log metrics and artifacts to MLflow"""
    # Log models
    mlflow.keras.log_model(autoencoder_F, "fraud_autoencoder")
    mlflow.keras.log_model(autoencoder_NF, "nonfraud_autoencoder")
    
    # Log datasets
    # mlflow.log_artifact("input data/dev.csv", "datasets")
    # mlflow.log_artifact("input data/oos.csv", "datasets")
    # mlflow.log_artifact("input data/oot.csv", "datasets")

    mlflow.log_artifact(f'artifacts/{run_id}/feature selection/abnormal_'+method+'_importance.csv', "feature_importance")
    mlflow.log_artifact(f'artifacts/{run_id}/feature selection/normal_'+method+'_importance.csv', "feature_importance")
    
    # Log dataset profiles
    dev_profile = dev_data.describe()
    oos_profile = oos_data.describe()
    oot_profile = oot_data.describe()
    
    # Save and log profiles
    dev_profile.to_csv(f"artifacts/{run_id}/feature selection/dev_profile.csv")
    oos_profile.to_csv(f"artifacts/{run_id}/feature selection/oos_profile.csv")
    oot_profile.to_csv(f"artifacts/{run_id}/feature selection/oot_profile.csv")
    
    mlflow.log_artifact(f"artifacts/{run_id}/feature selection/dev_profile.csv", "data_profiles")
    mlflow.log_artifact(f"artifacts/{run_id}/feature selection/oos_profile.csv", "data_profiles")
    mlflow.log_artifact(f"artifacts/{run_id}/feature selection/oot_profile.csv", "data_profiles")
  
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
        f'artifacts/{run_id}/figures/nonfraud_autoencoder_loss.png',
        f'artifacts/{run_id}/figures/normal_'+method+'_importance.png',
        f'artifacts/{run_id}/figures/abnormal_'+method+'_importance.png'
    ])
    
    for fig_path in figure_paths:
        mlflow.log_artifact(fig_path, "important figures")
    
    # Log feature selection results
    with open(f"artifacts/{run_id}/feature selection/features_dropped.txt", "r") as f:
        mlflow.log_text(f.read(), "features_dropped.txt")