import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient
import os
import pandas as pd

def init_mlflow(experiment_name="Feature_Selection_Pipeline"):
    """Initialize MLflow experiment"""
    mlflow.set_tracking_uri("http://localhost:5000")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
