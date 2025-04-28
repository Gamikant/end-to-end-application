from prepare_data import *
from feature_selection import *
from autoencoder import *
from model import *
import os
import seaborn as sns
from mlflow_utils import init_mlflow, log_metrics_and_artifacts
import mlflow
import logging
from mlflow.tracking.fluent import _logger as mlflow_logger
import tensorflow as tf

def fs(default_hyperparameters):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')

    # Configure separate logger for pipeline
    pipeline_logger = logging.getLogger('feature_selection')
    pipeline_logger.setLevel(logging.WARNING)

    # Create file handler
    handler = logging.FileHandler('feature_selection.log')
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    pipeline_logger.addHandler(handler)

    # Initialize MLflow
    init_mlflow()
    
    # Start MLflow run
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params({
            "feature_selection_method": default_hyperparameters['feature_selection'],
            "feature_threshold": default_hyperparameters['feature_threshold'],
            "autoencoder_epochs": default_hyperparameters['autoencoder']['epochs'],
            "autoencoder_batch_size": default_hyperparameters['autoencoder']['batch_size']
        })
        dev, oos, oot = load_data(default_hyperparameters["train_file"], 
                                default_hyperparameters["validation_file"], 
                                default_hyperparameters["test_file"])
        pipeline_logger.info("-------------------------------------------------")
        pipeline_logger.info("Data loaded successfully.")
        dev_scaled, oos_scaled, oot_scaled, scaler = standardize_data(dev, oos, oot)
        pipeline_logger.info("-------------------------------------------------")
        pipeline_logger.info("Data scaled successfully.")
        feature_names = list(dev.drop(['Class'], axis=1).columns)
        dev_F, dev_NF, oos_F, oos_NF = split_data(dev, oos)
        pipeline_logger.info("-------------------------------------------------")
        pipeline_logger.info("Data split into fraud and non-fraud successfully.")
        scaled_dev_F = scaler.transform(dev_F)
        scaled_dev_NF = scaler.transform(dev_NF)
        scaled_oos_F = scaler.transform(oos_F)
        scaled_oos_NF = scaler.transform(oos_NF)

        script_dir = os.path.dirname(os.path.realpath(__file__))
        os.makedirs(os.path.join(script_dir, f'../artifacts/{run.info.run_id}/feature selection'), exist_ok=True)
        os.makedirs(os.path.join(script_dir, f'../artifacts/{run.info.run_id}/encoded data'), exist_ok=True)
        os.makedirs(os.path.join(script_dir, f'../artifacts/{run.info.run_id}/figures'), exist_ok=True)

        # Perform feature selection
        pipeline_logger.info("-------------------------------------------------")
        pipeline_logger.info("Performing feature selection...")
        features_to_drop, importance_scores_normal, importance_scores_abnormal, history_F, history_NF = feature_selection(
            scaled_dev_F, scaled_dev_NF, scaled_oos_F, scaled_oos_NF,
            feature_names,
            default_hyperparameters['feature_selection'],
            default_hyperparameters['feature_threshold'],
            run.info.run_id,
            default_hyperparameters['autoencoder']['ratios'],
            default_hyperparameters['autoencoder']['hidden_activation'],
            default_hyperparameters['autoencoder']['dropout'],
            default_hyperparameters['autoencoder']['optimizer'],
            default_hyperparameters['autoencoder']['loss'],
            default_hyperparameters['autoencoder']['epochs'],
            default_hyperparameters['autoencoder']['batch_size']
        )
        
        # Log metrics and artifacts
        figure_paths = [
            f'artifacts/{run.info.run_id}/figures/normal_fpi_importance.png',
            f'artifacts/{run.info.run_id}/figures/abnormal_fpi_importance.png'
        ]
        
        log_metrics_and_artifacts(
            importance_scores_normal,
            importance_scores_abnormal,
            feature_names,
            features_to_drop,
            history_F,  # Fraud autoencoder history
            history_NF, # Non-fraud autoencoder history
            figure_paths,
            run.info.run_id
        )
        
        return features_to_drop, run.info.run_id