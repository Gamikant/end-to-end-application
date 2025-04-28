import json
import logging
import mlflow
import traceback
from pipeline import *
from feature_selection import *
from prepare_data import *
from autoencoder import *
from mlflow_utils import init_mlflow, log_metrics_and_artifacts
import sys
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['KERAS_VERBOSE'] = '0'
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import tensorflow as tf
import argparse
import io

# Configure logging
logging.basicConfig(
    filename='feature_selection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def run_feature_selection(config_path):
    try:
        # Load hyperparameters
        with open(config_path) as f:
            default_hyperparameters = json.load(f)
        
        logging.info(f"Default parameters: {default_hyperparameters}")
        
        # Initialize MLflow
        init_mlflow()
        
        with mlflow.start_run() as run:
            try:
                logging.info("Running initial feature selection using default parameters")
                
                # Log parameters
                mlflow.log_params({
                    "feature_selection_method": default_hyperparameters['feature_selection'],
                    "feature_threshold": default_hyperparameters['feature_threshold'],
                    "autoencoder_params": default_hyperparameters['autoencoder']
                })
                
                # Run feature selection
                results = fs(default_hyperparameters, run.info.run_id)
                method = default_hyperparameters['feature_selection']
                
                # Log all metrics and artifacts
                log_metrics_and_artifacts(
                    results['importance_scores_normal'],
                    results['importance_scores_abnormal'],
                    results['feature_names'],
                    method,
                    results['features_dropped'],
                    results['history_F'],
                    results['history_NF'],
                    [f'artifacts/{run.info.run_id}/figures/normal_'+method+'_importance.png', f'artifacts/{run.info.run_id}/figures/abnormal_'+method+'_importance.png'],
                    results['autoencoder_F'],
                    results['autoencoder_NF'],
                    results['dev'],
                    results['oos'],
                    results['oot'],
                    run.info.run_id
                )
                
                # Log success status
                mlflow.log_param("status", "SUCCESS")
                mlflow.log_metric("num_features_dropped", len(results['features_dropped']))
                
                logging.info(f"Feature selection completed. Dropped features: {results['features_dropped']}")
                
                os.system('cls' if os.name == 'nt' else 'clear')
                return {
                    "status": "success",
                    "features_dropped": results['features_dropped'],
                    "mlflow_run_id": run.info.run_id,
                    "metrics": {
                        "features_removed": len(results['features_dropped'])
                    }
                }
                
            except Exception as e:
                logging.error(f"Error during feature selection: {str(e)}")
                logging.error(traceback.format_exc())
                mlflow.log_param("status", "FAILED")
                mlflow.log_param("error_message", str(e))
                return {
                    "status": "error",
                    "message": str(e),
                    "traceback": traceback.format_exc()
                }
                
    except Exception as e:
        logging.error(f"Error in setup: {str(e)}")
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(3)

    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('mlflow').setLevel(logging.WARNING)

    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    try:
        result = run_feature_selection(args.config)
        print(json.dumps(result))
    except Exception:
        exit(1)