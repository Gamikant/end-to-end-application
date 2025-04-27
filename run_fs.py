import json
import logging
import mlflow
import traceback
from pipeline import *
from feature_selection import *
from prepare_data import *
from autoencoder import *
from mlflow_utils import init_mlflow, log_metrics_and_artifacts

# Configure logging
logging.basicConfig(
    filename='feature_selection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def run_feature_selection():
    try:
        # Load hyperparameters
        with open('default_hyperparameters.json', 'r') as f:
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
                results = fs(default_hyperparameters)

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
                    ['figures/normal_'+method+'_importance.png', 'figures/abnormal_'+method+'_importance.png'],
                    results['autoencoder_F'],
                    results['autoencoder_NF'],
                    results['dev'],
                    results['oos'],
                    results['oot']
                )
                
                # Log success status
                mlflow.log_param("status", "SUCCESS")
                mlflow.log_metric("num_features_dropped", len(results['features_dropped']))
                
                logging.info(f"Feature selection completed. Dropped features: {results['features_dropped']}")
                
                return results['features_dropped']
                
            except Exception as e:
                logging.error(f"Error during feature selection: {str(e)}")
                logging.error(traceback.format_exc())
                mlflow.log_param("status", "FAILED")
                mlflow.log_param("error_message", str(e))
                raise
                
    except Exception as e:
        logging.error(f"Error in setup: {str(e)}")
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    try:
        run_feature_selection()
    except Exception:
        exit(1)