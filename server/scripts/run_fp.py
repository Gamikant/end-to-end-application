import json
import logging
import mlflow
import traceback
from pipeline import *
from feature_selection import *
from prepare_data import *
from autoencoder import *
from mlflow_utils import init_mlflow, log_metrics_and_artifacts
import warnings
import argparse
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['KERAS_VERBOSE'] = '0'

# Configure logging
logging.basicConfig(
    filename='fraud_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def run_pipeline(config_path):
    try:
        # Load hyperparameters
        with open(config_path) as f:
            default_hyperparameters = json.load(f)
        
        logging.info(f"Default parameters: {default_hyperparameters}")
        
        # Initialize MLflow
        init_mlflow("Fraud_Pipeline")
        
        with mlflow.start_run() as run:
            try:
                logging.info("Running the final pipeline with default parameters and dropped features")
                
                # Log parameters
                mlflow.log_params({
                    "feature_selection_method": default_hyperparameters['feature_selection'],
                    "autoencoder_params": default_hyperparameters['autoencoder'],
                    "model_name": default_hyperparameters['model'],
                    "model_params": default_hyperparameters['model_params'],
                })
                
                # Run feature selection
                results = pipeline(default_hyperparameters, run.info.run_id)

                mlflow.keras.log_model(results["final_autoencoder"], "models/autoencoder")
                mlflow.keras.log_model(results["final_encoder_trained"], "models/encoder")
                mlflow.sklearn.log_model(results["reg_model"], "models/reg_model")

                mlflow.log_metrics({
                    "f1_score": results['f1'],
                    "precision_score": results['precision'],
                    "recall_score": results['recall']  # Convert to list for logging
                })

                mlflow.log_artifact(f"artifacts/{run.info.run_id}/predictions/confusion_matrix.png", "predictions")
                mlflow.log_artifact(f"artifacts/{run.info.run_id}/predictions/predictions.csv", "predictions")

                # mlflow.log_artifact("input data/dev.csv", "datasets")
                # mlflow.log_artifact("input data/oos.csv", "datasets")
                # mlflow.log_artifact("input data/oot.csv", "datasets")

                mlflow.log_artifact(f"artifacts/{run.info.run_id}/encoded data/encoded_dev.csv", "encoded datasets")
                mlflow.log_artifact(f"artifacts/{run.info.run_id}/encoded data/encoded_oos.csv", "encoded datasets")
                mlflow.log_artifact(f"artifacts/{run.info.run_id}/encoded data/encoded_oot.csv", "encoded datasets")

                mlflow.log_artifact(f"artifacts/{run.info.run_id}/saved best models/encoder_model.h5", "models")
                mlflow.log_artifact(f"artifacts/{run.info.run_id}/saved best models/autoencoder_model.h5", "models")
                mlflow.log_artifact(f"artifacts/{run.info.run_id}/saved best models/logistic_model.pkl", "models")
                
                # Log success status
                mlflow.log_param("status", "SUCCESS")
                
                logging.info(f"Pipeline Run Successfully Completed")
                logging.info(f"F1 Score: {results['f1']}")
                os.system('cls' if os.name == 'nt' else 'clear')
                return {
                    "status": "success",
                    "mlflow_run_id": run.info.run_id
                }
            except Exception as e:
                logging.error(f"Error during run pipeline: {str(e)}")
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
        result = run_pipeline(args.config)
        print(json.dumps(result))
        sys.stdout.flush()
    except Exception:
        print(json.dumps({"status": "error", "message": str(e)}))
        exit(1)