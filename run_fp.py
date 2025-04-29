import json
import logging
import mlflow
import traceback
from pipeline import *
from feature_selection import *
from prepare_data import *
from autoencoder import *
from mlflow_utils import *

# Configure logging
logging.basicConfig(
    filename='fraud_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def run_pipeline():
    try:
        # Load hyperparameters
        with open('default_hyperparameters.json', 'r') as f:
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
                results = pipeline(default_hyperparameters)

                mlflow.keras.log_model(results["final_autoencoder"], "models/autoencoder")
                mlflow.keras.log_model(results["final_encoder_trained"], "models/encoder")
                mlflow.sklearn.log_model(results["reg_model"], "models/reg_model")

                mlflow.log_metrics({
                    "f1_score": results['f1'],
                    "precision_score": results['precision'],
                    "recall_score": results['recall']  # Convert to list for logging
                })

                mlflow.log_artifact("predictions/confusion_matrix.png", "predictions")
                mlflow.log_artifact("predictions/predictions.csv", "predictions")

                mlflow.log_artifact("input data/dev.csv", "datasets")
                mlflow.log_artifact("input data/oos.csv", "datasets")
                mlflow.log_artifact("input data/oot.csv", "datasets")

                mlflow.log_artifact("encoded data/encoded_dev.csv", "encoded datasets")
                mlflow.log_artifact("encoded data/encoded_oos.csv", "encoded datasets")
                mlflow.log_artifact("encoded data/encoded_oot.csv", "encoded datasets")

                mlflow.log_artifact("saved best models/encoder_model.h5", "models")
                mlflow.log_artifact("saved best models/autoencoder_model.h5", "models")
                mlflow.log_artifact("saved best models/logistic_model.pkl", "models")
                
                # Log success status
                mlflow.log_param("status", "SUCCESS")
                
                logging.info(f"Pipeline Run Successfully Completed")
                logging.info(f"F1 Score: {results['f1']}")
                
            except Exception as e:
                logging.error(f"Error during run pipeline: {str(e)}")
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
        run_pipeline()
    except Exception:
        exit(1)