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
                mlflow.keras.log_model(results["autoencoder_F"], "fraud_autoencoder")
                mlflow.keras.log_model(results["autoencoder_NF"], "nonfraud_autoencoder")
                
                # Log datasets
                mlflow.log_artifact("input data/dev.csv", "datasets")
                mlflow.log_artifact("input data/oos.csv", "datasets")
                mlflow.log_artifact("input data/oot.csv", "datasets")

                mlflow.log_artifact('feature selection/abnormal_'+method+'_importance.csv', "feature_importance")
                mlflow.log_artifact('feature selection/normal_'+method+'_importance.csv', "feature_importance")
            
                # Log fraud autoencoder training metrics
                mlflow.log_metric("fraud_autoencoder_final_loss", results["history_F"].history['loss'][-1])
                mlflow.log_metric("fraud_autoencoder_final_val_loss", results["history_F"].history['val_loss'][-1])
                
                # Log non-fraud autoencoder training metrics
                mlflow.log_metric("nonfraud_autoencoder_final_loss", results["history_NF"].history['loss'][-1])
                mlflow.log_metric("nonfraud_autoencoder_final_val_loss", results["history_NF"].history['val_loss'][-1])
                
                # Log training curves for both autoencoders
                import matplotlib.pyplot as plt
                
                # Plot fraud autoencoder loss
                plt.figure(figsize=(10, 6))
                plt.plot(results["history_F"].history['loss'], label='Training Loss')
                plt.plot(results["history_F"].history['val_loss'], label='Validation Loss')
                plt.title('Fraud Autoencoder Training History')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.savefig('figures/fraud_autoencoder_loss.png')
                plt.close()
                
                # Plot non-fraud autoencoder loss
                plt.figure(figsize=(10, 6))
                plt.plot(results["history_NF"].history['loss'], label='Training Loss')
                plt.plot(results["history_NF"].history['val_loss'], label='Validation Loss')
                plt.title('Non-Fraud Autoencoder Training History')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.savefig('figures/nonfraud_autoencoder_loss.png')
                plt.close()
                
                # Log number of features dropped
                mlflow.log_metric("num_features_dropped", len(results["features_dropped"]))
                
                figure_paths = ['figures/normal_'+method+'_importance.png', 
                                'figures/abnormal_'+method+'_importance.png',
                                'figures/fraud_autoencoder_loss.png',
                                'figures/nonfraud_autoencoder_loss.png']
                
                for fig_path in figure_paths:
                    mlflow.log_artifact(fig_path, "important figures")
                
                # Log feature selection results
                with open("feature selection/features_dropped.txt", "r") as f:
                    mlflow.log_text(f.read(), "features_dropped.txt")
                
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