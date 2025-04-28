import argparse
import json
import logging
import mlflow
import traceback
from pipeline import *
from feature_selection import *
from prepare_data import *
from autoencoder import *
from model import *
import sys
import os
import warnings
import tensorflow as tf

def main(config_path):
    with open(config_path) as f:
        default_hyperparameters = json.load(f)

    logging.info(f"Default parameters: {default_hyperparameters}")

    try:
        logging.info("Running initial feature selection using default parameters")
        features_dropped, run_id = fs(default_hyperparameters)
        logging.info(f"Feature selection completed. Dropped features: {features_dropped}")
        os.system('cls' if os.name == 'nt' else 'clear')
        return {
            "status": "success",
            "features_dropped": features_dropped,
            "mlflow_run_id": run_id,
            "metrics": {
                "features_removed": len(features_dropped)
            }
        }
    except Exception as e:
        logging.error(f"Error during feature selection: {str(e)}")
        logging.error(traceback.format_exc())
        # mlflow.end_run(status="FAILED")
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    os.environ['KERAS_VERBOSE'] = '0'
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(3)

    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('mlflow').setLevel(logging.WARNING)

    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        filename='feature_selection.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    try:
        result = main(args.config)
        print(json.dumps(result))
        # logging.info(json.dumps(result))
    except Exception as e:
        logging.info(json.dumps({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }))
        sys.exit(1)





