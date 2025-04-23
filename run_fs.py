import json
import logging
import mlflow
import traceback
from pipeline import *
from feature_selection import *
from prepare_data import *
from autoencoder import *
from model import *

# Configure logging
logging.basicConfig(
    filename='feature_selection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

with open('default_hyperparameters.json', 'r') as f:
    default_hyperparameters = json.load(f)

logging.info(f"Default parameters: {default_hyperparameters}")

try:
    logging.info("Running initial feature selection using default parameters")
    features_dropped = fs(default_hyperparameters)
    logging.info(f"Feature selection completed. Dropped features: {features_dropped}")
except Exception as e:
    logging.error(f"Error during feature selection: {str(e)}")
    logging.error(traceback.format_exc())
    mlflow.end_run(status="FAILED")



