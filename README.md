# Feature Selection + Fraud Detection Pipeline

This pipeline performs feature selection using autoencoders on financial transaction data, with both fraud and non-fraud cases handled separately. The pipeline uses MLflow for experiment tracking and DVC for data version control.
It then drops those features and runs the main fraud detection pipeline.

## Prerequisites

- Python 3.8+
- Git
- DVC
- MLflow

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Gamikant/end-to-end-application
cd Application
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Data Setup

1. Place your input data files in the `input_data` folder with the following structure:
```
input_data/
    ├── dev.csv     # Training data
    ├── oos.csv     # Out-of-sample validation data
    └── oot.csv     # Out-of-time test data
```

2. Initialize DVC and add your data:
```bash
git init
dvc init
dvc add input_data
git add input_data.dvc .dvc
git commit -m "Initialize DVC with data"
```

## Configuration

Modify the `default_hyperparameters.json` file to adjust:
- File paths
- Feature selection parameters
- Autoencoder architecture
- Model parameters

Example configuration:
```json
{
    "train_file": "input_data/dev.csv",
    "validation_file": "input_data/oos.csv",
    "test_file": "input_data/oot.csv",
    "feature_selection": "fpi",
    "feature_threshold": 0.1,
    "autoencoder": {
        "ratios": [0.8, 0.5, 0.2],
        "epochs": 10,
        "batch_size": 32
        // ... other parameters
    },
    "model": {
        // ... all parameters
    }
}
```

## Running the Pipeline

1. Start the MLflow tracking server:
```bash
mlflow ui --port 5000
```

2. In a new terminal, run the feature selection pipeline:
```bash
python run_fs.py
```
3. In the same terminal after feature selection, run the fraud detection pipeline:
```bash
python run_fp.py
```

3. View results:
- MLflow UI: http://localhost:5000
- Check generated files in:
  - `feature_selection/` - Feature importance scores
  - `figures/` - Visualizations
  - `saved best models/` - All fraud detection pipeline models saved here
  - `encoded data/` - Dataset encoded using encoder
  - `predictions/` - Predictions on test data
  - `feature_selection.log` - Execution logs for feature selection process
  - `fraud_pipeline.log` - Excecution logs for fraud detection pipeline

## Output Files

The pipeline generates:
- Feature importance scores (CSV)
- Feature importance visualizations
- Autoencoder training history plots
- Autoencoder & Encoder model files
- Logistic Regression model files
- Prediction files
- Encoded data
- List of dropped features
- Detailed logs

## For Website Integration

1. API Integration:
   - Use the MLflow REST API to track experiments
   - Example endpoint: `http://localhost:5000/api/2.0/mlflow`

2. Real-time Monitoring:
```python
# Example code to get real-time metrics
from mlflow.tracking import MlflowClient

client = MlflowClient()
run = client.get_run(run_id)
metrics = run.data.metrics
```

3. Result Retrieval:
```python
# Example code to get feature selection results
with open("feature_selection/features_dropped.txt", "r") as f:
    dropped_features = f.read()
```

## Directory Structure
```
Application/
├── input_data/            # Input data files
├── feature_selection/     # Generated results
├── figures/              # Generated plots
├── encoded data/         # Data encoded by encoder
├── predictions/          # Predictions on test data
├── save best models/     # All final pipeline models stored here
├── model.py              # Regression model utilities
├── run_fp.py            # Main script for Part 2
├── run_fs.py            # Main script for Part 1
├── pipeline.py          # Pipeline implementation
├── feature_selection.py # Feature selection logic
├── autoencoder.py      # Autoencoder model
├── prepare_data.py     # Data preparation
└── mlflow_utils.py     # MLflow utilities

```

## Troubleshooting

1. MLflow Connection Issues:
   - Ensure MLflow server is running on port 5000
   - Check firewall settings

2. Data Loading Issues:
   - Verify CSV file format
   - Check column names match expected format

3. Memory Issues:
   - Adjust batch_size in hyperparameters
   - Use smaller dataset for testing

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request