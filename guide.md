# Data Preparation Guide

## Input Data Requirements

### Things you must provide
Your clean train, validatio & test files in the `input data` folder:
```
input data/
├── train.csv           # Training dataset
├── validation.csv      # Validation dataset
└── test.csv            # Test dataset
```
Then specify the file names and the target column name in the `default_hyperparameters.json` file:
```python
{
    "train_file": "input data/train.csv",             # Train file name
    "validation_file": "input data/validation.csv",   # Validation file name
    "test_file": "input data/test.csv",               # Test file name
    "target_column": "Class",                         # Target column     
    # Rest of the parameters
}
```

### Data Requirements
1. **Format**
   - CSV files only
   - Comma-separated values
   - UTF-8 encoding

2. **Data Quality**
   - No missing values (nulls)
   - Only numeric features (categorical allowed)
   - No text variables
   - Binary target variable (0/1)

3. **Column Names**
   - Unique column names
   - No spaces in column names
   - Target variable must be specified

### Example Data Format

```csv
feature1,feature2,feature3,target
0.234,1.456,0,0
0.567,2.123,0,1
0.789,0.234,1,0
```

---

# Fraud Detection Pipeline

## Overview
An end-to-end machine learning pipeline for supervised anomaly detection, specifically designed for imbalanced datasets. The pipeline consists of:
1. Feature Selection
2. Fraud Detection Model Training

## Prerequisites
- Python 3.8+
- Windows 10 or later
- Minimum 8GB RAM
- Clean input data (see Data Preparation Guide above)

## Installation

```powershell
# Create virtual environment
python -m venv venv

# Activate environment
.\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## Project Structure
```
Application/
├── input data/           # Your input datasets
├── feature selection/    # Feature selection results
├── figures/             # Performance plots
├── encoded data/        # Processed datasets
├── predictions/         # Model predictions
├── save best models/    # Trained models
├── mlflow/              # Experiment tracking
├── run_fs.py           # Feature selection script
└── run_fp.py           # Fraud detection script
```

## Usage

### 1. Feature Selection
```powershell
python run_fs.py
```

Outputs:
- `feature selection/features_dropped.txt`
- `feature selection/abnormal_*_importance.csv`
- `feature selection/normal_*_importance.csv`
- Various plots in `figures` folder

### 2. Fraud Detection
```powershell
python run_fp.py
```

### 3. View Results
```powershell
# Start MLflow UI
mlflow ui --port 5000
```
Access dashboard at: http://localhost:5000

## Output Files

### Feature Selection
- `features_dropped.txt`: Features to remove
- `*_importance.csv`: Feature importance scores
- Performance plots in figures

### Model Results
- Predictions in predictions
    - Confusion matrix
    - Test data predictions (0/1)
- Model files in `saved best models/`
- Metrics in MLflow dashboard

## Monitoring & Logging
- Check feature_selection.log for features selection logs
- Check fraud_pipeline.log for fraud detection logs
- Monitor MLflow for metrics
- Review terminal output

## Troubleshooting

### Common Issues
1. **Memory Errors**
   - Close other applications
   - Reduce batch size in default_hyperparameters.json

2. **Import Errors**
   ```powershell
   # Reinstall requirements
   pip install -r requirements.txt
   ```

3. **Missing Folders**
   ```powershell
   # Create directories
   mkdir "input data" "feature selection" "figures" "encoded data" "predictions" "save best models" "mlflow"
   ```

## Best Practices
1. Always backup input data
2. Monitor system resources
3. Review feature importance scores
4. Validate model performance
5. Check logs regularly

## Support
For issues:
1. Check log files
2. Review MLflow dashboard
3. Verify input data format
4. Monitor system resources

## Version Control (Optional)
```powershell
# Initialize DVC
dvc init

# Track input data
dvc add "input data/"
```

For detailed implementation questions or issues, please check the documentation or raise an issue.