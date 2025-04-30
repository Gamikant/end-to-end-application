import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import joblib
import numpy as np
import os
import tensorflow as tf

def train_model(X_train, y_train, X_test, y_test, model_type, params, threshold, run_id):

    percentile = (1-y_train.mean()) * 100

    model = LogisticRegression() if model_type == 'LogisticRegression' else None
    if model is None:
        raise ValueError("Unsupported model type. Only LogisticRegression is supported.")
    model.set_params(**params)
    model.fit(X_train, y_train)
    
    if threshold:
        threshold = threshold
    else:
        threshold = np.percentile(model.predict_proba(X_test)[:, 1], percentile)
    
    # Ensure predictions are discrete class labels
    if hasattr(model, 'predict_proba'):
        y_pred = (model.predict_proba(X_test)[:, 1] > threshold).astype(int)
    else:
        y_pred = model.predict(X_test)

    predictions_df = pd.DataFrame({'Predictions': y_pred})
    predictions_df.to_csv(f'artifacts/{run_id}/predictions/predictions.csv', index=False)
    
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)
    return model, f1, precision, recall, confusion_mat, predictions_df

def save_results(encoded_dev, encoded_oos, encoded_oot, autoencoder_model, encoder_model, logistic_model, run_id):
    pd.DataFrame(encoded_dev).to_csv(f'artifacts/{run_id}/encoded data/encoded_dev.csv', index=False)
    pd.DataFrame(encoded_oos).to_csv(f'artifacts/{run_id}/encoded data/encoded_oos.csv', index=False)
    pd.DataFrame(encoded_oot).to_csv(f'artifacts/{run_id}/encoded data/encoded_oot.csv', index=False)
    encoder_model.save(f'artifacts/{run_id}/saved best models/encoder_model.h5')
    autoencoder_model.save(f'artifacts/{run_id}/saved best models/autoencoder_model.h5')
    joblib.dump(logistic_model, f'artifacts/{run_id}/saved best models/logistic_model.pkl')