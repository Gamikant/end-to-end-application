from prepare_data import *
from feature_selection import *
from autoencoder import *
import os
import seaborn as sns

def fs(default_hyperparameters):
    dev, oos, oot = load_data(default_hyperparameters["train_file"], 
                            default_hyperparameters["validation_file"], 
                            default_hyperparameters["test_file"])
        
    print("-------------------------------------------------")
    print("Data loaded successfully.")
    dev_scaled, oos_scaled, oot_scaled, scaler = standardize_data(dev, oos, oot)
    print("-------------------------------------------------")
    print("Data scaled successfully.")
    feature_names = list(dev.drop(['Class'], axis=1).columns)
    dev_F, dev_NF, oos_F, oos_NF = split_data(dev, oos)
    print("-------------------------------------------------")
    print("Data split into fraud and non-fraud successfully.")
    scaled_dev_F = scaler.transform(dev_F)
    scaled_dev_NF = scaler.transform(dev_NF)
    scaled_oos_F = scaler.transform(oos_F)
    scaled_oos_NF = scaler.transform(oos_NF)

    os.makedirs('feature selection', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    # Perform feature selection
    print("-------------------------------------------------")
    print("Performing feature selection...")
    # Get both autoencoders and feature selection results
    features_to_drop, importance_scores_normal, importance_scores_abnormal, \
    history_F, history_NF, autoencoder_F, autoencoder_NF = feature_selection(
        scaled_dev_F, scaled_dev_NF, scaled_oos_F, scaled_oos_NF,
        feature_names,
        default_hyperparameters['feature_selection'],
        default_hyperparameters['feature_threshold'],
        default_hyperparameters['autoencoder']['ratios'],
        default_hyperparameters['autoencoder']['hidden_activation'],
        default_hyperparameters['autoencoder']['dropout'],
        default_hyperparameters['autoencoder']['optimizer'],
        default_hyperparameters['autoencoder']['loss'],
        default_hyperparameters['autoencoder']['epochs'],
        default_hyperparameters['autoencoder']['batch_size']
    )

    with open('feature selection/features_dropped.txt', 'w') as f:
        f.write(str(features_to_drop))
    
    return {
        'features_dropped': features_to_drop,
        'importance_scores_normal': importance_scores_normal,
        'importance_scores_abnormal': importance_scores_abnormal,
        'history_F': history_F,
        'history_NF': history_NF,
        'autoencoder_F': autoencoder_F,
        'autoencoder_NF': autoencoder_NF,
        'dev': dev,
        'oos': oos,
        'oot': oot,
        'feature_names': feature_names
    }