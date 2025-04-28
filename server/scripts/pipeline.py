from prepare_data import *
from feature_selection import *
from autoencoder import *
from model import *
import os
import seaborn as sns
from mlflow_utils import init_mlflow, log_metrics_and_artifacts
import mlflow
import logging
from mlflow.tracking.fluent import _logger as mlflow_logger
import tensorflow as tf

def fs(default_hyperparameters, run_id):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')

    # Configure separate logger for pipeline
    pipeline_logger = logging.getLogger('feature_selection')
    pipeline_logger.setLevel(logging.WARNING)

    # Create file handler
    handler = logging.FileHandler('feature_selection.log')
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    pipeline_logger.addHandler(handler)
    
    dev, oos, oot = load_data(default_hyperparameters["train_file"], 
                            default_hyperparameters["validation_file"], 
                            default_hyperparameters["test_file"])
        
    pipeline_logger.info("-------------------------------------------------")
    pipeline_logger.info("Data loaded successfully.")
    dev_scaled, oos_scaled, oot_scaled, scaler = standardize_data(dev, oos, oot)
    pipeline_logger.info("-------------------------------------------------")
    pipeline_logger.info("Data scaled successfully.")
    feature_names = list(dev.drop(['Class'], axis=1).columns)
    dev_F, dev_NF, oos_F, oos_NF = split_data(dev, oos)
    pipeline_logger.info("-------------------------------------------------")
    pipeline_logger.info("Data split into fraud and non-fraud successfully.")
    scaled_dev_F = scaler.transform(dev_F)
    scaled_dev_NF = scaler.transform(dev_NF)
    scaled_oos_F = scaler.transform(oos_F)
    scaled_oos_NF = scaler.transform(oos_NF)

    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.makedirs(os.path.join(script_dir, f'../artifacts/{run_id}/feature selection'), exist_ok=True)
    os.makedirs(os.path.join(script_dir, f'../artifacts/{run_id}/encoded data'), exist_ok=True)
    os.makedirs(os.path.join(script_dir, f'../artifacts/{run_id}/figures'), exist_ok=True)
    os.makedirs(os.path.join(script_dir, f'../artifacts/{run_id}/predictions'), exist_ok=True)

    # Perform feature selection
    pipeline_logger.info("-------------------------------------------------")
    pipeline_logger.info("Performing feature selection...")
    # Get both autoencoders and feature selection results
    (
        features_to_drop, 
        importance_scores_normal,
        importance_scores_abnormal,
        history_F, 
        history_NF,
        autoencoder_F,  # Now receives the model
        autoencoder_NF   # Now receives the model
    ) = feature_selection(
        scaled_dev_F, scaled_dev_NF, scaled_oos_F, scaled_oos_NF,
        feature_names,
        default_hyperparameters['feature_selection'],
        default_hyperparameters['feature_threshold'],
        run_id,
        default_hyperparameters['autoencoder']['ratios'],
        default_hyperparameters['autoencoder']['hidden_activation'],
        default_hyperparameters['autoencoder']['dropout'],
        default_hyperparameters['autoencoder']['optimizer'],
        default_hyperparameters['autoencoder']['loss'],
        default_hyperparameters['autoencoder']['epochs'],
        default_hyperparameters['autoencoder']['batch_size']
    )

    with open(f"artifacts/{run_id}/feature selection/features_dropped.txt", "w") as f:
        for feature in features_to_drop:
            f.write(f"{feature}\n")
    
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

def pipeline(hyperparameters, run_id):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')

    # Configure separate logger for pipeline
    pipeline_logger = logging.getLogger('feature_selection')
    pipeline_logger.setLevel(logging.WARNING)

    # Create file handler
    handler = logging.FileHandler('feature_selection.log')
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    pipeline_logger.addHandler(handler)

    dev, oos, oot = load_data(hyperparameters["train_file"], hyperparameters["validation_file"], hyperparameters["test_file"])
    pipeline_logger.info("-------------------------------------------------")
    pipeline_logger.info("Data loaded successfully.")
    dev_scaled, oos_scaled, oot_scaled, scaler = standardize_data(dev, oos, oot)
    pipeline_logger.info("-------------------------------------------------")
    pipeline_logger.info("Data scaled successfully.")
    feature_names = list(dev.drop([hyperparameters["target_column"]], axis=1).columns)
    dev_F, dev_NF, oos_F, oos_NF = split_data(dev, oos)
    pipeline_logger.info("-------------------------------------------------")
    pipeline_logger.info("Data split into fraud and non-fraud successfully.")
    scaled_dev_F = scaler.transform(dev_F)
    scaled_dev_NF = scaler.transform(dev_NF)
    scaled_oos_F = scaler.transform(oos_F)
    scaled_oos_NF = scaler.transform(oos_NF)

    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.makedirs(os.path.join(script_dir, f'../artifacts/{run_id}/feature selection'), exist_ok=True)
    os.makedirs(os.path.join(script_dir, f'../artifacts/{run_id}/encoded data'), exist_ok=True)
    os.makedirs(os.path.join(script_dir, f'../artifacts/{run_id}/figures'), exist_ok=True)
    os.makedirs(os.path.join(script_dir, f'../artifacts/{run_id}/predictions'), exist_ok=True)
    
    if os.path.exists(f'artifacts/{run_id}/feature selection/features_dropped.txt'):
        with open(f'artifacts/{run_id}/feature selection/features_dropped.txt', 'r') as f:
            features_to_drop = [i.strip()[1:-1] for i in f.read()[1:-1].split(',')]

        pipeline_logger.info("-------------------------------------------------")
        pipeline_logger.info("Features to drop loaded from file.")
    else:
        pipeline_logger.info("-------------------------------------------------")
        pipeline_logger.info("No features to drop. Dropping 0 features")
        features_to_drop = []

    # Drop features and continue with pipeline
    pipeline_logger.info("-------------------------------------------------")
    pipeline_logger.info("Dropping features...")
    new_dev_scaled = drop_features(dev_scaled, features_to_drop, feature_names)
    new_oos_scaled = drop_features(oos_scaled, features_to_drop, feature_names)
    new_oot_scaled = drop_features(oot_scaled, features_to_drop, feature_names)
    pipeline_logger.info("Features dropped successfully.")
    
    # Continue with existing pipeline code...
    if hyperparameters['train_on'] == 'normal':
        train_on = drop_features(scaled_dev_NF, features_to_drop, feature_names)
        val_on = drop_features(scaled_oos_NF, features_to_drop, feature_names)
    elif hyperparameters['train_on'] == 'abnormal':
        train_on = drop_features(scaled_dev_F, features_to_drop, feature_names)
        val_on = drop_features(scaled_dev_F, features_to_drop, feature_names)
    
    pipeline_logger.info("-------------------------------------------------")
    pipeline_logger.info("Training main autoencoder with dropped features..")
    final_autoencoder = build_autoencoder(train_on.shape[1], 
                                                hyperparameters['autoencoder']['ratios'],
                                                hyperparameters['autoencoder']['hidden_activation'], 
                                                hyperparameters['autoencoder']['dropout'],
                                                hyperparameters['autoencoder']['optimizer'], 
                                                hyperparameters['autoencoder']['loss'])
    
    history_final = final_autoencoder.fit(
        train_on, train_on,
        epochs=hyperparameters['autoencoder']['epochs'],
        batch_size=hyperparameters['autoencoder']['batch_size'],
        validation_data=(val_on, val_on)
    )
    final_encoder_trained = Sequential(final_autoencoder.layers[:4])  # Extract encoder part

    pipeline_logger.info("Main autoencoder trained successfully.")

    pipeline_logger.info("-------------------------------------------------")
    pipeline_logger.info("Encoding data with trained autoencoder...")
    encoded_dev = pd.DataFrame(encode_data(final_encoder_trained, new_dev_scaled))
    encoded_oos = pd.DataFrame(encode_data(final_encoder_trained, new_oos_scaled))
    encoded_oot = pd.DataFrame(encode_data(final_encoder_trained, new_oot_scaled))

    # Saving encoded data
    encoded_dev.to_csv(f'artifacts/{run_id}/encoded data/encoded_dev.csv', index=False)
    encoded_oos.to_csv(f'artifacts/{run_id}/encoded data/encoded_oos.csv', index=False)
    encoded_oot.to_csv(f'artifacts/{run_id}/encoded data/encoded_oot.csv', index=False)
    pipeline_logger.info("Data encoded successfully.")

    encoded_dev2 = pd.concat([encoded_dev, dev[hyperparameters['target_column']]], axis=1)
    encoded_oos2 = pd.concat([encoded_oos, oos[hyperparameters['target_column']]], axis=1)
    encoded_train = pd.concat([encoded_dev2, encoded_oos2], axis=0)
    pipeline_logger.info("-------------------------------------------------")
    pipeline_logger.info("Training regression model...")
    reg_model, f1, precision, recall, confusion_mat, predictions_df = train_model(encoded_train.drop(columns=encoded_train.columns[-1]),
                                                        encoded_train[encoded_train.columns[-1]], 
                                                        encoded_oot, 
                                                        oot[hyperparameters['target_column']],
                                                        hyperparameters["model"], 
                                                        hyperparameters["model_params"],
                                                        hyperparameters["model_threshold"],
                                                        run_id)
    pipeline_logger.info("Regression model trained successfully.")
    pipeline_logger.info("-------------------------------------------------")
    pipeline_logger.info("Results:")
    pipeline_logger.info(f'1. f1_score = {f1}')
    pipeline_logger.info(f'2. precision = {precision}')
    pipeline_logger.info(f'3. recall = {recall}')
    pipeline_logger.info(f'4. confusion_matrix = {confusion_mat}')
    # Save confusion matrix as a seaborn heatmap image
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'artifacts/{run_id}/predictions/confusion_matrix.png')

    save_results(encoded_dev, encoded_oos, encoded_oot, 
                final_autoencoder, final_encoder_trained, reg_model, run_id)
    
    return {
        'encoded_dev': encoded_dev,
        'encoded_oos': encoded_oos,
        'encoded_oot': encoded_oot,
        'final_autoencoder': final_autoencoder,
        'final_encoder_trained': final_encoder_trained,
        'reg_model': reg_model,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': confusion_mat,
        'predictions_df': predictions_df
    }