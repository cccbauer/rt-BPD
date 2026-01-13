
# Deployment code for PDA prediction

import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

def predict_pda(eeg_features, participant_id, model_file):
    '''
    Predict PDA from EEG features for a specific participant.
    
    Parameters:
    -----------
    eeg_features : dict
        Dictionary containing feature arrays
    participant_id : str
        Participant identifier (e.g., 'sub-dmnelf001')
    model_file : str
        Path to the saved model file
    
    Returns:
    --------
    pda_predictions : array
        Predicted PDA values
    '''
    
    # Load model
    with open(model_file, 'rb') as f:
        model_info = pickle.load(f)
    
    # Extract required features
    feature_vector = []
    for feat_name in model_info['feature_names']:
        if feat_name in eeg_features:
            feature_vector.append(eeg_features[feat_name])
        else:
            raise ValueError(f"Missing required feature: {feat_name}")
    
    # Combine features
    X = np.column_stack(feature_vector)
    
    # Apply outlier check (optional)
    # You may want to check if features are within reasonable ranges
    
    # Scale features
    X_scaled = model_info['scaler'].transform(X)
    
    # Predict
    pda_predictions = model_info['model'].predict(X_scaled)
    
    # Apply outlier threshold to predictions
    pda_predictions = np.clip(pda_predictions, 
                             -model_info['outlier_threshold'], 
                             model_info['outlier_threshold'])
    
    return pda_predictions

# Example usage:
# pda = predict_pda(eeg_features, 'sub-dmnelf001', 'deployment_model_sub-dmnelf001.pkl')
