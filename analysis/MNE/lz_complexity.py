"""
Lempel-Ziv Complexity Module
===========================
"""

import numpy as np
from tqdm import tqdm

def compute_lz_features_matlab_style(eeg_data, fs, window_length=1.0, overlap=0.5, 
                                   complexity_type='exhaustive', use_fast=True):
    """
    Simplified Lempel-Ziv complexity computation.
    """
    
    n_channels, n_samples = eeg_data.shape
    window_samples = int(window_length * fs)
    step_samples = int(window_samples * (1 - overlap))
    n_windows = (n_samples - window_samples) // step_samples + 1
    
    print(f"Computing LZ complexity ({complexity_type})...")
    
    # Pre-allocate arrays
    lz_complexity = np.zeros((n_windows, n_channels))
    lz_times = np.zeros(n_windows)
    
    # Pre-compute channel medians for binarization
    channel_medians = np.median(eeg_data, axis=1)
    
    # Process windows
    for win_idx in tqdm(range(n_windows), desc=f"LZ {complexity_type}"):
        start = win_idx * step_samples
        end = start + window_samples
        
        if end > n_samples:
            break
            
        lz_times[win_idx] = start / fs + window_length / 2
        
        # Compute LZ for each channel
        for ch in range(n_channels):
            # Extract and binarize window
            window_data = eeg_data[ch, start:end]
            binary_seq = (window_data > channel_medians[ch]).astype(int)
            
            # Simplified LZ complexity calculation
            binary_str = ''.join(binary_seq.astype(str))
            
            # Count unique substrings (simplified approach)
            substrings = set()
            max_len = min(10, len(binary_str) // 4)
            
            for length in range(1, max_len + 1):
                for i in range(len(binary_str) - length + 1):
                    substrings.add(binary_str[i:i+length])
            
            c = len(substrings)
            if len(binary_str) > 1:
                c_norm = c / (len(binary_str) / np.log2(len(binary_str)))
            else:
                c_norm = c
            
            lz_complexity[win_idx, ch] = c_norm
    
    # Trim to actual number of windows
    actual_windows = win_idx
    
    features = {
        'lz_complexity_mean': np.mean(lz_complexity[:actual_windows], axis=1),
        'lz_complexity_std': np.std(lz_complexity[:actual_windows], axis=1),
        'lz_complexity_max': np.max(lz_complexity[:actual_windows], axis=1),
        'lz_complexity_min': np.min(lz_complexity[:actual_windows], axis=1),
        'lz_times': lz_times[:actual_windows]
    }
    
    # Add spatial features if enough channels
    if n_channels >= 20:
        features['lz_complexity_frontal'] = np.mean(lz_complexity[:actual_windows, :10], axis=1)
        features['lz_complexity_posterior'] = np.mean(lz_complexity[:actual_windows, -10:], axis=1)
        features['lz_frontal_posterior_diff'] = (
            features['lz_complexity_frontal'] - features['lz_complexity_posterior']
        )
    
    return features
