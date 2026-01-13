"""
Automated EEG Channel Cleaning Module
====================================
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def automated_channel_cleaning(raw, ecg_channel_names=None, z_score_threshold=5.0, 
                              correlation_threshold=0.2, powerline_threshold=15.0,
                              save_report=True, report_path=None):
    """
    Simplified automated EEG channel cleaning.
    """
    
    # Common ECG channel name patterns
    ecg_patterns = ['ECG', 'EKG', 'ecg', 'ekg', 'ECG1', 'ECG2', 'EKG1', 'EKG2']
    
    # Find and remove ECG channels
    ecg_channels_found = []
    if ecg_channel_names is None:
        for ch_name in raw.ch_names:
            if any(pattern in ch_name for pattern in ecg_patterns):
                ecg_channels_found.append(ch_name)
    else:
        ecg_channels_found = ecg_channel_names
    
    # Remove ECG channels
    if ecg_channels_found:
        print(f"Removing ECG channels: {ecg_channels_found}")
        raw = raw.copy().drop_channels(ecg_channels_found)
    else:
        print("No ECG channels found to remove")
        raw = raw.copy()
    
    # Get data for analysis
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    
    bad_channels = []
    
    # 1. Variance-based detection
    channel_vars = np.var(data, axis=1)
    z_scores = np.abs((channel_vars - np.median(channel_vars)) / np.median(channel_vars))
    
    bad_by_variance = []
    for idx, (z_score, ch_name) in enumerate(zip(z_scores, raw.ch_names)):
        if z_score > z_score_threshold:
            bad_by_variance.append(ch_name)
            print(f"  - {ch_name}: High variance (z-score: {z_score:.2f})")
        elif channel_vars[idx] < np.median(channel_vars) * 0.01:
            bad_by_variance.append(ch_name)
            print(f"  - {ch_name}: Nearly flat")
    
    bad_channels.extend(bad_by_variance)
    
    # 2. Correlation-based detection
    downsampled_data = data[:, ::10]
    corr_matrix = np.corrcoef(downsampled_data)
    
    bad_by_correlation = []
    for idx, ch_name in enumerate(raw.ch_names):
        if ch_name in bad_channels:
            continue
        
        correlations = corr_matrix[idx, :]
        correlations[idx] = 0
        mean_corr = np.mean(np.abs(correlations))
        
        if mean_corr < correlation_threshold:
            bad_by_correlation.append(ch_name)
            print(f"  - {ch_name}: Low correlation (mean: {mean_corr:.2f})")
    
    bad_channels.extend(bad_by_correlation)
    
    # Remove duplicates
    bad_channels = list(set(bad_channels))
    
    # Mark bad channels
    raw.info['bads'] = bad_channels
    
    print(f"\nTotal bad channels detected: {len(bad_channels)}")
    
    # Simple QC stats
    qc_stats = {
        'bad_channels': bad_channels,
        'bad_by_variance': bad_by_variance,
        'bad_by_correlation': bad_by_correlation,
        'ecg_channels_removed': ecg_channels_found
    }
    
    # Optional: Save a simple plot
    if save_report and report_path:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(channel_vars)), channel_vars)
        ax.set_xlabel('Channel')
        ax.set_ylabel('Variance')
        ax.set_title('Channel Variance Distribution')
        for i, ch in enumerate(bad_channels):
            if ch in raw.ch_names:
                idx = raw.ch_names.index(ch)
                ax.bar(idx, channel_vars[idx], color='red')
        plt.tight_layout()
        plt.savefig(report_path)
        plt.close()
    
    return raw, bad_channels, qc_stats
