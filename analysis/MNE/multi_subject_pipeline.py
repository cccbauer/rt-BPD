"""
Multi-Subject Pipeline for PDA Prediction from EEG Data
======================================================

This pipeline trains a robust model across multiple subjects to predict
Positive Diametric Activity (PDA) between DMN and CEN from EEG features.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import joblib
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from scipy.signal import stft
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


class MultiSubjectPDAPipeline:
    """
    Complete pipeline for multi-subject PDA prediction from EEG data.
    """
    
    def __init__(self, base_dir='./data', output_dir='./results'):
        """
        Initialize the pipeline.
        
        Parameters:
        -----------
        base_dir : str
            Base directory containing subject data
        output_dir : str
            Output directory for results
        """
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.subjects_data = {}
        self.combined_features = None
        self.combined_targets = None
        self.subject_ids = None
        self.trained_model = None
        self.scaler = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Pipeline parameters
        self.params = {
            'eeg': {
                'z_score_threshold': 5.0,
                'correlation_threshold': 0.2,
                'powerline_threshold': 15.0
            },
            'pda': {
                'hrf_delay': 5.0,
                'target_fs': 1.0
            },
            'stft': {
                'window_sec': 1.0,
                'overlap': 0.5
            },
            'lz': {
                'window_length': 2.0,
                'overlap': 0.5,
                'complexity_type': 'exhaustive',
                'use_fast': True
            },
            'features': {
                'n_top_features': 20,
                'use_lagged_features': True,
                'lag_samples': [1, 2, 3, 4, 5, 6]
            },
            'model': {
                'test_size': 0.2,
                'cv_folds': 5,
                'random_state': 42
            }
        }
    
    def process_single_subject(self, subject_id, eeg_file, pda_file):
        """
        Process a single subject's data.
        
        Parameters:
        -----------
        subject_id : str
            Subject identifier
        eeg_file : str
            Path to EEG file
        pda_file : str
            Path to PDA file (CSV with CEN and DMN columns)
        
        Returns:
        --------
        dict : Subject data including features and targets
        """
        print(f"\n{'='*70}")
        print(f"Processing Subject: {subject_id}")
        print(f"{'='*70}")
        
        subject_output_dir = os.path.join(self.output_dir, subject_id)
        os.makedirs(subject_output_dir, exist_ok=True)
        
        # 1. Load and clean EEG data
        print("\n1. Loading and cleaning EEG data...")
        raw_processed, bad_channels, qc_stats = self._clean_eeg_data(
            eeg_file, subject_output_dir
        )
        
        # 2. Load PDA data
        print("\n2. Loading PDA data...")
        pda_df = pd.read_csv(pda_file)
        
        # Calculate PDA (CEN - DMN)
        if 'cen' in pda_df.columns and 'dmn' in pda_df.columns:
            pda_signal = pda_df['cen'].values - pda_df['dmn'].values
        else:
            raise ValueError(f"PDA file must contain 'cen' and 'dmn' columns")
        
        # 3. Align PDA with EEG
        print("\n3. Aligning PDA with EEG...")
        pda_aligned_z, pda_time_aligned = self._align_pda_to_eeg(
            pda_signal, raw_processed
        )
        
        # 4. Extract features
        print("\n4. Extracting features...")
        
        # 4a. STFT features
        stft_features = self._extract_stft_features(
            raw_processed, pda_aligned_z, pda_time_aligned
        )
        
        # 4b. LZ complexity features
        lz_features = self._extract_lz_features(
            raw_processed, pda_aligned_z, pda_time_aligned
        )
        
        # 4c. Advanced features (spatial, connectivity, etc.)
        advanced_features = self._extract_advanced_features(
            raw_processed, stft_features, pda_aligned_z, pda_time_aligned
        )
        
        # 5. Combine all features
        print("\n5. Combining features...")
        all_features = self._combine_features(
            stft_features, lz_features, advanced_features
        )
        
        # 6. Save subject data
        subject_data = {
            'subject_id': subject_id,
            'features': all_features['feature_matrix'],
            'feature_names': all_features['feature_names'],
            'target': pda_aligned_z,
            'time_points': pda_time_aligned,
            'eeg_info': {
                'n_channels': len(raw_processed.ch_names),
                'bad_channels': bad_channels,
                'duration': raw_processed.times[-1]
            }
        }
        
        # Save to file
        np.savez(
            os.path.join(subject_output_dir, f'{subject_id}_processed.npz'),
            **subject_data
        )
        
        print(f"\nSubject {subject_id} processing complete!")
        print(f"Features shape: {subject_data['features'].shape}")
        print(f"Target shape: {subject_data['target'].shape}")
        
        return subject_data
    
    def _clean_eeg_data(self, eeg_file, output_dir):
        """Clean EEG data using automated bad channel detection."""
        from automated_channel_cleaning import automated_channel_cleaning
        
        # Load raw data
        raw = mne.io.read_raw_edf(eeg_file, preload=True)
        
        # Run automated cleaning
        raw_cleaned, bad_channels, qc_stats = automated_channel_cleaning(
            raw,
            z_score_threshold=self.params['eeg']['z_score_threshold'],
            correlation_threshold=self.params['eeg']['correlation_threshold'],
            powerline_threshold=self.params['eeg']['powerline_threshold'],
            save_report=True,
            report_path=os.path.join(output_dir, 'qc_report.png')
        )
        
        # Remove bad channels
        good_channels = [ch for ch in raw_cleaned.ch_names if ch not in bad_channels]
        raw_processed = raw_cleaned.copy().pick_channels(good_channels)
        
        return raw_processed, bad_channels, qc_stats
    
    def _align_pda_to_eeg(self, pda_signal, raw_data):
        """Align PDA signal to EEG timeline with HRF delay."""
        # Original PDA timing
        pda_fs = 1 / 1.2  # ~0.833 Hz
        pda_time_orig = np.arange(len(pda_signal)) * 1.2
        
        # Apply HRF shift
        hrf_delay = self.params['pda']['hrf_delay']
        pda_time_shifted = pda_time_orig - hrf_delay
        
        # Target time points
        eeg_duration = raw_data.times[-1]
        target_fs = self.params['pda']['target_fs']
        target_time = np.arange(0, min(eeg_duration, pda_time_shifted[-1]), 1/target_fs)
        
        # Only use valid PDA samples
        valid_idx = pda_time_shifted >= 0
        
        # Interpolate
        interp_func = interp1d(
            pda_time_shifted[valid_idx], 
            pda_signal[valid_idx], 
            kind='cubic',
            bounds_error=False,
            fill_value='extrapolate'
        )
        pda_resampled = interp_func(target_time)
        
        # Z-score normalization
        pda_resampled_z = (pda_resampled - np.mean(pda_resampled)) / np.std(pda_resampled)
        
        return pda_resampled_z, target_time
    
    def _extract_stft_features(self, raw_processed, pda_aligned_z, pda_time_aligned):
        """Extract STFT-based band power features."""
        fs = raw_processed.info['sfreq']
        
        # STFT parameters
        window_sec = self.params['stft']['window_sec']
        nperseg = int(window_sec * fs)
        noverlap = int(self.params['stft']['overlap'] * fs)
        
        # Get EEG data
        picks_eeg = mne.pick_types(raw_processed.info, eeg=True)
        eeg_data = raw_processed.get_data(picks=picks_eeg)
        
        # Compute STFT
        f, t_stft, Zxx = stft(eeg_data, fs=fs, nperseg=nperseg, 
                             noverlap=noverlap, axis=1)
        
        # Calculate power
        power = np.abs(Zxx) ** 2
        power = power.transpose(2, 0, 1)  # (n_windows, n_channels, n_freqs)
        
        # Extract band powers
        bands = {
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'low_gamma': (30, 50)
        }
        
        band_powers = {}
        for band_name, (low, high) in bands.items():
            band_mask = (f >= low) & (f <= high)
            band_powers[band_name] = np.mean(power[:, :, band_mask], axis=2)
        
        # Add ratio features
        eps = 1e-10
        band_powers['theta_alpha_ratio'] = band_powers['theta'] / (band_powers['alpha'] + eps)
        band_powers['beta_alpha_ratio'] = band_powers['beta'] / (band_powers['alpha'] + eps)
        
        return {
            'band_powers': band_powers,
            'stft_times': t_stft,
            'power': power,
            'freqs': f
        }
    
    def _extract_lz_features(self, raw_processed, pda_aligned_z, pda_time_aligned):
        """Extract Lempel-Ziv complexity features."""
        from lz_complexity import compute_lz_features_matlab_style
        
        picks_eeg = mne.pick_types(raw_processed.info, eeg=True)
        eeg_data = raw_processed.get_data(picks=picks_eeg)
        fs = raw_processed.info['sfreq']
        
        # Compute LZ complexity
        features = compute_lz_features_matlab_style(
            eeg_data, fs,
            window_length=self.params['lz']['window_length'],
            overlap=self.params['lz']['overlap'],
            complexity_type=self.params['lz']['complexity_type'],
            use_fast=self.params['lz']['use_fast']
        )
        
        return features
    
    def _extract_advanced_features(self, raw_processed, stft_features, 
                                  pda_aligned_z, pda_time_aligned):
        """Extract advanced features including spatial and connectivity."""
        fs = raw_processed.info['sfreq']
        picks_eeg = mne.pick_types(raw_processed.info, eeg=True)
        channel_names = [raw_processed.ch_names[i] for i in picks_eeg]
        
        band_powers = stft_features['band_powers']
        stft_times = stft_features['stft_times']
        
        features = {}
        
        # Spatial features
        frontal_channels = []
        parietal_channels = []
        
        for i, ch in enumerate(channel_names):
            ch_upper = ch.upper()
            if any(x in ch_upper for x in ['FP1', 'FP2', 'F3', 'F4', 'F7', 'F8', 'FZ']):
                frontal_channels.append(i)
            if any(x in ch_upper for x in ['P3', 'P4', 'PZ', 'P7', 'P8']):
                parietal_channels.append(i)
        
        # Frontal-Parietal alpha gradient
        if frontal_channels and parietal_channels:
            frontal_alpha = np.mean(band_powers['alpha'][:, frontal_channels], axis=1)
            parietal_alpha = np.mean(band_powers['alpha'][:, parietal_channels], axis=1)
            features['frontal_parietal_alpha_gradient'] = parietal_alpha - frontal_alpha
        
        # Network indices
        if frontal_channels and parietal_channels:
            dmn_index = (np.mean(band_powers['alpha'][:, parietal_channels], axis=1) - 
                        np.mean(band_powers['beta'][:, frontal_channels], axis=1))
            cen_index = (np.mean(band_powers['beta'][:, frontal_channels], axis=1) - 
                        np.mean(band_powers['alpha'][:, parietal_channels], axis=1))
        else:
            dmn_index = np.mean(band_powers['alpha'], axis=1) - np.mean(band_powers['beta'], axis=1)
            cen_index = np.mean(band_powers['beta'], axis=1) - np.mean(band_powers['alpha'], axis=1)
        
        features['dmn_index'] = dmn_index
        features['cen_index'] = cen_index
        features['network_competition'] = cen_index - dmn_index
        
        return {
            'features': features,
            'times': stft_times
        }
    
    def _combine_features(self, stft_features, lz_features, advanced_features):
        """Combine all features and align with PDA."""
        from scipy.interpolate import interp1d
        
        all_aligned_features = {}
        
        # Add STFT band powers
        for band_name, band_data in stft_features['band_powers'].items():
            # Mean across channels
            feat_mean = np.mean(band_data, axis=1)
            all_aligned_features[f'{band_name}_mean'] = feat_mean
            
            # Std across channels
            feat_std = np.std(band_data, axis=1)
            all_aligned_features[f'{band_name}_std'] = feat_std
        
        # Add LZ features
        for feat_name, feat_data in lz_features.items():
            if feat_name != 'lz_times' and isinstance(feat_data, np.ndarray):
                all_aligned_features[feat_name] = feat_data
        
        # Add advanced features
        for feat_name, feat_data in advanced_features['features'].items():
            if isinstance(feat_data, np.ndarray):
                all_aligned_features[feat_name] = feat_data
        
        # Get common time array (use STFT times as reference)
        ref_times = stft_features['stft_times']
        
        # Create feature matrix
        feature_list = []
        feature_names = []
        
        for feat_name, feat_data in all_aligned_features.items():
            if feat_data.ndim == 1 and len(feat_data) == len(ref_times):
                feature_list.append(feat_data.reshape(-1, 1))
                feature_names.append(feat_name)
        
        feature_matrix = np.hstack(feature_list)
        
        # Add lagged features if requested
        if self.params['features']['use_lagged_features']:
            lag_samples = self.params['features']['lag_samples']
            key_features = ['network_competition', 'cen_index', 'dmn_index', 
                          'beta_alpha_ratio_mean', 'lz_complexity_mean']
            
            for lag in lag_samples:
                for feat in key_features:
                    if feat in feature_names:
                        feat_idx = feature_names.index(feat)
                        lagged_feat = np.roll(feature_matrix[:, feat_idx], lag)
                        lagged_feat[:lag] = lagged_feat[lag]
                        
                        feature_list.append(lagged_feat.reshape(-1, 1))
                        feature_names.append(f'{feat}_lag{lag}s')
            
            feature_matrix = np.hstack(feature_list)
        
        return {
            'feature_matrix': feature_matrix,
            'feature_names': feature_names
        }
    
    def process_all_subjects(self, subject_files):
        """
        Process all subjects in the dataset.
        
        Parameters:
        -----------
        subject_files : list of tuples
            List of (subject_id, eeg_file, pda_file) tuples
        """
        print(f"\n{'='*70}")
        print(f"PROCESSING {len(subject_files)} SUBJECTS")
        print(f"{'='*70}")
        
        for subject_id, eeg_file, pda_file in subject_files:
            try:
                subject_data = self.process_single_subject(
                    subject_id, eeg_file, pda_file
                )
                self.subjects_data[subject_id] = subject_data
            except Exception as e:
                print(f"\nError processing subject {subject_id}: {e}")
                continue
        
        print(f"\n\nSuccessfully processed {len(self.subjects_data)} subjects")
    
    def combine_subjects_data(self):
        """Combine data from all subjects for training."""
        print(f"\n{'='*70}")
        print("COMBINING MULTI-SUBJECT DATA")
        print(f"{'='*70}")
        
        all_features = []
        all_targets = []
        all_subject_ids = []
        
        # Get common feature names (use first subject as reference)
        ref_subject = list(self.subjects_data.values())[0]
        common_features = ref_subject['feature_names']
        
        for subject_id, data in self.subjects_data.items():
            # Ensure feature consistency
            if data['feature_names'] == common_features:
                all_features.append(data['features'])
                all_targets.append(data['target'])
                all_subject_ids.extend([subject_id] * len(data['target']))
            else:
                print(f"Warning: Feature mismatch for subject {subject_id}")
        
        # Combine arrays
        self.combined_features = np.vstack(all_features)
        self.combined_targets = np.hstack(all_targets)
        self.subject_ids = np.array(all_subject_ids)
        self.feature_names = common_features
        
        print(f"\nCombined data shape:")
        print(f"  Features: {self.combined_features.shape}")
        print(f"  Targets: {self.combined_targets.shape}")
        print(f"  Subjects: {len(np.unique(self.subject_ids))}")
    
    def train_models(self):
        """Train and evaluate multiple models with cross-validation."""
        print(f"\n{'='*70}")
        print("TRAINING MULTI-SUBJECT MODELS")
        print(f"{'='*70}")
        
        # Feature selection
        top_features, correlations = self._select_top_features()
        
        # Create feature matrix with selected features
        feature_indices = [self.feature_names.index(f) for f in top_features]
        X = self.combined_features[:, feature_indices]
        y = self.combined_targets
        groups = self.subject_ids
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Models to evaluate
        models = {
            'Ridge': Ridge(alpha=1.0),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000),
            'RandomForest': RandomForestRegressor(
                n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=42, n_jobs=-1
            )
        }
        
        # Group K-Fold for subject-wise cross-validation
        gkf = GroupKFold(n_splits=self.params['model']['cv_folds'])
        
        results = {}
        
        print("\nCross-Validation Results:")
        print("-" * 50)
        
        for model_name, model in models.items():
            # Cross-validation with groups
            cv_scores = []
            
            for train_idx, val_idx in gkf.split(X_scaled, y, groups):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                corr, _ = pearsonr(y_val, y_pred)
                cv_scores.append(corr)
            
            cv_scores = np.array(cv_scores)
            
            # Fit on full data
            model.fit(X_scaled, y)
            y_pred_full = model.predict(X_scaled)
            
            results[model_name] = {
                'model': model,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'full_correlation': pearsonr(y, y_pred_full)[0],
                'full_r2': r2_score(y, y_pred_full)
            }
            
            print(f"\n{model_name}:")
            print(f"  CV correlations: {cv_scores.round(3)}")
            print(f"  Mean CV r: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
            print(f"  Full data r: {results[model_name]['full_correlation']:.3f}")
        
        # Select best model
        best_model_name = max(results.keys(), 
                            key=lambda x: results[x]['cv_mean'])
        
        print(f"\n{'='*50}")
        print(f"BEST MODEL: {best_model_name}")
        print(f"Mean CV correlation: {results[best_model_name]['cv_mean']:.3f}")
        print(f"{'='*50}")
        
        # Store results
        self.model_results = results
        self.best_model = results[best_model_name]['model']
        self.selected_features = top_features
        
        # Create visualization
        self._plot_model_results(results, X_scaled, y, groups)
        
        return results
    
    def _select_top_features(self):
        """Select top features based on correlation with target."""
        correlations = {}
        
        for i, feat_name in enumerate(self.feature_names):
            feat_data = self.combined_features[:, i]
            if np.std(feat_data) > 1e-10:
                corr, _ = pearsonr(feat_data, self.combined_targets)
                correlations[feat_name] = abs(corr)
        
        # Sort by correlation
        sorted_features = sorted(correlations.items(), 
                               key=lambda x: x[1], reverse=True)
        
        # Get top N features
        n_features = self.params['features']['n_top_features']
        top_features = [f[0] for f in sorted_features[:n_features]]
        
        print(f"\nTop {n_features} features:")
        for i, (feat, corr) in enumerate(sorted_features[:n_features]):
            print(f"{i+1:2d}. {feat:<40} |r|={corr:.3f}")
        
        return top_features, correlations
    
    def _plot_model_results(self, results, X_scaled, y, groups):
        """Create visualization of model results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Multi-Subject Model Results', fontsize=16)
        
        # 1. Model comparison
        ax = axes[0, 0]
        model_names = list(results.keys())
        cv_means = [results[m]['cv_mean'] for m in model_names]
        cv_stds = [results[m]['cv_std'] for m in model_names]
        
        bars = ax.bar(model_names, cv_means, yerr=cv_stds, capsize=5)
        ax.set_ylabel('Mean CV Correlation')
        ax.set_title('Model Performance Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Highlight best model
        best_idx = np.argmax(cv_means)
        bars[best_idx].set_color('darkgreen')
        
        # 2. Subject-wise performance
        ax = axes[0, 1]
        best_model = self.best_model
        
        subject_correlations = {}
        unique_subjects = np.unique(groups)
        
        for subject in unique_subjects:
            subject_mask = groups == subject
            X_subj = X_scaled[subject_mask]
            y_subj = y[subject_mask]
            
            y_pred = best_model.predict(X_subj)
            corr, _ = pearsonr(y_subj, y_pred)
            subject_correlations[subject] = corr
        
        subjects = list(subject_correlations.keys())
        correlations = list(subject_correlations.values())
        
        ax.bar(range(len(subjects)), correlations)
        ax.set_xlabel('Subject')
        ax.set_ylabel('Correlation')
        ax.set_title('Per-Subject Performance (Best Model)')
        ax.set_xticks(range(len(subjects)))
        ax.set_xticklabels(subjects, rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Feature importance (for tree-based models)
        ax = axes[1, 0]
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            
            top_features = [self.selected_features[i] for i in indices]
            top_importances = importances[indices]
            
            y_pos = np.arange(len(top_features))
            ax.barh(y_pos, top_importances)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features)
            ax.set_xlabel('Importance')
            ax.set_title('Top 10 Feature Importances')
            ax.grid(True, alpha=0.3, axis='x')
        else:
            ax.text(0.5, 0.5, 'Feature importance\nnot available\nfor this model',
                   ha='center', va='center', transform=ax.transAxes)
        
        # 4. Predictions scatter plot
        ax = axes[1, 1]
        y_pred = self.best_model.predict(X_scaled)
        
        # Downsample for clarity
        downsample = 10
        ax.scatter(y[::downsample], y_pred[::downsample], 
                  alpha=0.5, s=20)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        ax.set_xlabel('Actual PDA')
        ax.set_ylabel('Predicted PDA')
        ax.set_title(f'Predictions (r={pearsonr(y, y_pred)[0]:.3f})')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_results.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def test_on_new_subject(self, subject_id, eeg_file, pda_file=None):
        """
        Test the trained model on a new subject.
        
        Parameters:
        -----------
        subject_id : str
            Subject identifier
        eeg_file : str
            Path to EEG file
        pda_file : str, optional
            Path to PDA file (for evaluation)
        
        Returns:
        --------
        dict : Test results including predictions
        """
        print(f"\n{'='*70}")
        print(f"TESTING ON NEW SUBJECT: {subject_id}")
        print(f"{'='*70}")
        
        # Process the subject
        test_data = self.process_single_subject(subject_id, eeg_file, pda_file)
        
        # Extract selected features
        feature_indices = [test_data['feature_names'].index(f) 
                          for f in self.selected_features]
        X_test = test_data['features'][:, feature_indices]
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predict
        predictions = self.best_model.predict(X_test_scaled)
        
        results = {
            'subject_id': subject_id,
            'predictions': predictions,
            'time_points': test_data['time_points']
        }
        
        # If true PDA provided, evaluate
        if pda_file is not None:
            y_true = test_data['target']
            correlation, _ = pearsonr(y_true, predictions)
            r2 = r2_score(y_true, predictions)
            rmse = np.sqrt(mean_squared_error(y_true, predictions))
            
            results.update({
                'true_pda': y_true,
                'correlation': correlation,
                'r2': r2,
                'rmse': rmse
            })
            
            print(f"\nTest Results:")
            print(f"  Correlation: {correlation:.3f}")
            print(f"  R²: {r2:.3f}")
            print(f"  RMSE: {rmse:.3f}")
            
            # Plot results
            self._plot_test_results(results)
        
        return results
    
    def _plot_test_results(self, results):
        """Plot test results for a single subject."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        time_points = results['time_points']
        predictions = results['predictions']
        true_pda = results['true_pda']
        
        # Time series
        ax = axes[0]
        ax.plot(time_points, true_pda, 'k-', label='True PDA', alpha=0.8)
        ax.plot(time_points, predictions, 'r--', label='Predicted PDA', alpha=0.8)
        ax.set_ylabel('PDA (z-score)')
        ax.set_title(f"Test Subject: {results['subject_id']} (r={results['correlation']:.3f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Residuals
        ax = axes[1]
        residuals = true_pda - predictions
        ax.plot(time_points, residuals, 'g-', alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Residuals')
        ax.set_title('Prediction Errors')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"test_{results['subject_id']}.png"),
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filename='multi_subject_pda_model.pkl'):
        """Save the trained model and necessary components."""
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'selected_features': self.selected_features,
            'model_results': self.model_results,
            'params': self.params,
            'n_subjects_trained': len(np.unique(self.subject_ids))
        }
        
        filepath = os.path.join(self.output_dir, filename)
        joblib.dump(model_data, filepath)
        print(f"\nModel saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load a previously trained model."""
        model_data = joblib.load(filepath)
        self.best_model = model_data['model']
        self.scaler = model_data['scaler']
        self.selected_features = model_data['selected_features']
        self.model_results = model_data['model_results']
        self.params = model_data['params']
        print(f"Model loaded from: {filepath}")
        print(f"Trained on {model_data['n_subjects_trained']} subjects")
