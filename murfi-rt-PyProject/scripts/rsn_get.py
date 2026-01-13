#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import os
import sys
import subprocess

subjID = sys.argv[1]
ica_version = sys.argv[2]

if ica_version == 'multi_run':
    ica_directory = f'../subjects/{subjID}/rest/rs_network.gica/groupmelodic.ica/'
    dmn_component = f'{ica_directory}/dmn_uthresh.nii'
    cen_component_base = f'{ica_directory}/cen_uthresh'
elif ica_version == 'single_run':
    ica_directory = f'../subjects/{subjID}/rest/rs_network.ica/'
    dmn_component = f'{ica_directory}/dmn_uthresh.nii'
    cen_component_base = f'{ica_directory}/cen_uthresh'

# Define file paths
correlfile = f'{ica_directory}/template_rsn_correlations_with_ICs.txt' 
split_outfile = f'{ica_directory}/melodic_IC_'

# Read correlation file
fslcc_info = pd.read_csv(correlfile, sep=' ', skipinitialspace=True, header=None)
fslcc_info.columns = ['ic_number', 'yeo_network_number', 'correlation']

# Absolute value of correlations
fslcc_info['correlation_abs'] = np.abs(fslcc_info.correlation)

# Correlations specifically with DMN and CEN
dmn_info = fslcc_info[fslcc_info.yeo_network_number == 1]
cen_info = fslcc_info[fslcc_info.yeo_network_number == 2]

# Select strongest DMN IC
dmn_strongest_ic = dmn_info.nlargest(1, 'correlation_abs')
dmn_ic_selection = int(dmn_strongest_ic.ic_number.iloc[0]) - 1

print('DMN:', dmn_strongest_ic[['ic_number', 'correlation', 'correlation_abs']])
print(f'DMN: melodic_IC_{dmn_ic_selection}')

# Pull DMN IC
dmnfuncfile = split_outfile + '%0.4d.nii' % dmn_ic_selection
os.system('cp %s %s' % (dmnfuncfile, dmn_component))

# Flip DMN if needed
if float(dmn_strongest_ic.correlation.iloc[0]) < 0:
    print('Flipping IC Loadings for DMN')
    os.system(f'fslmaths {dmn_component} -mul -1 {dmn_component}')

# Analyze lateralization of top CEN components
print('\n===== Analyzing CEN Component Lateralization =====')

cen_positive = cen_info[cen_info.correlation > 0].copy()
cen_positive = cen_positive.sort_values('correlation', ascending=False)
top_cen = cen_positive.head(5)  # Check top 5

lateralization_scores = []

for idx, (i, row) in enumerate(top_cen.iterrows()):
    ic_num = int(row.ic_number) - 1
    ic_corr = float(row.correlation)
    
    ic_file = split_outfile + '%0.4d.nii' % ic_num
    
    # Split into hemispheres and count voxels
    left_file = f'{ica_directory}/temp_ic{ic_num}_left.nii'
    right_file = f'{ica_directory}/temp_ic{ic_num}_right.nii'
    
    os.system(f'fslmaths {ic_file} -roi 0 64 0 -1 0 -1 0 -1 {left_file}')
    os.system(f'fslmaths {ic_file} -roi 64 64 0 -1 0 -1 0 -1 {right_file}')
    
    # Get intensity ranges
    left_range = subprocess.check_output(f'fslstats {left_file} -R', shell=True).decode().strip()
    right_range = subprocess.check_output(f'fslstats {right_file} -R', shell=True).decode().strip()
    
    left_max = float(left_range.split()[1])
    right_max = float(right_range.split()[1])
    
    # Calculate lateralization index: (L-R)/(L+R), ranges from -1 (right) to +1 (left)
    if (left_max + right_max) > 0:
        lat_index = (left_max - right_max) / (left_max + right_max)
    else:
        lat_index = 0
    
    lateralization_scores.append({
        'ic': ic_num,
        'ic_number': int(row.ic_number),
        'correlation': ic_corr,
        'left_max': left_max,
        'right_max': right_max,
        'lat_index': lat_index
    })
    
    print(f'IC {int(row.ic_number):2d} (r={ic_corr:.3f}): Left_max={left_max:6.2f}, Right_max={right_max:6.2f}, Lat_index={lat_index:+.3f}')
    
    # Clean up temp files
    os.system(f'rm {left_file} {right_file}')

# Find the most bilateral component (closest to 0 lateralization index)
lat_df = pd.DataFrame(lateralization_scores)
lat_df['abs_lat_index'] = lat_df['lat_index'].abs()
lat_df = lat_df.sort_values('abs_lat_index')

print('\n===== Most Bilateral CEN Components =====')
print(lat_df[['ic_number', 'correlation', 'left_max', 'right_max', 'lat_index']].head())

# Strategy: Pick the component with best balance of correlation and bilaterality
# Weight: 70% correlation, 30% bilaterality
#lat_df['score'] = lat_df['correlation'] * 0.7 + (1 - lat_df['abs_lat_index']) * 0.3 * lat_df['correlation'].max()
#Weight: 40% correlation, 60% bilaterality
lat_df['score'] = lat_df['correlation'] * 0.4 + (1 - lat_df['abs_lat_index']) * 0.6 * lat_df['correlation'].max()
best_cen = lat_df.sort_values('score', ascending=False).iloc[0]

print(f'\n===== Selected CEN Component =====')
print(f"IC {int(best_cen['ic_number'])} (r={best_cen['correlation']:.3f}, lat_index={best_cen['lat_index']:+.3f})")

# Use only the most bilateral component
cen_ic_num = int(best_cen['ic'])
cenfuncfile = split_outfile + '%0.4d.nii' % cen_ic_num
cen_component = f'{cen_component_base}_combined.nii'

os.system('cp %s %s' % (cenfuncfile, cen_component))
print(f'Using IC {int(best_cen["ic_number"])} as CEN component')
print('=' * 50)
