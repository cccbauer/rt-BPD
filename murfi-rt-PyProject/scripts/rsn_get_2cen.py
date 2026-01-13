#!/usr/bin/env python
# coding: utf-8
# In[1]:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nipype.interfaces import fsl
from nipype.interfaces.fsl import MotionOutliers
import os.path
import subprocess
import os
from glob import glob
import sys
import pandas as pd
from nipype.interfaces.fsl import ImageStats

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

# Define file paths for file with correlations and IC output files
correlfile = f'{ica_directory}/template_rsn_correlations_with_ICs.txt' 
split_outfile = f'{ica_directory}/melodic_IC_'

'''
Read correlation file
3 columns: IC#, Yeo Network # (DMN=1, CEN=2), Correlation
'''
fslcc_info = pd.read_csv(correlfile, sep=' ', skipinitialspace=True, header=None)
fslcc_info.columns = ['ic_number', 'yeo_network_number', 'correlation']

# Absolute value of correlations (ICs could be negatively correlated with corresponding networks)
fslcc_info['correlation_abs'] = np.abs(fslcc_info.correlation)
fslcc_info.sort_values(by=['correlation_abs', 'yeo_network_number'], ascending=False, inplace=True)

# Correlations specifically with DMN and CEN
dmn_info = fslcc_info[fslcc_info.yeo_network_number == 1]
cen_info = fslcc_info[fslcc_info.yeo_network_number == 2]

# Select IC with strongest absolute value correlation for DMN
dmn_strongest_ic = dmn_info[dmn_info.correlation_abs == dmn_info.correlation_abs.max()].head(1)

# Select TOP 2 ICs for CEN (to capture bilateral components)
cen_top_ics = cen_info.nlargest(2, 'correlation_abs')

dmn_ic_selection = int(dmn_strongest_ic.ic_number.iloc[0]) - 1

print('DMN:', dmn_strongest_ic)
print('CEN components:')
print(cen_top_ics)
print(f'DMN: melodic_IC_{dmn_ic_selection}')

# Pull DMN IC
dmnfuncfile = split_outfile + '%0.4d.nii' % dmn_ic_selection
os.system('cp %s %s' % (dmnfuncfile, dmn_component))

# Flip DMN if needed
if float(dmn_strongest_ic.correlation.iloc[0]) < 0:
    print('Flipping IC Loadings for DMN')
    os.system(f'fslmaths {dmn_component} -mul -1 {dmn_component}')

# Process CEN components
cen_components = []
for idx, (i, row) in enumerate(cen_top_ics.iterrows()):
    cen_ic_num = int(row.ic_number) - 1
    cen_correlation = float(row.correlation)
    
    print(f'CEN component {idx+1}: melodic_IC_{cen_ic_num} (r={cen_correlation:.3f})')
    
    cenfuncfile = split_outfile + '%0.4d.nii' % cen_ic_num
    cen_component = f'{cen_component_base}_{idx+1}.nii'
    
    os.system('cp %s %s' % (cenfuncfile, cen_component))
    
    # Flip if needed
    if cen_correlation < 0:
        print(f'Flipping IC Loadings for CEN component {idx+1}')
        os.system(f'fslmaths {cen_component} -mul -1 {cen_component}')
    
    cen_components.append(cen_component)

# Combine CEN components
if len(cen_components) > 1:
    print(f'Combining {len(cen_components)} CEN components...')
    combined_cen = f'{cen_component_base}_combined.nii'
    
    # Start with first component
    os.system(f'fslmaths {cen_components[0]} {combined_cen}')
    
    # Add remaining components
    for comp in cen_components[1:]:
        os.system(f'fslmaths {combined_cen} -add {comp} {combined_cen}')
    
    print(f'Combined CEN saved to: {combined_cen}')
else:
    # Only one component, just copy it
    os.system(f'cp {cen_components[0]} {cen_component_base}_combined.nii')
