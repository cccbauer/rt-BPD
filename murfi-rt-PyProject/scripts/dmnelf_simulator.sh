#!/bin/bash
# Clemens Bauer
# Modified by Paul Bloom November 2022

# Serves nii volumes already on the computer's hard drive to MURFI (for simulating runs)

TR=${1}

singularity exec ./murfi2.sif ./servedata_custom.sh 1200 ${TR} ~/dmnelf/murfi-rt-PyProject/scripts/img/ 3

