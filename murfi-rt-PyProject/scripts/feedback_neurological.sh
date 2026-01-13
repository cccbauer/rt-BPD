#! /bin/bash
# Clemens Bauer
# Modified by Paul Bloom December 2022

## 4 ARGS: [subid] [ses] [run] [step]

# Add this line to your script before any FSL commands
export FSLOUTPUTTYPE=NIFTI

subj=$1
step=$2
ses='ses-lo1'
run='run-01'

# Set initial paths
subj_dir=../subjects/$subj
cwd=$(pwd)
absolute_path=$(dirname $cwd)
subj_dir_absolute="${absolute_path}/subjects/$subj"
fsl_scripts=../scripts/fsl_scripts

# Set template files (use non-LPS versions)
template_dmn='FSL_7networks_DMN.nii'
template_cen='FSL_7networks_CEN.nii'
SCRIPT_PATH=$(dirname $(realpath -s $0))
template_path=${SCRIPT_PATH}/MNI152_T1_2mm_brain

# Set paths & check that computers are properly connected with scanner via Ethernet
if [ ${step} = setup ]
then
    clear
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo "+ Wellcome to MURFI real-time Neurofeedback"
    echo "+ running " ${step}
    export MURFI_SUBJECTS_DIR="${absolute_path}/subjects/"
    export MURFI_SUBJECT_NAME=$subj
    echo "+ subject ID: "$MURFI_SUBJECT_NAME
    echo "+ working dir: $MURFI_SUBJECTS_DIR"
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo "checking the presence of scanner and stim computer"
    ping -c 3 192.168.2.1
    ping -c 3 192.168.2.6
    echo "make sure Wi-Fi is off"
    echo "make sure you are Wired Connected to rt-fMRI"
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
fi  

# run MURFI for 2vol scan (to be used for registering masks to native space)  
if [ ${step} = 2vol ]
then
    clear
    echo "ready to receive 2 volume scan"
    singularity exec /home/rt/singularity-images/murfi-sif_latest.sif murfi -f $subj_dir/xml/2vol.xml
fi

if  [ ${step} = feedback ]
then
    clear
    echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo "ready to receive rtdmn feedback scan"
    export MURFI_SUBJECTS_DIR="${absolute_path}/subjects/"
    export MURFI_SUBJECT_NAME=$subj 
    singularity exec /home/rt/singularity-images/murfi-sif_latest.sif murfi -f $subj_dir_absolute/xml/rtdmn.xml
fi

if  [ ${step} = resting_state ]
then
    clear
    echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo "ready to receive resting state scan"
    export MURFI_SUBJECTS_DIR="${absolute_path}/subjects/"
    export MURFI_SUBJECT_NAME=$subj
    singularity exec /home/rt/singularity-images/murfi-sif_latest.sif murfi -f $subj_dir/xml/rest.xml
fi

if  [ ${step} = extract_rs_networks ]
then
    clear
    echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo "+ compiling resting state run into analysis folder"
    
    # Check if ICA results already exist
    if [[ -d "$subj_dir/rest/rs_network.gica" ]] || [[ -d "$subj_dir/rest/rs_network.ica" ]]; then
        existing_ica=""
        if [[ -d "$subj_dir/rest/rs_network.gica" ]]; then
            existing_ica="Multi-run ICA (rs_network.gica)"
        elif [[ -d "$subj_dir/rest/rs_network.ica" ]]; then
            existing_ica="Single-run ICA (rs_network.ica)"
        fi
        
        if ! zenity --question --title="ICA Results Already Exist" \
            --text="<span foreground='orange'><b>Warning: ICA results already exist!</b></span>\n\n${existing_ica} directory found.\n\nDo you want to overwrite and re-run ICA?\n(This will take ~25 minutes)" \
            --ok-label="Overwrite and Re-run" \
            --cancel-label="Keep Existing" \
            --width=500; then
            
            echo "Keeping existing ICA results, skipping extraction"
            return 0
        else
            echo "Removing existing ICA results and re-running..."
            rm -rf "$subj_dir/rest/rs_network.gica" 
            rm -rf "$subj_dir/rest/rs_network.ica"
        fi
    fi
    
    expected_volumes=250
    runstring="Resting state runs should have ${expected_volumes} volumes\n"
    for i in {0..10};
    do
        run_volumes=$(find ${subj_dir_absolute}/img/ -type f \( -iname "img-0000${i}*" \) | wc -l)
        Tw1_image=$(find ${subj_dir_absolute}/rest/ -type f \( -iname "*Tw1*" \) | wc -l)
        if [ ${run_volumes} -ne 0 ]
        then
            runstring="${runstring}\nRun ${i}: ${run_volumes} volumes"
            Tw1_runstring="${Tw1_runstring}\nTw1 image: ${Tw1_image}"
        fi
    done

    # use zenity to allow user to choose which resting volume to use
    input_string=$(zenity --forms --title="Which resting state runs to use for ICA?" \
        --separator=" " --width 600 --height 600 \
        --add-entry="First Input Run #" \
        --add-entry="Second Input Run #" --text="`printf "${runstring}"`"\
        --add-combo="`printf "How many resting runs to use for ICA?\nOnly use runs that have 200+ volumes for ICA"`" --combo-values "2 (default) |1 (only to be used if there aren't 2 viable runs to use)")

    if [[ $? == 1 ]];
    then
        exit 0
    fi

    read -a input_array <<< $input_string
    rest_runA_num=${input_array[0]}
    rest_runB_num=${input_array[1]}

    # Use 2 resting runs for ICA
    echo ${input_array[2]}
    if [[ ${input_array[2]} == '2' ]] ;
    then
        echo "Using run ${rest_runA_num} and run ${rest_runB_num}"
        echo "Pre-processing images before ICA..."

        rest_runA_filename=$subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold.nii'
        rest_runB_filename=$subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-02_bold.nii' 

        volsA=$(find ${subj_dir_absolute}/img/ -type f \( -iname "img-0000${rest_runA_num}*" \))
        volsB=$(find ${subj_dir_absolute}/img/ -type f \( -iname "img-0000${rest_runB_num}*" \)) 
        fslmerge -tr $rest_runA_filename $volsA 1.2
        fslmerge -tr $rest_runB_filename $volsB 1.2

        rest_runA_volumes=$(fslnvols $rest_runA_filename)
        rest_runB_volumes=$(fslnvols $rest_runB_filename)
        if [ ${rest_runA_volumes} -ne ${expected_volumes} ] || [ ${rest_runB_volumes} -ne ${expected_volumes} ]; 
        then
            echo "WARNING! ${rest_runA_volumes} volumes of resting-state data found for run 1."
            echo "${rest_runB_volumes} volumes of resting-state data found for run 2. ${expected_volumes} expected?"

            minvols=$(( rest_runA_volumes < rest_runB_volumes ? rest_runA_volumes : rest_runB_volumes ))
            echo "Clipping runs so that both have ${minvols} volumes"
            fslroi $rest_runA_filename $rest_runA_filename 0 $minvols
            fslroi $rest_runB_filename $rest_runB_filename 0 $minvols
        else
            minvols=$expected_volumes
        fi

        echo "+ computing resting state networks this will take about 25 minutes"
        echo "+ started at: $(date)"

        mcflirt -in $rest_runA_filename -out $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt.nii'
        mcflirt -in $rest_runB_filename -out $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-02_bold_mcflirt.nii'

        fslmaths $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt.nii' \
            -Tmedian $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median.nii'

        fslmaths $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-02_bold_mcflirt.nii' \
            -Tmedian $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-02_bold_mcflirt_median.nii'
                
        flirt -cost leastsq -dof 6  -noresample -noresampblur \
            -in $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-02_bold_mcflirt_median.nii' \
            -ref $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median.nii' \
            -out $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run2_median_to_run1_median.nii' \
            -omat $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run2_median_to_run1_median.mat' 

        slices $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median.nii' \
            $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run2_median_to_run1_median.nii' \
            -o $subj_dir_absolute/qc/flirt_median_rest_check.gif 

        flirt -noresample -noresampblur -interp nearestneighbour \
            -in  $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-02_bold_mcflirt.nii' \
            -ref $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median.nii'  \
            -out $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-02_bold_mcflirt_run1space.nii' \
            -init $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run2_median_to_run1_median.mat' \
            -applyxfm

        bet $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median.nii' \
            $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median_bet.nii' \
            -R -f 0.4 -g 0 -m 

        slices $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median.nii' \
            $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median_bet.nii' \
            -o $subj_dir_absolute/qc/rest_skullstrip_check_run1.gif

        fslmaths $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt.nii' \
            -mas $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median_bet_mask.nii' \
            $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_masked.nii'

        fslmaths $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-02_bold_mcflirt_run1space.nii' \
            -mas $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median_bet_mask.nii' \
            $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-02_bold_mcflirt_run1space_masked.nii'

        ica_run1_input=$subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_masked.nii'
        ica_run2_input=$subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-02_bold_mcflirt_run1space_masked.nii'
        reference_vol_for_ica=$subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median_bet.nii'

        cp $fsl_scripts/basic_ica_template.fsf $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_'$run'_bold'.fsf
        OUTPUT_dir=$subj_dir_absolute/rest/rs_network
        sed -i "s#DATA1#$ica_run1_input#g" $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_'$run'_bold'.fsf
        sed -i "s#DATA2#$ica_run2_input#g" $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_'$run'_bold'.fsf
        sed -i "s#OUTPUT#$OUTPUT_dir#g" $subj_dir/rest/$subj'_'$ses'_task-rest_'$run'_bold'.fsf
        sed -i "s#REFERENCE_VOL#$reference_vol_for_ica#g" $subj_dir/rest/$subj'_'$ses'_task-rest_'$run'_bold'.fsf 

        sed -i "s/set fmri(npts) 250/set fmri(npts) ${minvols}/g" $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_'$run'_bold'.fsf
        feat $subj_dir/rest/$subj'_'$ses'_task-rest_'$run'_bold'.fsf        
    else
        # Single run ICA
        echo "Using run ${rest_runA_num} for single-run ICA"

        rest_runA_filename=$subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold'.nii
        volsA=$(find ${subj_dir_absolute}/img/ -type f \( -iname "img-0000${rest_runA_num}*" \))
        fslmerge -tr $rest_runA_filename $volsA 1.2

        rest_runA_volumes=$(fslnvols $rest_runA_filename)
        echo "${rest_runA_volumes} volumes of resting-state data found for run 1."
        echo "+ computing resting state networks this will take about 25 minutes"
        echo "+ started at: $(date)"

        mcflirt -in $rest_runA_filename -out $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt.nii'

        fslmaths $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt.nii' \
            -Tmedian $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median.nii'
     
        bet $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median.nii' \
            $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median_bet.nii' \
            -R -f 0.4 -g 0 -m 

        slices $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median.nii' \
            $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median_bet.nii' \
            -o $subj_dir_absolute/qc/rest_skullstrip_check_run1.gif

        fslmaths $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt.nii' \
            -mas $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median_bet_mask.nii' \
            $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_masked.nii'
        
        ica_run1_input=$subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_masked.nii'
        reference_vol_for_ica=$subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median_bet.nii'

        cp fsl_scripts/basic_ica_template_single_run.fsf $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_'$run'_bold'.fsf
        OUTPUT_dir=$subj_dir_absolute/rest/rs_network
        sed -i "s#DATA#$ica_run1_input#g" $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_'$run'_bold'.fsf
        sed -i "s#OUTPUT#$OUTPUT_dir#g" $subj_dir/rest/$subj'_'$ses'_task-rest_'$run'_bold'.fsf
        sed -i "s#REFERENCE_VOL#$reference_vol_for_ica#g" $subj_dir/rest/$subj'_'$ses'_task-rest_'$run'_bold'.fsf 

        sed -i "s/set fmri(npts) 250/set fmri(npts) ${rest_runA_volumes}/g" $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_'$run'_bold'.fsf
        feat $subj_dir/rest/$subj'_'$ses'_task-rest_'$run'_bold'.fsf   
    fi                        
fi

if [ ${step} = process_roi_masks ]; then

    clear
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo "+ Generating DMN & CEN Masks "
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

    # Set up ICA directory (multi-run or single-run)
    ica_directory=$subj_dir/rest/rs_network.gica/groupmelodic.ica/

    if [ -d $ica_directory ]; then
        ica_version='multi_run'
    elif [ -d "${subj_dir}/rest/rs_network.ica/filtered_func_data.ica/" ]; then
        ica_directory="${subj_dir}/rest/rs_network.ica/" 
        ica_version='single_run'
    else
        echo "Error: no ICA directory found for ${subj}. Exiting now..."
        exit 0
    fi
    echo "+ ICA is: ${ica_version}"

    # Make output file for correlations
    correlfile=$ica_directory/template_rsn_correlations_with_ICs.txt
    touch ${correlfile}
    template_networks='template_networks.nii'

    # Use study_ref as the native reference (NEUROLOGICAL orientation)
    examplefunc=${subj_dir}/xfm/study_ref.nii
    examplefunc_mask=$subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median_bet_mask.nii'

    # Handle single-run vs multi-run ICA
    if [ $ica_version == 'single_run' ]; then
        infile=$ica_directory/filtered_func_data.ica/melodic_IC.nii
    else
        infile=$ica_directory/melodic_IC.nii
        mkdir -p ${ica_directory}/reg

        # Calculate transformation: study_ref <-> MNI
        example_func2mni=${ica_directory}/reg/example_func2mni
        example_func2mni_mat=${ica_directory}/reg/example_func2mni.mat
        mni2example_func=${ica_directory}/reg/mni2example_func.nii
        mni2example_func_mat=${ica_directory}/reg/mni2example_func.mat

        # Register study_ref to MNI and get transformation matrix
        flirt -in ${examplefunc} -ref MNI152_T1_2mm_brain -out ${example_func2mni} -omat ${example_func2mni_mat}
        
        # Invert to get MNI -> study_ref transformation
        convert_xfm -omat ${mni2example_func_mat} -inverse ${example_func2mni_mat}

        # Transform melodic ICs from MNI space to study_ref space
        echo "+ Converting melodic_IC from MNI to study_ref space"
        flirt -in $infile -ref ${examplefunc} -out $ica_directory/melodic_IC_studyref.nii -init ${mni2example_func_mat} -applyxfm -interp trilinear
        infile=$ica_directory/melodic_IC_studyref.nii
    fi

    # Define mask paths
    dmn_uthresh=$ica_directory/dmn_uthresh.nii

    # Register template networks to study_ref space
    template2example_func=${ica_directory}/reg/template_networks_studyref.nii
    dmn2example_func=${ica_directory}/reg/template_dmn_studyref.nii
    cen2example_func=${ica_directory}/reg/template_cen_studyref.nii

    flirt -in MNI152_T1_2mm_brain -ref ${examplefunc} -out ${mni2example_func} -init ${mni2example_func_mat} -applyxfm
    flirt -in ${template_networks} -ref ${examplefunc} -out ${template2example_func} -init ${mni2example_func_mat} -applyxfm
    flirt -in ${template_dmn} -ref ${examplefunc} -out ${dmn2example_func} -init ${mni2example_func_mat} -applyxfm
    flirt -in ${template_cen} -ref ${examplefunc} -out ${cen2example_func} -init ${mni2example_func_mat} -applyxfm

    # Correlate ICs with templates
    rm -f ${correlfile}
    fslcc --noabs -p 8 -t -1 -m ${examplefunc_mask} ${infile} ${template2example_func} >> ${correlfile}

    # Split ICs
    split_outfile=$ica_directory/melodic_IC_
    fslsplit ${infile} ${split_outfile}

    # Select ICs
    python rsn_get.py ${subj} ${ica_version}

    # Use the COMBINED CEN from rsn_get.py
	cen_uthresh=$ica_directory/cen_uthresh_combined.nii

	## Thresholded masks
	dmn_thresh=$ica_directory/dmn_thresh.nii
	cen_thresh=$ica_directory/cen_thresh.nii

	# Hard code the number of voxels desired for each mask
	num_voxels_desired=2000

	# DMN processing (unchanged)
	fslmaths ${dmn_uthresh} -mul ${dmn2example_func} ${dmn_uthresh}
	voxels_in_dmn=$(fslstats ${dmn_uthresh} -V | awk '{print $1}')
	percentile_dmn=$(python -c "print(100*(1-${num_voxels_desired}/${voxels_in_dmn}))")
	dmn_thresh_value=$(fslstats ${dmn_uthresh} -P ${percentile_dmn})
	fslmaths ${dmn_uthresh} -thr ${dmn_thresh_value} -bin ${dmn_thresh} -odt short

	# CEN processing - ADAPTIVE BILATERAL THRESHOLDING
	echo ""
	echo "===== CEN Adaptive Bilateral Thresholding ====="

	# Multiply by template
	fslmaths ${cen_uthresh} -mul ${cen2example_func} ${cen_uthresh}

	# Split into hemispheres
	cen_left=${ica_directory}/cen_uthresh_left.nii
	cen_right=${ica_directory}/cen_uthresh_right.nii
	fslmaths ${cen_uthresh} -roi 0 64 0 -1 0 -1 0 -1 ${cen_left}
	fslmaths ${cen_uthresh} -roi 64 64 0 -1 0 -1 0 -1 ${cen_right}

	# Get the number of non-zero voxels in each hemisphere
	voxels_in_left=$(fslstats ${cen_left} -V | awk '{print $1}')
	voxels_in_right=$(fslstats ${cen_right} -V | awk '{print $1}')

	echo "Non-zero voxels - Left: ${voxels_in_left}, Right: ${voxels_in_right}"

	# Calculate how many voxels to keep from each hemisphere proportionally
	total_voxels=$((voxels_in_left + voxels_in_right))
	voxels_left_target=$(python -c "print(int(${num_voxels_desired} * ${voxels_in_left} / ${total_voxels}))")
	voxels_right_target=$(python -c "print(int(${num_voxels_desired} * ${voxels_in_right} / ${total_voxels}))")

	# But ensure at least 30% from each hemisphere (for true bilateral coverage)
	min_voxels_per_hem=$(python -c "print(int(${num_voxels_desired} * 0.3))")
	if [ ${voxels_left_target} -lt ${min_voxels_per_hem} ]; then
	    voxels_left_target=${min_voxels_per_hem}
	fi
	if [ ${voxels_right_target} -lt ${min_voxels_per_hem} ]; then
	    voxels_right_target=${min_voxels_per_hem}
	fi

	echo "Target voxels - Left: ${voxels_left_target}, Right: ${voxels_right_target}"

	# Threshold each hemisphere to keep its target number of voxels
	percentile_left=$(python -c "print(100*(1-${voxels_left_target}/${voxels_in_left}))")
	left_thresh_value=$(fslstats ${cen_left} -P ${percentile_left})
	fslmaths ${cen_left} -thr ${left_thresh_value} -bin ${ica_directory}/cen_thresh_left.nii -odt short

	percentile_right=$(python -c "print(100*(1-${voxels_right_target}/${voxels_in_right}))")
	right_thresh_value=$(fslstats ${cen_right} -P ${percentile_right})
	fslmaths ${cen_right} -thr ${right_thresh_value} -bin ${ica_directory}/cen_thresh_right.nii -odt short

	# Combine hemispheres
	fslmaths ${ica_directory}/cen_thresh_left.nii -add ${ica_directory}/cen_thresh_right.nii ${cen_thresh} -odt short

	# Verify final counts
	final_left=$(fslstats ${ica_directory}/cen_thresh_left.nii -V | awk '{print $1}')
	final_right=$(fslstats ${ica_directory}/cen_thresh_right.nii -V | awk '{print $1}')
	final_total=$(fslstats ${cen_thresh} -V | awk '{print $1}')

	echo "Final CEN mask - Left: ${final_left}, Right: ${final_right}, Total: ${final_total} voxels"
	echo "======================================="

	echo "Number of voxels in dmn mask: $(fslstats ${dmn_thresh} -V | head -c 5)"
	echo "Number of voxels in cen mask: $(fslstats ${cen_thresh} -V | head -c 5)"

    cp ${dmn_thresh} ${subj_dir}/mask/dmn_native_rest.nii
    cp ${cen_thresh} ${subj_dir}/mask/cen_native_rest.nii

    # Display masks
    fsleyes $examplefunc ${mni2example_func} ${dmn_thresh} -cm blue ${cen_thresh} -cm red

fi

# For registering masks in resting state space to 2vol space
if [ ${step} = register ]
then
    clear
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo "+ Registering masks to study_ref"
    
    # Version sort handles series numbers naturally
    latest_ref=$(ls -v $subj_dir/xfm/series[0-9]*_ref.nii 2>/dev/null | grep -E "series[0-9]+_ref\.nii$" | tail -n1)
    latest_ref="${latest_ref::-4}"
    
    if [ -z "$latest_ref" ]; then
        echo "ERROR: No series reference file found!"
        exit 1
    fi
    
    study_ref=${subj_dir}/xfm/study_ref.nii

    if [ ! -f ${subj_dir}/xfm/localizer_ref.nii ]
    then
        mv ${study_ref} ${subj_dir}/xfm/localizer_ref.nii
    fi
    echo "+ Registering masks to reference image from most recent series: ${latest_ref}"
    echo "+ study_ref.nii is now ${latest_ref}"

    cp ${latest_ref}.nii ${study_ref}

    bet ${latest_ref} ${latest_ref}_brain -R -f 0.4 -g 0 -m
    slices ${latest_ref} ${latest_ref}_brain_mask -o $subj_dir/qc/2vol_skullstrip_brain_mask_check.gif

    rm -rf $subj_dir/xfm/epi2reg
    mkdir -p $subj_dir/xfm/epi2reg

    ica_directory=$subj_dir/rest/rs_network.gica/groupmelodic.ica/
    reg_dir=${ica_directory}/reg

    if [ -d $ica_directory ]
    then
        ica_version='multi_run'
    elif [ -d "${subj_dir}/rest/rs_network.ica/filtered_func_data.ica/" ] 
    then
        ica_directory="${subj_dir}/rest/rs_network.ica/" 
        ica_version='single_run'
    else
        echo "Error: no ICA directory found for ${subj}. Exiting now..."
        exit 0
    fi
    
    # Note: Masks are already in study_ref space from process_roi_masks step
    # If study_ref changed (new 2vol scan), we need to register from old study_ref to new study_ref
    
    # For now, assuming masks just need to be copied if study_ref hasn't changed
    # If it has changed, add registration here
    
    echo "+ Masks are already in study_ref space"
    echo "+ If study_ref has changed, manual registration may be needed"
    
    # Display final masks
    echo "+ INSPECT"
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    xdg-open $subj_dir/qc/2vol_skullstrip_brain_mask_check.gif
    fsleyes ${latest_ref}_brain $subj_dir/mask/cen_native_rest.nii -cm red $subj_dir/mask/dmn_native_rest.nii -cm blue
fi

if [ ${step} = cleanup ]
then
    input_string=$(zenity --forms --title="Delete files?" \
    --separator=" " \
    --text="`printf "Are you sure you want to clean up the directory and delete files for ${subj}?\nThe entire img folder will be deleted, as well as raw bold data from the rest directory"`" \
    --cancel-label "Exit" --ok-label "Delete files")
    ret=$?

    if [[ $ret == 1 ]];
    then
        exit 0
    fi

    rm -rf $subj_dir/img
    rm -f $subj_dir/rest/*bold.nii
    rm -f $subj_dir/rest/*bold_mcflirt.nii
    rm -f $subj_dir/rest/*bold_mcflirt_masked.nii
fi

# Backup: register template masks to 2vol space
if [ ${step} = backup_reg_mni_masks_to_2vol ]
then
    clear
    mni_template=MNI152_T1_2mm_brain.nii
    dmn_mni=FSL_7networks_DMN.nii
    cen_mni=FSL_7networks_CEN.nii

    two_vol_ref=$(ls -t $subj_dir/xfm/series*.nii | head -n1)
    two_vol_ref="${two_vol_ref::-4}"
    
    two_vol_ref_bet=${subj_dir}/xfm/two_vol_ref_bet.nii
    two_vol_ref2mni=${subj_dir}/xfm/two_vol_ref2mni.nii
    two_vol_ref2mni_mat=${subj_dir}/xfm/two_vol_ref2mni.mat
    mni2_two_vol_ref_mat=${subj_dir}/xfm/mni2_two_vol_ref.mat

    study_ref=${subj_dir}/xfm/study_ref.nii
    if [ ! -f ${subj_dir}/xfm/localizer_ref.nii ]
    then
        mv ${study_ref} ${subj_dir}/xfm/localizer_ref.nii
    fi
    echo "Registering masks to reference image from most recent series: ${two_vol_ref}"
    echo "study_ref.nii is now ${two_vol_ref}"

    cp ${two_vol_ref}.nii ${study_ref}

    bet ${two_vol_ref} ${two_vol_ref_bet} -R -f 0.4 -g 0 -m

    flirt -in ${two_vol_ref_bet} -ref ${mni_template} -out ${two_vol_ref2mni} -omat ${two_vol_ref2mni_mat}
    convert_xfm -omat ${mni2_two_vol_ref_mat} -inverse ${two_vol_ref2mni_mat}

    flirt -in ${dmn_mni} -ref ${two_vol_ref_bet} -out $subj_dir/mask/dmn.nii -init ${mni2_two_vol_ref_mat} -applyxfm -interp nearestneighbour -datatype short
    flirt -in ${cen_mni} -ref ${two_vol_ref_bet} -out $subj_dir/mask/cen.nii -init ${mni2_two_vol_ref_mat} -applyxfm -interp nearestneighbour -datatype short

fi
