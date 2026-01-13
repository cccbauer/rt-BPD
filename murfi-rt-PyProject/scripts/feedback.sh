#! /bin/bash
# Clemens Bauer
# Modified by Paul Bloom December 2022


## 4 ARGS: [subid] [ses] [run] [step]

#Step 1: disable wireless internet, set MURFI_SUBJECTS_DIR, and NAMEcd
#Step 2: receive 2 volume scan
#Step 3: create masks
#Step 4: run murfi for realtime

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



# Set template files (NEUROLOGICAL, not LPS)
template_dmn='DMNax_brainmaskero2.nii'
template_cen='CENa_brainmaskero2.nii'
SCRIPT_PATH=$(dirname $(realpath -s $0))
template_lps_path=${SCRIPT_PATH}/MNI152_T1_2mm_LPS_brain
#echo $template_lps_path

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
    #echo "disabling wireless internet"
    #ifdown wlan0
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
            rm -rf "$subj_dir/rest/rs_network.gica" "$subj_dir/rest/rs_network.ica"
        fi
    fi
    expected_volumes=250
    runstring="Resting state runs should have ${expected_volumes} volumes\n"
    for i in {0..10};
    do
        # Find # of volumes in each run
        #ls $subj_dir/img/
        #echo run_volumes=$(find ${subj_dir_absolute}/img/ -type f \( -iname "img-0000${i}*" \) | wc -l)
        run_volumes=$(find ${subj_dir_absolute}/img/ -type f \( -iname "img-0000${i}*" \) | wc -l)
	Tw1_image=$(find ${subj_dir_absolute}/rest/ -type f \( -iname "*Tw1*" \) | wc -l)
	if [ ${run_volumes} -ne 0 ]
        then
            runstring="${runstring}\nRun ${i}: ${run_volumes} volumes"
	    Tw1_runstring="${Tw1_runstring}\nTw1 image: ${Tw1_image}"
        fi
    done

    # use zenity to allow user to choose which resting volume to use (and how many runs to use)
    input_string=$(zenity --forms --title="Which resting state runs to use for ICA?" \
        --separator=" " --width 600 --height 600 \
        --add-entry="First Input Run #" \
        --add-entry="Second Input Run #" --text="`printf "${runstring}"`"\
        --add-combo="`printf "How many resting runs to use for ICA?\nOnly use runs that have 200+ volumes for ICA"`" --combo-values "2 (default) |1 (only to be used if there aren't 2 viable runs to use)")

    # check that exit button hasn't been clicked
    if [[ $? == 1 ]];
    then
        exit 0
    fi

    # parse zenity output using space as delimiter
    read -a input_array <<< $input_string
    rest_runA_num=${input_array[0]}
    rest_runB_num=${input_array[1]}

    # Use 2 resting runs for ICA
    echo ${input_array[2]}
    if [[ ${input_array[2]} == '2' ]] ;
    then
        echo "Using run ${rest_runA_num} and run ${rest_runB_num}"
        echo "Pre-processing images before ICA..."

        # merge individual volumes to make 1 file for each resting state run
        rest_runA_filename=$subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold.nii'
        rest_runB_filename=$subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-02_bold.nii' 

        # Merge all volumes in each run 
        volsA=$(find ${subj_dir_absolute}/img/ -type f \( -iname "img-0000${rest_runA_num}*" \))
        volsB=$(find ${subj_dir_absolute}/img/ -type f \( -iname "img-0000${rest_runB_num}*" \)) 
        fslmerge -tr $rest_runA_filename $volsA 1.2
        fslmerge -tr $rest_runB_filename $volsB 1.2

        # figure out how many volumes of resting state data there were to be used in ICA
        rest_runA_volumes=$(fslnvols $rest_runA_filename)
        rest_runB_volumes=$(fslnvols $rest_runB_filename)
        if [ ${rest_runA_volumes} -ne ${expected_volumes} ] || [ ${rest_runB_volumes} -ne ${expected_volumes} ]; 
        then
            echo "WARNING! ${rest_runA_volumes} volumes of resting-state data found for run 1."
            echo "${rest_runB_volumes} volumes of resting-state data found for run 2. ${expected_volumes} expected?"

            # calculate minimum volumes (which run has fewer, then use fslroi to cut both runs to this minimum)
            minvols=$(( rest_runA_volumes < rest_runB_volumes ? rest_runA_volumes : rest_runB_volumes ))
            echo "Clipping runs so that both have ${minvols} volumes"
            fslroi $rest_runA_filename $rest_runA_filename 0 $minvols
            fslroi $rest_runB_filename $rest_runB_filename 0 $minvols
        else
            minvols=$expected_volumes
        fi

        echo "+ computing resting state networks this will take about 25 minutes"
        echo "+ started at: $(date)"

        # realign volumes pre-FEAT
        mcflirt -in $rest_runA_filename -out $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt.nii'
        mcflirt -in $rest_runB_filename -out $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-02_bold_mcflirt.nii'

        # get median volume of 1st run
        fslmaths $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt.nii' \
            -Tmedian $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median.nii'

        # get median volume of 2st run
        fslmaths $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-02_bold_mcflirt.nii' \
            -Tmedian $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-02_bold_mcflirt_median.nii'
                
        # calculate registration matrix of median of 2nd run to median of 1st run
        flirt -cost leastsq -dof 6  -noresample -noresampblur \
            -in $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-02_bold_mcflirt_median.nii' \
            -ref $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median.nii' \
            -out $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run2_median_to_run1_median.nii' \
            -omat $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run2_median_to_run1_median.mat' 


        # check registration of median of 2nd run to median of 1st run
        slices $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median.nii' \
            $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run2_median_to_run1_median.nii' \
            -o $subj_dir_absolute/qc/flirt_median_rest_check.gif 

        # use registration matrix previously calculated to warp entire 2nd run to median of 1st run
        flirt -noresample -noresampblur -interp nearestneighbour \
            -in  $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-02_bold_mcflirt.nii' \
            -ref $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median.nii'  \
            -out $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-02_bold_mcflirt_run1space.nii' \
            -init $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run2_median_to_run1_median.mat' \
            -applyxfm

        # skullstrip median of 1st run & check the generated mask
        bet $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median.nii' \
            $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median_bet.nii' \
            -R -f 0.4 -g 0 -m 

        slices $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median.nii' \
            $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median_bet.nii' \
            -o $subj_dir_absolute/qc/rest_skullstrip_check_run1.gif

        # mask both runs by the mask from skullstriped median of 1st run
        fslmaths $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt.nii' \
            -mas $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median_bet_mask.nii' \
            $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_masked.nii'

        fslmaths $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-02_bold_mcflirt_run1space.nii' \
            -mas $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median_bet_mask.nii' \
            $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-02_bold_mcflirt_run1space_masked.nii'


        
        ica_run1_input=$subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_masked.nii'
        ica_run2_input=$subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-02_bold_mcflirt_run1space_masked.nii'
        reference_vol_for_ica=$subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median_bet.nii'
	#reference_vol_for_ica=$template_lps_path

        # update FEAT template with paths and # of volumes of resting state run
        cp $fsl_scripts/basic_ica_template.fsf $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_'$run'_bold'.fsf
        OUTPUT_dir=$subj_dir_absolute/rest/rs_network
        sed -i "s#DATA1#$ica_run1_input#g" $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_'$run'_bold'.fsf
        sed -i "s#DATA2#$ica_run2_input#g" $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_'$run'_bold'.fsf
        sed -i "s#OUTPUT#$OUTPUT_dir#g" $subj_dir/rest/$subj'_'$ses'_task-rest_'$run'_bold'.fsf
        sed -i "s#REFERENCE_VOL#$reference_vol_for_ica#g" $subj_dir/rest/$subj'_'$ses'_task-rest_'$run'_bold'.fsf 

        # update fsf to match number of rest volumes
        sed -i "s/set fmri(npts) 250/set fmri(npts) ${minvols}/g" $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_'$run'_bold'.fsf
        feat $subj_dir/rest/$subj'_'$ses'_task-rest_'$run'_bold'.fsf        
    else
        # Use just a single run for ICA (only to be used when 2 isn't viable)
        echo "Using run ${rest_runA_num} for single-run ICA"

        # merge individual volumes to make 1 file for each resting state run
        rest_runA_filename=$subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold'.nii
        volsA=$(find ${subj_dir_absolute}/img/ -type f \( -iname "img-0000${rest_runA_num}*" \))
        fslmerge -tr $rest_runA_filename $volsA 1.2

        # figure out how many volumes of resting state data there were to be used in ICA
        rest_runA_volumes=$(fslnvols $rest_runA_filename)
        echo "${rest_runA_volumes} volumes of resting-state data found for run 1."
        echo "+ computing resting state networks this will take about 25 minutes"
        echo "+ started at: $(date)"

         # realign volumes pre-FEAT
        mcflirt -in $rest_runA_filename -out $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt.nii'

        # get median volume of 1st run
        fslmaths $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt.nii' \
            -Tmedian $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median.nii'
     
        # skullstrip median of 1st run & check the generated mask
        bet $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median.nii' \
            $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median_bet.nii' \
            -R -f 0.4 -g 0 -m 

        slices $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median.nii' \
            $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median_bet.nii' \
            -o $subj_dir_absolute/qc/rest_skullstrip_check_run1.gif

        # mask run1 by the mask from skullstriped median of 1st run
        fslmaths $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt.nii' \
            -mas $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median_bet_mask.nii' \
            $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_masked.nii'
        
        ica_run1_input=$subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_masked.nii'
        reference_vol_for_ica=$subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median_bet.nii'

        # update FEAT template with paths and # of volumes of resting state run
        cp fsl_scripts/basic_ica_template_single_run.fsf $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_'$run'_bold'.fsf
        OUTPUT_dir=$subj_dir_absolute/rest/rs_network
        sed -i "s#DATA#$ica_run1_input#g" $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_'$run'_bold'.fsf
        sed -i "s#OUTPUT#$OUTPUT_dir#g" $subj_dir/rest/$subj'_'$ses'_task-rest_'$run'_bold'.fsf
        sed -i "s#REFERENCE_VOL#$reference_vol_for_ica#g" $subj_dir/rest/$subj'_'$ses'_task-rest_'$run'_bold'.fsf 

        # update fsf to match number of rest volumes
        sed -i "s/set fmri(npts) 250/set fmri(npts) ${rest_runA_volumes}/g" $subj_dir_absolute/rest/$subj'_'$ses'_task-rest_'$run'_bold'.fsf
        feat $subj_dir/rest/$subj'_'$ses'_task-rest_'$run'_bold'.fsf   
    fi                        
fi

##Complete Processing Pipeline:
#Current State:

#Melodic ICs: 128×128×74, 2mm iso, NEUROLOGICAL (Group ICA space)
#Templates: 91×109×91, 2mm iso, NEUROLOGICAL (MNI space)
#MNI Brain: 91×109×91, 2mm iso, RADIOLOGICAL (MNI space)
#Median_bet: 128×128×68, 2×2×2.2mm, NEUROLOGICAL (Native functional)
#Study_ref: 128×128×68, 2×2×2.2mm, NEUROLOGICAL (MURFI native - same as median_bet!)


if [ ${step} = process_roi_masks ]
then
    clear
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo "+ Generating DMN & CEN Masks "
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    

    # Set up file paths needed for mask creation
    # first look for ICA feat directory based on multiple runs (.gica directory)
    ica_directory=$subj_dir/rest/rs_network.gica/groupmelodic.ica/

    if [ -d $ica_directory ]
    then
        ica_version='multi_run'
   
    # if ICA feat dir for multi-run ICA isn't present, look for single-run version
    elif [ -d "${subj_dir}/rest/rs_network.ica/filtered_func_data.ica/" ] 
    then
        ica_directory="${subj_dir}/rest/rs_network.ica/" 
        ica_version='single_run'
    else
        echo "Error: no ICA directory found for ${subj}. Exiting now..."
        exit 0
    fi
    echo "+ ICA is: ${ica_version}"


    # Make output file to store correlations with template networks
    correlfile=$ica_directory/template_rsn_correlations_with_ICs.txt
    touch ${correlfile}
    template_networks='template_networks.nii'

    # If single-session, then ICA was done in native space
    if [ $ica_version == 'single_run' ]
    then
        # ICs in native space
        infile=$ica_directory/filtered_func_data.ica/melodic_IC.nii
    
        # Set reference volume for single run
        examplefunc=$subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median_bet.nii'
        examplefunc_mask=$subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median_bet_mask.nii'
        
        # For single-run, resample templates to native space
        mkdir -p ${ica_directory}/reg
        
        # Calculate registration from MNI to native space
        mni2native_mat=${ica_directory}/reg/mni2native.mat
        mni2example_func=${ica_directory}/reg/mni2native.nii
        flirt -in MNI152_T1_2mm_brain -ref ${examplefunc} -out ${mni2example_func} -omat ${mni2native_mat}
        
        # Resample template networks to match native space dimensions
        template2example_func=${ica_directory}/reg/template_networks_native.nii
        dmn2example_func=${ica_directory}/reg/template_dmn_native.nii
        cen2example_func=${ica_directory}/reg/template_cen_native.nii
        
        flirt -in ${template_networks} -ref ${examplefunc} -out ${template2example_func} -init ${mni2native_mat} -applyxfm
        flirt -in ${template_dmn} -ref ${examplefunc} -out ${dmn2example_func} -init ${mni2native_mat} -applyxfm
        flirt -in ${template_cen} -ref ${examplefunc} -out ${cen2example_func} -init ${mni2native_mat} -applyxfm
    
    else # Multi run
        # ICs in group template space (median of run1)
        infile=$ica_directory/melodic_IC.nii
        mkdir -p ${ica_directory}/reg

        # Set reference
        examplefunc=$subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median_bet.nii'
        examplefunc_mask=$subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median_bet_mask.nii'

        # Resample melodic ICs to match examplefunc using applywarp (clean resampling, no transformation)
        echo "+ Resampling melodic_IC to match examplefunc dimensions (74 slices → 68 slices)"
        applywarp --in=$infile --ref=$examplefunc --out=$ica_directory/melodic_IC_examplefunc.nii --interp=trilinear
        
        infile=$ica_directory/melodic_IC_examplefunc.nii

        # Register templates from MNI to native space
        example_func2mni=${ica_directory}/reg/example_func2mni
        example_func2mni_mat=${ica_directory}/reg/example_func2mni.mat
        mni2example_func=${ica_directory}/reg/mni2example_func.nii
        mni2example_func_mat=${ica_directory}/reg/mni2example_func.mat

        flirt -in ${examplefunc} -ref MNI152_T1_2mm_brain -out ${example_func2mni} -omat ${example_func2mni_mat} -dof 12 -cost corratio
        convert_xfm -omat ${mni2example_func_mat} -inverse ${example_func2mni_mat}
        
        ## Paths for registered template files
        template2example_func=${ica_directory}/reg/template_networks2example_func.nii
        dmn2example_func=${ica_directory}/reg/template_dmn2example_func.nii
        cen2example_func=${ica_directory}/reg/template_cen2example_func.nii

        # Register template networks from MNI to native space
        flirt -in MNI152_T1_2mm_brain -ref ${examplefunc} -out ${mni2example_func} -init ${mni2example_func_mat} -applyxfm
        flirt -in ${template_networks} -ref ${examplefunc} -out ${template2example_func} -init ${mni2example_func_mat} -applyxfm
        flirt -in ${template_dmn} -ref ${examplefunc} -out ${dmn2example_func} -init ${mni2example_func_mat} -applyxfm
        flirt -in ${template_cen} -ref ${examplefunc} -out ${cen2example_func} -init ${mni2example_func_mat} -applyxfm
    fi

    # Set paths for unthresholded masks (will be in native space)
    dmn_uthresh=$ica_directory/dmn_uthresh.nii

    # Correlate ICA components with templates (both now in native space)
    rm -f ${correlfile}
    fslcc --noabs -p 8 -t -1 -m ${examplefunc_mask} ${infile} ${template2example_func}>>${correlfile}

    # Split ICs to separate files
    split_outfile=$ica_directory/melodic_IC_
    fslsplit ${infile} ${split_outfile}

    # Selection of ICs most highly correlated with template networks
    # Uses lateralization analysis to select most bilateral component
    python rsn_get.py ${subj} ${ica_version}

    # Use the selected CEN from rsn_get.py
    cen_uthresh=$ica_directory/cen_uthresh_combined.nii

    ## Thresholded masks
    dmn_thresh=$ica_directory/dmn_thresh.nii
    cen_thresh=$ica_directory/cen_thresh.nii

    # Hard code the number of voxels desired for each mask
    num_voxels_desired=2000

    # Zero out voxels not included in the template masks (multiply by template)
    fslmaths ${dmn_uthresh} -mul ${dmn2example_func} ${dmn_uthresh}
    fslmaths ${cen_uthresh} -mul ${cen2example_func} ${cen_uthresh}

    # Get number of non-zero voxels in masks, calculate percentile cutoff needed
    voxels_in_dmn=$(fslstats ${dmn_uthresh} -V | awk '{print $1}')
    percentile_dmn=$(python -c "print(100*(1-${num_voxels_desired}/${voxels_in_dmn}))")
    voxels_in_cen=$(fslstats ${cen_uthresh} -V | awk '{print $1}')
    percentile_cen=$(python -c "print(100*(1-${num_voxels_desired}/${voxels_in_cen}))")

    # Get threshold based on percentile
    dmn_thresh_value=$(fslstats ${dmn_uthresh} -P ${percentile_dmn})
    cen_thresh_value=$(fslstats ${cen_uthresh} -P ${percentile_cen})

    # Threshold masks 
    fslmaths ${dmn_uthresh} -thr ${dmn_thresh_value} -bin ${dmn_thresh} -odt short
    fslmaths ${cen_uthresh} -thr ${cen_thresh_value} -bin ${cen_thresh} -odt short

    echo "Number of voxels in dmn mask: $(fslstats ${dmn_thresh} -V | head -c 5)"
    echo "Number of voxels in cen mask: $(fslstats ${cen_thresh} -V | head -c 5)"

    # Copy masks to participant's mask directory (in native/study_ref space for MURFI)
    cp ${dmn_thresh} ${subj_dir}/mask/dmn_native_rest.nii
    cp ${cen_thresh} ${subj_dir}/mask/cen_native_rest.nii
    
    # At the end of process_roi_masks, after creating masks:
    cp ${dmn_thresh} ${subj_dir}/mask/dmn_rest_original.nii
    cp ${cen_thresh} ${subj_dir}/mask/cen_rest_original.nii

    # OPTIONAL: Save masks in MNI space for archival/comparison
    if [ $ica_version == 'multi_run' ]; then
        mkdir -p ${subj_dir}/mask/mni
        
        # Transform masks back to standard MNI space
        echo "+ Saving masks in MNI space"
        flirt -in ${dmn_thresh} -ref MNI152_T1_2mm_brain -out ${subj_dir}/mask/mni/dmn_mni.nii -init ${example_func2mni_mat} -applyxfm -interp nearestneighbour -datatype short
        flirt -in ${cen_thresh} -ref MNI152_T1_2mm_brain -out ${subj_dir}/mask/mni/cen_mni.nii -init ${example_func2mni_mat} -applyxfm -interp nearestneighbour -datatype short
    fi

    # Display masks with FSLEYES
    fsleyes $examplefunc ${mni2example_func} ${dmn_thresh} -cm blue ${cen_thresh} -cm red

fi

# For registering masks in resting state space to 2vol space
if [ ${step} = register ]
then
    clear
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo "+ Registering masks to study_ref"
    
    # Version sort handles series numbers naturally (series1, series2, series3...)
    latest_ref=$(ls -v $subj_dir/xfm/series[0-9]*_ref.nii 2>/dev/null | grep -E "series[0-9]+_ref\.nii$" | tail -n1)
    # Remove the .nii extension
    latest_ref="${latest_ref::-4}"
    
    # Check if we found a valid reference
    if [ -z "$latest_ref" ]; then
        echo "ERROR: No series reference file found!"
        exit 1
    fi
    
    study_ref=${subj_dir}/xfm/study_ref.nii

    # Create QC directory if it doesn't exist
    mkdir -p $subj_dir/qc

    # if localizer_ref images doesn't exist yet, make it
    if [ ! -f ${subj_dir}/xfm/localizer_ref.nii ]
    then
        cp ${study_ref} ${subj_dir}/xfm/localizer_ref.nii
    fi
    
    echo "+ Registering masks to reference image from most recent series: ${latest_ref}"
    echo "+ study_ref.nii is now ${latest_ref}"

    # Move the latest reference image to be study_ref
    # study_ref.nii is used by MURFI to register to 1st volume of feedback runs
    # So ROI masks need to be in the same space as study_ref.nii
    cp ${latest_ref}.nii ${study_ref}

    bet ${latest_ref} ${latest_ref}_brain -R -f 0.4 -g 0 -m
    slices ${latest_ref} ${latest_ref}_brain_mask -o $subj_dir/qc/2vol_skullstrip_brain_mask_check.gif

    rm -rf $subj_dir/xfm/epi2reg
    mkdir -p $subj_dir/xfm/epi2reg

    # Determine ICA directory
    ica_directory=$subj_dir/rest/rs_network.gica/groupmelodic.ica/

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
    
    # Register resting state space (median of rest run1) to 2vol space (study_ref)
    examplefunc=$subj_dir_absolute/rest/$subj'_'$ses'_task-rest_run-01_bold_mcflirt_median_bet.nii'
    
    # Calculate transformation matrix from rest to study_ref
    flirt -in $examplefunc -ref ${latest_ref}_brain -out $subj_dir/xfm/epi2reg/rest2studyref_brain -omat $subj_dir/xfm/epi2reg/rest2studyref.mat
    
    # Make registration image for inspection
    slices $subj_dir/xfm/epi2reg/rest2studyref_brain ${latest_ref}_brain -o $subj_dir/qc/rest_warp_to_2vol_native_check.gif

    # Register masks from ORIGINAL rest space to study_ref space
    for mask_name in {'dmn','cen'};
    do 
        echo "+ REGISTERING ${mask_name} TO study_ref" 

        # Check if original rest-space masks exist
        if [ ! -f "$subj_dir/mask/${mask_name}_rest_original.nii" ]; then
            echo "ERROR: Original rest-space mask not found: ${mask_name}_rest_original.nii"
            echo "Please run process_roi_masks step first"
            exit 1
        fi

        # Register mask from ORIGINAL rest space to study_ref space
        flirt -in $subj_dir/mask/${mask_name}_rest_original.nii \
              -ref ${latest_ref}_brain \
              -out $subj_dir/mask/${mask_name}_temp \
              -init $subj_dir/xfm/epi2reg/rest2studyref.mat \
              -applyxfm -interp nearestneighbour -datatype short    

        # Use study_ref's brain mask with 4-voxel erosion to ensure masks stay inside brain
        fslmaths ${latest_ref}_brain_mask -ero -ero -ero -ero ${latest_ref}_brain_mask_ero4

        # Multiply registered mask by eroded study_ref brain mask
        fslmaths $subj_dir/mask/${mask_name}_temp.nii.gz -mul ${latest_ref}_brain_mask_ero4 $subj_dir/mask/${mask_name}_studyref.nii.gz -odt short

        # Uncompress
        if [ -f "$subj_dir/mask/${mask_name}_studyref.nii.gz" ]; then
            gunzip -f $subj_dir/mask/${mask_name}_studyref.nii.gz
        fi
        
        # Apply final brain mask cleanup
        fslmaths ${latest_ref}_brain -bin ${latest_ref}_brain_bin
        fslmaths $subj_dir/mask/${mask_name}_studyref.nii -mul ${latest_ref}_brain_bin $subj_dir/mask/${mask_name}_studyref.nii -odt short
        
        # Report final voxel count
        final_voxels=$(fslstats $subj_dir/mask/${mask_name}_studyref.nii -V | awk '{print $1}')
        echo "   Final ${mask_name} mask in study_ref space: ${final_voxels} voxels"
        
        # Clean up temporary files
        rm -f $subj_dir/mask/${mask_name}_temp.nii*
    done

    # Create final MURFI-ready masks (what MURFI actually uses)
    echo "+ Creating final MURFI masks (cen.nii and dmn.nii)"
    cp $subj_dir/mask/cen_studyref.nii $subj_dir/mask/cen.nii
    cp $subj_dir/mask/dmn_studyref.nii $subj_dir/mask/dmn.nii
    
    echo "   Created mask/cen.nii ($(fslstats $subj_dir/mask/cen.nii -V | awk '{print $1}') voxels)"
    echo "   Created mask/dmn.nii ($(fslstats $subj_dir/mask/dmn.nii -V | awk '{print $1}') voxels)"

    # Clean up erosion and temp files
    rm -f ${latest_ref}_brain_mask_ero4.nii*
    rm -f ${latest_ref}_brain_bin.nii*

    echo "+ INSPECT"
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    xdg-open $subj_dir/qc/2vol_skullstrip_brain_mask_check.gif
    xdg-open $subj_dir/qc/rest_warp_to_2vol_native_check.gif
    fsleyes ${latest_ref}_brain $subj_dir/mask/cen.nii -cm red $subj_dir/mask/dmn.nii -cm blue

fi

if [ ${step} = cleanup ]
then
    input_string=$(zenity --forms --title="Delete files?" \
    --separator=" " \
    --text="`printf "Are you sure you want to clean up the directory and delete files for ${subj}?\nThe entire img folder will be deleted, as well as raw bold data from the rest directory"`" \
    --cancel-label "Exit" --ok-label "Delete files")
    ret=$?

    # If user selects the Exit button, then quit MURFI
    if [[ $ret == 1 ]];
    then
        exit 0
    fi

    # Delete img folder and large bold files from the rest folder
    rm -rf $subj_dir/img
    rm -f $subj_dir/rest/*bold.nii
    rm -f $subj_dir/rest/*bold_mcflirt.nii
    rm -f $subj_dir/rest/*bold_mcflirt_masked.nii
fi


# ONLY if regular mask generation isn't possible
# As a backup, we can register the template masks to 2vol space
if [ ${step} = backup_reg_mni_masks_to_2vol ]
then
    clear
    mni_template=MNI152_T1_2mm_LPS_brain.nii
    dmn_mni=DMNax_brainmaskero2_lps.nii
    cen_mni=CENa_brainmaskero2_lps.nii

    two_vol_ref=$(ls -t $subj_dir/xfm/series*.nii | head -n1)
    two_vol_ref="${two_vol_ref::-4}"
    
    two_vol_ref_bet=${subj_dir}/xfm/two_vol_ref_bet.nii
    two_vol_ref2mni=${subj_dir}/xfm/two_vol_ref2mni.nii
    two_vol_ref2mni_mat=${subj_dir}/xfm/two_vol_ref2mni.mat
    mni2_two_vol_ref_mat=${subj_dir}/xfm/mni2_two_vol_ref.mat

    study_ref=${subj_dir}/xfm/study_ref.nii
    # if localizer_ref images doesn't exist yet, make it
    if [ ! -f ${subj_dir}/xfm/localizer_ref.nii ]
    then
        mv ${study_ref} ${subj_dir}/xfm/localizer_ref.nii
    fi
    echo "Registering masks to reference image from most recent series: ${two_vol_ref}"
    echo "study_ref.nii is now ${two_vol_ref}"

    # Move the latest reference image to be study_ref
    # study_ref.nii is used by MURFI to register to 1st volume of feedback runs
    # So ROI masks need to be in the same space as study_ref.nii
    cp ${two_vol_ref}.nii ${study_ref}


    # skullstrip 2vol before registration
    bet ${two_vol_ref} ${two_vol_ref_bet} -R -f 0.4 -g 0 -m # changed from -f 0.6


    # first, register the 2vol to mni
    # then calculate the inverse of the registration
    flirt -in ${two_vol_ref_bet} -ref ${mni_template} -out ${two_vol_ref2mni} -omat ${two_vol_ref2mni_mat}
    convert_xfm -omat ${mni2_two_vol_ref_mat} -inverse ${two_vol_ref2mni_mat}


    # "apply" the inverse of the registration to dmn/cen masks
    #put them in the participant mask folder

    #DMN
    flirt -in ${dmn_mni} -ref ${two_vol_ref_bet} -out $subj_dir/mask/dmn.nii -init ${mni2_two_vol_ref_mat} -applyxfm -interp nearestneighbour -datatype short

    #CEN
    flirt -in ${cen_mni} -ref ${two_vol_ref_bet} -out $subj_dir/mask/cen.nii -init ${mni2_two_vol_ref_mat} -applyxfm -interp nearestneighbour -datatype short

fi
