#!/bin/bash
# RtBPD Real-time Neurofeedback Launch Script
# New advanced GUI structure adapted from R33
# Created by Clemens C.C. Bauer 09.2025


# Add this line to your script before any FSL commands
export FSLOUTPUTTYPE=NIFTI
echo "Setting FSLOUTPUTTYPE to NIFTI"


# Initialize conda environment
if [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate base
elif [ -f ~/.bashrc ]; then
    source ~/.bashrc
    if command -v conda &> /dev/null; then
        conda activate base
    fi
fi

# Set up logging
LOG_DIR="../logs"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/rtbpd_session_$(date +%F_%H-%M-%S).log"

# Function to log with timestamp
log_message() {
    echo "[$(date +%F_%T)] $1" | tee -a "$MAIN_LOG"
}

# Function to check system requirements
check_requirements() {
    log_message "Checking system requirements..."
    
    # Check for required software
    local missing_deps=()
    
    command -v fsl >/dev/null 2>&1 || missing_deps+=("FSL")
    command -v singularity >/dev/null 2>&1 || missing_deps+=("Singularity")
    python -c "import nipype" 2>/dev/null || missing_deps+=("nipype")
    python -c "import nilearn" 2>/dev/null || missing_deps+=("nilearn")
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        zenity --error --text="Missing dependencies: ${missing_deps[*]}\nPlease install before continuing."
        exit 1
    fi
    
    # Check CPU cores for MELODIC optimization
    local cpu_cores=$(nproc)
    export OMP_NUM_THREADS=$cpu_cores
    export FSL_CPU_CORES=$cpu_cores
    log_message "System configured for $cpu_cores CPU cores"
}

# Function to check required files and show status
check_required_files() {
    local participant_id=$1
    
    log_message "Checking required files for $participant_id"
    
    # Check what files are missing for RtBPD
    subject_dir="../subjects/$participant_id"
    xml_dir="$subject_dir/xml"
    mask_dir="$subject_dir/mask"
    
    xml_ready=false
    masks_ready=false
    subject_ready=false
    
    # Check if subject directory exists
    if [[ -d "$subject_dir" ]]; then
        subject_ready=true
    fi
    
    # Check essential XML files for RtBPD
    if [[ -f "$xml_dir/2vol.xml" ]] && [[ -f "$xml_dir/rest.xml" ]] && [[ -f "$xml_dir/rtdmn.xml" ]]; then
        xml_ready=true
    fi
    
    # Check masks status
    if [[ -d "$mask_dir" ]]; then
        mask_count=$(find "$mask_dir" -name "*.nii" -type f 2>/dev/null | wc -l)
        if [[ $mask_count -gt 0 ]]; then
            masks_ready=true
            log_message "Masks found: $mask_count files in $mask_dir"
        fi
    fi
    
    # Show status and appropriate options
    if [[ "$subject_ready" == true ]] && [[ "$xml_ready" == true ]] && [[ "$masks_ready" == true ]]; then
        # All files ready - show setup only
        status_text="<span foreground='green'>✓ Subject directory: Complete</span>\n<span foreground='green'>✓ Essential XML files: Complete</span>\n<span foreground='green'>✓ Subject masks: Complete</span>\n\nAll files ready for neurofeedback!"
        action=$(zenity --list --title="RtBPD Subject: $participant_id Ready" \
            --text="$status_text" \
            --column="Action" --column="Description" \
            "setup" "Setup system to start neurofeedback" \
            "download_files" "Download/update files from server" \
            --width=700 --height=350 \
            --cancel-label="Exit")
    else
        # Some files missing - show status and all options
        status_text=""
        if [[ "$subject_ready" == true ]]; then
            status_text+="<span foreground='green'>✓ Subject directory: Exists</span>\n"
        else
            status_text+="<span foreground='orange'>✗ Subject directory: Missing</span>\n"
        fi
        
        if [[ "$xml_ready" == true ]]; then
            status_text+="<span foreground='green'>✓ Essential XML files: Complete</span>\n"
        else
            status_text+="<span foreground='orange'>✗ Essential XML files: Missing</span>\n"
        fi
        
        if [[ "$masks_ready" == true ]]; then
            status_text+="<span foreground='green'>✓ Subject masks: Complete</span>\n"
        else
            status_text+="<span foreground='orange'>✗ Subject masks: Missing</span>\n"
        fi
        
        status_text+="\nComplete missing items before setup:"
        
        action=$(zenity --list --title="RtBPD Subject: $participant_id - Files Status" \
            --text="$status_text" \
            --column="Action" --column="Description" \
            "create" "Create subject directory and setup files" \
            "download_files" "Download files from server" \
            "setup" "Setup system to start neurofeedback" \
            --width=700 --height=400 \
            --cancel-label="Exit")
    fi
    
    if [[ -z "$action" ]]; then
        log_message "No action selected, exiting"
        exit 0
    fi
    
    # Execute selected action
    execute_rtbpd_step "$participant_id" "$action"
}

# Main GUI function for RtBPD
main_gui() {
    # Step 1: Always ask for subject ID first
    if [[ -z $MURFI_SUBJECT_NAME ]] || [[ "$1" == "--new-subject" ]]; then
        # Get subject number
        subject_number=$(zenity --entry --title="RtBPD Real-time Neurofeedback" \
            --text="RtBPD Study - Real-time Neurofeedback Training\n\nEnter subject number (e.g., 001, 999, 123...):\n\nWill create: sub-rtbpd[NUMBER]" \
            --width=450 \
            --cancel-label "Exit")
        
        ret=$?
        if [[ $ret == 1 ]] || [[ -z "$subject_number" ]]; then
            log_message "RtBPD session cancelled by user"
            exit 0
        fi
        
        # Format participant_id
        participant_id="sub-rtbpd${subject_number}"
        log_message "RtBPD subject ID: $participant_id"
        
        # Step 2: Check if subject directory exists
        subject_dir="../subjects/$participant_id"
        
        if [[ -d "$subject_dir" ]]; then
            # Subject exists - ask overwrite or proceed
            if zenity --question --text="<span foreground='red'><b>Subject $participant_id already exists!</b></span>\n\nDo you want to overwrite the existing directory?" --width=450; then
                # Overwrite - delete and recreate
                log_message "Overwriting existing subject directory for $participant_id"
                rm -rf "$subject_dir"
                
                # Create fresh directory structure
                mkdir -p "$subject_dir"/{xml,mask,xfm,rest,img,log,fsfs,qc}
                
                # Copy template files if they exist
                if [[ -d "../subjects/rtbpd_template" ]]; then
    			cp -r ../subjects/rtbpd_template/* "$subject_dir/"
    			# Move XML files from xml_orig to xml root
   		if [[ -d "$subject_dir/xml/xml_orig" ]]; then
        			cp "$subject_dir/xml/xml_orig"/*.xml "$subject_dir/xml/"
    			fi
   		log_message "Fresh directory created with template files for $participant_id"
                    zenity --info --text="Directory recreated successfully!\n\nFresh structure ready for $participant_id" --width=450
                else
                    log_message "No template directory found, created empty structure"
                    zenity --info --text="Directory recreated successfully!\n\nEmpty structure ready for $participant_id" --width=450
                fi
                
                # Create command log
                echo "# RtBPD MURFI Command Log for $participant_id" > "$subject_dir/murfi_command_log.txt"k......
            else
                # Proceed with existing directory
                log_message "Using existing directory for $participant_id"
            fi
        else
            # New subject - create directory structure
            log_message "Creating new subject directory for $participant_id"
            
            mkdir -p "$subject_dir"/{xml,mask,xfm,rest,img,log,fsfs}
            
            # Copy template files if they exist

            if [[ -d "../subjects/rtbpd_template" ]]; then
    		cp -r ../subjects/rtbpd_template/* "$subject_dir/"
    		# Move XML files from xml_orig to xml root
   		if [[ -d "$subject_dir/xml/xml_orig" ]]; then
        		cp "$subject_dir/xml/xml_orig"/*.xml "$subject_dir/xml/"
    		fi
   		log_message "Fresh directory created with template files for $participant_id"
                    
                zenity --info --text="New subject directory created!\n\nStructure ready for $participant_id" --width=450
            else
                log_message "No template directory found, created empty structure"
                zenity --info --text="New subject directory created!\n\nEmpty structure ready for $participant_id" --width=450
            fi
            
            # Create command log
            echo "# RtBPD MURFI Command Log for $participant_id" > "$subject_dir/murfi_command_log.txt"
        fi
        
        # Step 3: Check for required files
        export MURFI_SUBJECT_NAME="$participant_id"
        check_required_files "$participant_id"
        
    else
        # Subject already loaded - show RtBPD step selection
        step=$(zenity --list --title="RtBPD Real-time Neurofeedback" \
            --text="RtBPD PARTICIPANT: ${MURFI_SUBJECT_NAME}\nReal-time Neurofeedback Protocol\n\nSystem: $(nproc) CPU cores\n\nSelect processing step:" \
            --column="Step" --column="Description" \
            "setup" "Setup system to start neurofeedback" \
            "2vol" "Receive 2-volume scan for registration" \
            "resting_state" "Acquire resting state data" \
            "extract_rs_networks" "Extract resting state networks" \
            "process_roi_masks" "Process ROI masks" \
            "register" "Register masks to native space" \
            "transferpre" "Transfer pre-training scan" \
            "feedback" "Real-time neurofeedback session" \
            "transferpost" "Transfer post-training scan" \
            "cleanup_backup" "Clean up files and backup to server" \
            "download_files" "Download files from server" \
            "backup_reg_mni_masks_to_2vol" "Backup registered masks" \
            --width=800 --height=500 \
            --cancel-label "Exit")
        
        ret=$?
        if [[ $ret == 1 ]]; then
            log_message "RtBPD session ended by user for ${MURFI_SUBJECT_NAME}"
            exit 0
        fi
        
        # Execute selected step
        execute_rtbpd_step "$MURFI_SUBJECT_NAME" "$step"
    fi
}

# RtBPD-specific step execution
execute_rtbpd_step() {
    local participant_id=$1
    local step=$2
    
    local subj_dir="../subjects/$participant_id"
    local cmd_log="$subj_dir/murfi_command_log.txt"
    
    log_message "Executing RtBPD step: $step for $participant_id"
    
    case "$step" in
        'download_files')
            log_message "Downloading complete subject data for $participant_id"
            
            # Get username for server access
            username=$(zenity --entry --title="Server Authentication" \
                --text="Enter your username for server access:" \
                --width=400)
            
            if [[ -z "$username" ]]; then
                zenity --error --text="Username required for download"
                return
            fi
            
            echo "[$(date +%F_%T)] Downloading complete subject data for $participant_id" >> "$cmd_log"
            
            zenity --info --text="Downloading complete subject directory...\n\nYou will be prompted for password in terminal." --width=450
            
            # Download the complete backed-up subject directory
            rsync -avrz "${username}@xfer.discovery.neu.edu:/work/swglab/data/rtBPD/sourcedata/murfi/$participant_id/" "../subjects/$participant_id/"
            
            rsync_ret=$?
            if [[ $rsync_ret == 0 ]]; then
                log_message "Complete subject data downloaded successfully for $participant_id"
                zenity --info --text="Complete subject data downloaded successfully!\n\nAll files restored for $participant_id" --width=400
            else
                log_message "Download failed for $participant_id"
                zenity --error --text="Download failed.\n\nPlease check your credentials, network connection, and that the subject data exists on server." --width=400
            fi
            
            # Return to file checking after download
            check_required_files "$participant_id"
            return
            ;;
        'cleanup_backup')
            log_message "Running cleanup and backup for $participant_id"
            
            # Confirm cleanup operation
            if zenity --question --title="Delete files and backup?" \
                --text="Are you sure you want to clean up the directory and backup data for ${participant_id}?\n\nLarge data files will be deleted, then remaining data will be backed up to server" \
                --cancel-label "Cancel" --ok-label "Clean and Backup" \
                --width=500; then
                
                log_message "User confirmed cleanup and backup for $participant_id"
                
                # Step 1: Delete large files
                rm -rf "$subj_dir/img"
                rm -f "$subj_dir/rest/"*bold.nii
                rm -f "$subj_dir/rest/"*bold_mcflirt.nii
                rm -f "$subj_dir/rest/"*bold_mcflirt_masked.nii
                
                log_message "Cleanup completed for $participant_id"
                
                # Step 2: Get username for backup
                username=$(zenity --entry --title="Backup Authentication" \
                    --text="Enter your username for data backup to server:\n\nPassword will be requested in terminal" \
                    --cancel-label "Skip Backup" \
                    --width=400)
                
                # If user cancels, skip the backup
                if [[ -z "$username" ]]; then
                    log_message "Backup cancelled by user"
                    zenity --info --text="Cleanup completed.\nBackup skipped." --width=300
                    return
                fi
                
                # Step 3: Perform backup
                log_message "Starting backup for $participant_id to server"
                zenity --info --text="Starting backup...\n\nYou will be prompted for password in terminal.\nCheck terminal window." --width=400
                
                # Backup cleaned data to server
                local subj_dir_full="../subjects/$participant_id"
                rsync -avrz --chmod=g+rwx --perms "$subj_dir_full" "${username}@xfer.discovery.neu.edu:/work/swglab/data/rtBPD/sourcedata/murfi/"
                rsync_ret=$?
                
                # Check backup result
                if [[ $rsync_ret == 0 ]]; then
                    log_message "Cleanup and backup completed successfully for $participant_id"
                    zenity --info --text="Cleanup and backup completed successfully!\n\nData cleaned and backed up to server." --width=400
                else
                    log_message "Backup failed for $participant_id after cleanup"
                    zenity --error --text="Cleanup completed but backup failed.\n\nPlease check your credentials and network connection." --width=400
                fi
            else
                log_message "Cleanup and backup cancelled by user"
                zenity --info --text="Cleanup and backup cancelled." --width=300
            fi
            ;;
        'cleanup')
            log_message "Running cleanup for $participant_id"
            
            # Confirm cleanup operation
            if zenity --question --title="Delete files?" \
                --text="Are you sure you want to clean up the directory and delete files for ${participant_id}?\n\nLarge data files will be deleted" \
                --cancel-label "Cancel" --ok-label "Delete files" \
                --width=500; then
                
                log_message "User confirmed cleanup for $participant_id"
                
                # Delete large files
                rm -rf "$subj_dir/img"
                rm -f "$subj_dir/rest/"*bold.nii
                rm -f "$subj_dir/rest/"*bold_mcflirt.nii
                
                log_message "Cleanup completed for $participant_id"
                zenity --info --text="Cleanup completed successfully!" --width=400
            else
                log_message "Cleanup cancelled by user"
                zenity --info --text="Cleanup operation cancelled." --width=300
            fi
            ;;
        'backup')
            log_message "Backing up data for $participant_id"
            
            # Get username for server access
            username=$(zenity --entry --title="Backup Authentication" \
                --text="Enter your username for data backup to server:" \
                --width=400)
            
            if [[ -z "$username" ]]; then
                zenity --error --text="Username required for backup"
                return
            fi
            
            echo "[$(date +%F_%T)] Backing up data for $participant_id" >> "$cmd_log"
            
            zenity --info --text="Starting backup...\n\nYou will be prompted for password in terminal." --width=450
            
            # Backup to server directory (adapted from R33)
            local subj_dir_full="../subjects/$participant_id"
            rsync -avrz --chmod=g+rwx --perms "$subj_dir_full" "${username}@xfer.discovery.neu.edu:/work/swglab/data/rtBPD/sourcedata/murfi/"
            
            rsync_ret=$?
            if [[ $rsync_ret == 0 ]]; then
                log_message "Backup completed successfully for $participant_id"
                zenity --info --text="Backup completed successfully!" --width=400
            else
                log_message "Backup failed for $participant_id"
                zenity --error --text="Backup failed.\n\nPlease check your credentials and network connection." --width=400
            fi
            ;;
        'create')
            log_message "Running createxml.sh for $participant_id"
            echo "[$(date +%F_%T)] source createxml.sh $participant_id setup" >> "$cmd_log"
            clear
            #source createxml.sh "$participant_id" setup 2>&1 | tee -a "$MAIN_LOG"
            ;;
        'setup')
            log_message "Running system setup for RtBPD neurofeedback"
            echo "[$(date +%F_%T)] source feedback.sh $participant_id setup" >> "$cmd_log"
            
            # Run setup in current terminal, then return to GUI immediately
            echo "Running setup for $participant_id..."
            source feedback.sh "$participant_id" setup
            
            log_message "Setup completed for $participant_id"
            
            # Ask user about network connectivity with yellow warning
            if zenity --question --icon=warning \
                --text="<span foreground='#FF8C00'><b>Network Connectivity Check</b></span>\n\nDid the ping tests to the scanner and stimulation computer work correctly?\n\n• Scanner (192.168.2.1)\n• Stim Computer (192.168.2.6)\n\nSelect your choice:" \
                --ok-label="Pings Worked - Continue" \
                --cancel-label="Retry Setup" \
                --width=500; then
                
                log_message "User confirmed network connectivity is working for $participant_id"
            else
                log_message "User reported network issues, retrying setup for $participant_id"
                zenity --info --text="Retrying setup for network connectivity...\n\nRunning setup again." --width=400
                execute_rtbpd_step "$participant_id" "setup"
                return
            fi
            ;;
        'transferpre')
            log_message "Running transfer pre-training scan for $participant_id"
            echo "[$(date +%F_%T)] source feedback.sh $participant_id feedback transferpre" >> "$cmd_log"
            clear
            source feedback.sh "$participant_id" feedback
            ;;
        'transferpost')
            log_message "Running transfer post-training scan for $participant_id"
            echo "[$(date +%F_%T)] source feedback.sh $participant_id feedback transferpost" >> "$cmd_log"
            clear
            source feedback.sh "$participant_id" feedback
            ;;
        *)
            # Run standard feedback.sh steps
            log_message "Running feedback.sh for $participant_id with step $step"
            echo "[$(date +%F_%T)] feedback.sh $participant_id $step" >> "$cmd_log"
            clear
            source feedback.sh "$participant_id" "$step"
            ;;
    esac
    
    # Check execution status
    if [[ $? -eq 0 ]]; then
        log_message "RtBPD step '$step' completed successfully for $participant_id"
        
        # Auto-advance logic for certain steps
        case "$step" in
            'setup')
                zenity --info --text="Setup verified and complete for $participant_id!\n\nReady for neurofeedback workflow." --width=400
                ;;
            '2vol')
                zenity --info --text="2-volume scan complete!\n\nReady for next step." --width=400
                ;;
            'register')
                zenity --info --text="Mask registration complete!\n\nReady for transfer pre-training." --width=400
                ;;
            'transferpre')
                zenity --info --text="Transfer pre-training scan complete!\n\nReady for neurofeedback sessions." --width=400
                ;;
            'transferpost')
                zenity --info --text="Transfer post-training scan complete!\n\nRtBPD session finished." --width=400
                ;;
        esac
    fi
}

# Main execution for RtBPD
main() {
    log_message "Starting RtBPD Real-time Neurofeedback System"
    
    # Clear any existing subject name to force new input
    unset MURFI_SUBJECT_NAME
    
    # System checks
    check_requirements
    
    # Set RtBPD environment
    export MURFI_SUBJECTS_DIR="$(dirname $(pwd))/subjects/"
    export FSLOUTPUTTYPE=NIFTI_GZ
    
    # Display RtBPD study info first
    zenity --info --text="RtBPD Study\nReal-time Neurofeedback\n\nSystem ready with $(nproc) CPU cores" --width=400
    
    # After completing initial setup, go directly to neurofeedback workflow
    while true; do
        if [[ -n "$MURFI_SUBJECT_NAME" ]]; then
            main_gui
        else
            # If subject name is cleared, restart from beginning
            main_gui --new-subject
        fi
    done
    
    log_message "RtBPD MURFI session ended"
}

# Execute main function
main "$@"
