import os
import sys
import shutil
import argparse 
from datetime import datetime
from joblib import Parallel, delayed

import functions
from dicom_to_nifti import DicomToNiFTI
from motion_characterization import MotionCharacterization
from graph_transforms import GraphTransformDirectory
from remove_background import RemoveBackground
from start_motion_correction import StartMotionCorrection
from run_fmriprep import RunfMRIPrep


def parse_config_files(configuration_files):
    
    all_input_variables = {}
    for config_file in configuration_files:
        print(f"Processing Configuration File: {config_file}")
        config_file_path = os.path.abspath(config_file)
        input_variables = prepare_config_file_vars(config_file_path)
        all_input_variables[config_file_path] = input_variables    

    all_working_dirs = [input_variables['WORKING_DIRECTORY'] for input_variables in all_input_variables.values()]
    if len(set(all_working_dirs)) < len(all_working_dirs):
        print(f"\nERROR: Please do not use the same WORKING_DIRECTORY across multiple configuration files.")
        sys.exit(1)
    all_output_dirs = [input_variables['OUTPUT_DIRECTORY'] for input_variables in all_input_variables.values()]
    if len(set(all_output_dirs)) < len(all_output_dirs):
        print(f"\nERROR: Please do not use the same OUTPUT_DIRECTORY across multiple configuration files.")
        sys.exit(1)

    fmri_prep_participants = [
        (input_variables['FMRIPREP_PARTICIPANT_ID'], input_variables['FMRIPREP_SESSION_ID'], input_variables['FMRIPREP_RUN_NUM'], input_variables['TASK_NAME'])
        for input_variables in all_input_variables.values() 
        if input_variables['RUN_FMRIPREP'].lower() == 'true'
    ]
    if len(set(fmri_prep_participants)) < len(fmri_prep_participants):
        print(f"\nERROR: Please do not use the same combination of FMRIPREP_PARTICIPANT_ID, FMRIPREP_SESSION_ID, FMRIPREP_RUN_NUM, and TASK_NAME across multiple configuration files.")
        sys.exit(1)
    
    return all_input_variables


def validate_input_data(input_variables, config_file):

    def exit_message(dir_var_name, nifti_var_name, json_var_name):
        print("\nERROR: ")
        print(f"Please input a value for either {dir_var_name}")
        print("OR")
        print(f"{nifti_var_name} and {json_var_name}")
        print(f"Edit your configuration file at: {config_file}")
        sys.exit(1)
    
    if not input_variables['DICOM_DIRECTORY_PATH'] and not input_variables['NIFTI_IMAGE_PATH'] and not input_variables['JSON_FILE_PATH']:
        exit_message('DICOM_DIRECTORY_PATH', 'NIFTI_IMAGE_PATH', 'JSON_FILE_PATH')
        
    if input_variables['DICOM_DIRECTORY_PATH'] and input_variables['NIFTI_IMAGE_PATH']:
        exit_message('DICOM_DIRECTORY_PATH', 'NIFTI_IMAGE_PATH', 'JSON_FILE_PATH')
    
    if input_variables['DICOM_DIRECTORY_PATH'] and input_variables['JSON_FILE_PATH']:
        exit_message('DICOM_DIRECTORY_PATH', 'NIFTI_IMAGE_PATH', 'JSON_FILE_PATH')
    
    if input_variables['NIFTI_IMAGE_PATH'] and not input_variables['JSON_FILE_PATH']:
        exit_message('DICOM_DIRECTORY_PATH', 'NIFTI_IMAGE_PATH', 'JSON_FILE_PATH')
    
    if input_variables['JSON_FILE_PATH'] and not input_variables['NIFTI_IMAGE_PATH']:
        exit_message('DICOM_DIRECTORY_PATH', 'NIFTI_IMAGE_PATH', 'JSON_FILE_PATH')


def validate_input_variables(input_variables, config_file):
    
    def exit_message(message, config_file):
        print(f"\nERROR: {message}")
        print(f"Please edit: {config_file}")
        sys.exit(1)

    # validate_input_data(input_variables, config_file)
    vars_with_required_inputs = [
        'MOTION_THRESHOLD',
        'WORKING_DIRECTORY',
        'OUTPUT_DIRECTORY',

        # Script paths
        'RUN_PIPELINE_SCRIPT',
        

        'SMS_MI_REG_EXECUTABLE_PATH',

        'MCORR_MAIN_SCRIPT_DIR',
        'MCORR_MAIN_SCRIPT',
        'MCORR_OPERATORS_DIR',
        'MCORR_DIRECT_LIFTANDUNLIFT_CODES_DIR',
        
        # Conda paths
        'CONDA_INIT_PATH',
        'CONDA_ENV_NAME',
        'CONDA_ENV_PYTHON_PATH',

        # Other software paths (may only exist on compute nodes)
        'SMS_MI_REG_EXECUTABLE_PATH',
        'MATLAB_INSTALLATION_PATH',
        'DCM2NIIX_PATH',
        'BIOGRIDS_PATH',
        'DCMDJPEG_PATH',

        
    ]
    if input_variables['RUN_FMRIPREP'].lower() == 'true':
        vars_with_required_inputs.extend([
            'FMRIPREP_PARTICIPANT_ID',
            'FMRIPREP_SESSION_ID',
            'FMRIPREP_CONTAINER_PATH',
            'FMRIPREP_TEMPLATEFLOW_DIRECTORY',
            'FMRIPREP_LICENSE_PATH',
            'FMRIPREP_TMP_DIRECTORY',
            'FMRIPREP_INPUT_DIRECTORY',
            'FMRIPREP_DATASET_JSON_PATH',
            'FMRIPREP_OUTPUT_DIRECTORY',
            'FMRIPREP_WORKING_DIRECTORY',
        ])   

    if input_variables['WORKING_DIRECTORY'][-1] != '/' or input_variables['OUTPUT_DIRECTORY'][-1] != '/':
        exit_message(
            message=f"Please make sure WORKING_DIRECTORY and OUTPUT_DIRECTORY end in /",
            config_file=config_file
        )
    # paths that don't need to exist yet but must have an input, or variables that must exist but aren't paths
    for var_name in vars_with_required_inputs:
        if not var_name in input_variables.keys():
            exit_message(
                message=f"Your Config File is Missing the Variable: {var_name}",
                config_file=config_file
            )
        if not input_variables[var_name]:
            exit_message(
                message=f"Your Config File Must Include a Value for the Variable: {var_name}",
                config_file=config_file
            )
    
    if input_variables['REFERENCE_VOLUME_INDEX'] and input_variables['REFERENCE_VOLUME_PATH']:
        exit_message(
            message="Please enter a value for either: \
                \nREFERENCE_VOLUME_INDEX \
                \n or \
                \nREFERENCE_VOLUME_PATH \
                \n or neither. \
                \nPlease do not enter a value for both.",
            config_file=config_file
        )
    
    elif not input_variables['REFERENCE_VOLUME_INDEX'] and not input_variables['REFERENCE_VOLUME_PATH'] and not input_variables['GET_REFERENCE_VOLUME_SCRIPT_PATH']:
        exit_message(
            message="Please enter a value for: \
                \nGET_REFERENCE_VOLUME_SCRIPT_PATH \
                \n if you are not entering either: \
                \nREFERENCE_VOLUME_INDEX or REFERENCE_VOLUME_PATH",
            config_file=config_file
        )

    if input_variables['RUN_FMRIPREP'].lower() == 'true':
        
        if input_variables['INPUT_ANAT_DICOM_DIRECTORY']:
            nondicom_anat_paths = ('T1W_IMAGE_PATH', 'T1W_JSON_PATH', 'T2W_IMAGE_PATH', 'T2W_JSON_PATH')
            for path in nondicom_anat_paths:
                if input_variables[path]:
                    exit_message(
                        message=f"If you inputted a value for INPUT_ANAT_DICOM_DIRECTORY, the following variables should be empty: \
                                \n{', '.join(nondicom_anat_paths)}",
                        config_file=config_file)
       
        else:
            t1_pair = ('T1W_IMAGE_PATH', 'T1W_JSON_PATH')
            t2_pair = ('T2W_IMAGE_PATH', 'T2W_JSON_PATH')                

            for anat_pair in (t1_pair, t2_pair):
                message = f"\nPlease enter a value for BOTH:\n{anat_pair[0]}\nAND\n{anat_pair[1]}\nor NEITHER.\nPlease do not enter a value for just one."
                if "T1W" in anat_pair[0]:
                    if not input_variables[anat_pair[1]] and not input_variables[anat_pair[0]]:
                        exit_message(
                            message=f"Please enter a value for either: \
                                \nINPUT_ANAT_DICOM_DIRECTORY \
                                \n or \
                                \n{anat_pair[0]} and {anat_pair[1]}.",
                            config_file=config_file
                        )  

                elif input_variables[anat_pair[0]] and not input_variables[anat_pair[1]]:
                    exit_message(
                        message=message,
                        config_file=config_file
                    )
                    
                elif input_variables[anat_pair[1]] and not input_variables[anat_pair[0]]:
                    exit_message(
                        message=message,
                        config_file=config_file
                    )
       
        try:
            if float(input_variables['MOTION_THRESHOLD']) < 0 or  float(input_variables['MOTION_THRESHOLD']) > 100:
                raise ValueError
            
        except ValueError:
            exit_message(
                message="MOTION_THRESHOLD must contain a number between 0 and 100.",
                config_file=config_file
            )
                
        if not input_variables['FMRIPREP_FUNC_IMAGE_PATH'] and not ['MCORR_OUTPUT_FILENAME_PATTERN']:
            exit_message(
                message=f"If you are not inputting a path for FMRIPREP_FUNC_IMAGE_PATH, please enter the value of MCORR_OUTPUT_FILENAME_PATTERN \
                  \nso we can find the nifti image created by the motion correction software.",
                config_file=config_file
            )        
        check_if_ints = ['FMRIPREP_PARTICIPANT_ID', 'FMRIPREP_SESSION_ID']
        if input_variables['FMRIPREP_RUN_NUM']:
            check_if_ints.append('FMRIPREP_RUN_NUM')

        for input_variable_name in check_if_ints:
            try:
                if int(input_variables[input_variable_name]) < 0 or int(input_variables[input_variable_name]) > 100:
                    raise ValueError
            except ValueError:
                exit_message(
                    message=f"{input_variable_name} must contain an integer (between 0 and 100).",
                    config_file=config_file
                )


def prepare_config_file_vars(config_file):
    
    input_variables = functions.get_input_variables(config_file, verbose=True)
    validate_input_variables(input_variables, config_file)

    os.makedirs(input_variables['WORKING_DIRECTORY'], exist_ok=True)
    print(f"Working Directory at: {input_variables['WORKING_DIRECTORY']}")
    os.makedirs(input_variables['OUTPUT_DIRECTORY'], exist_ok=True)
    print(f"Output Directory at: {input_variables['OUTPUT_DIRECTORY']}")

    input_variables['MOTION_THRESHOLD'] = float(input_variables['MOTION_THRESHOLD'])

    # copy config file 
    print(f"Copying Your Configuration File to the Output Directory: {input_variables['OUTPUT_DIRECTORY']}")
    shutil.copy(
        src=config_file,
        dst=os.path.join(input_variables['OUTPUT_DIRECTORY'], os.path.basename(config_file))
    )
    
    return input_variables
       

def run_main_pipeline(start_time, config_file, input_variables):
    
    print(f"\nStarting Pipeline on Configuration File: {config_file}") 
    
    input_nifti_path = input_variables['NIFTI_IMAGE_PATH']
    input_json_path = input_variables['JSON_FILE_PATH']

    # ------------------------------------------------------------------------------------------------------------
    #   1. DECOMPRESS AND DO DCM2NIIX (IF DICOM DIRECTORY IS GIVEN)
    # ------------------------------------------------------------------------------------------------------------
    if input_variables['DICOM_DIRECTORY_PATH']:
        nifti_maker = DicomToNiFTI(
            dicom_directory=input_variables['DICOM_DIRECTORY_PATH'],
            working_directory=input_variables['WORKING_DIRECTORY'],
            dcmdjpeg_path=input_variables['DCMDJPEG_PATH'],
            dcm2niix_path=input_variables['DCM2NIIX_PATH'],
            target_sequence_number=input_variables['SERIES_NUMBER'],
            series_name=input_variables['TASK_NAME']
        )

        input_nifti_path = nifti_maker.return_nifti_image()
        input_json_path = nifti_maker.return_json_file()

        print(f"Using created NiFTI Image: {input_nifti_path}")
        print(f"Using created JSON File: {input_json_path}")

    # ------------------------------------------------------------------------------------------------------------
    #   2. MOTION CHARACTERIZATION SCRIPT
    # ------------------------------------------------------------------------------------------------------------  
    MotionCharacterization(
        working_directory=input_variables['WORKING_DIRECTORY'],
        output_directory=input_variables['OUTPUT_DIRECTORY'],
        target_sequence_number=input_variables['SERIES_NUMBER'],
        sms_mi_reg_executable_path=input_variables['SMS_MI_REG_EXECUTABLE_PATH'],
        dcm2niix_path=input_variables['DCM2NIIX_PATH'],
        reference_volume_path=input_variables['REFERENCE_VOLUME_PATH'],
        reference_volume_index=input_variables['REFERENCE_VOLUME_INDEX'],
        nifti_path=input_nifti_path,
        json_path=input_json_path,
        reference_volume_script=input_variables['GET_REFERENCE_VOLUME_SCRIPT_PATH'],
        motion_threshold_percent=input_variables['MOTION_THRESHOLD']
    )

    motion_char_end_time = datetime.now()
    print(f'Motion Characterization End Time: {motion_char_end_time.strftime("%A %B %d %Y at %I:%M:%S %p")}')
    print(f'Total Motion Characterization Time: {round((motion_char_end_time - start_time).total_seconds() / 60, 2)} Minutes')


    # ------------------------------------------------------------------------------------------------------------
    #   3. GRAPH MOTION CHARACTERIZATION 
    # ------------------------------------------------------------------------------------------------------------
    GraphTransformDirectory(
        transform_directory=input_variables['WORKING_DIRECTORY'],
        json_path=input_json_path, 
        output_directory=input_variables['OUTPUT_DIRECTORY'],
        input_rotation_unit='versor',
        plot_tile=f"{os.path.basename(input_nifti_path)} Motion Characterization Plots", 
        threshold_as_percent=input_variables['MOTION_THRESHOLD'],
        transform_suffix = ".tfm"
    )

    # ------------------------------------------------------------------------------------------------------------
    #   5. BACKGROUND REMOVAL SCRIPT
    # ------------------------------------------------------------------------------------------------------------
    
    bgremoved_output_nifti_path = os.path.join(input_variables['OUTPUT_DIRECTORY'], f"{os.path.basename(input_nifti_path).replace('.nii.gz', '')}_bgremoved.nii.gz")

    RemoveBackground(
        nifti_file_path=input_nifti_path,
        output_path=bgremoved_output_nifti_path
    )

    bg_remove_end_time = datetime.now()
    print(f'Background Removal End Time: {bg_remove_end_time.strftime("%A %B %d %Y at %I:%M:%S %p")}')
    print(f'Total Background Removal Time: {round((bg_remove_end_time - motion_char_end_time).total_seconds() / 60, 2)} Minutes')


    # ------------------------------------------------------------------------------------------------------------
    #   5. MOTION CORRECTION SCRIPT
    # ------------------------------------------------------------------------------------------------------------
    StartMotionCorrection(
        motion_correction_main_script_dir=input_variables['MCORR_MAIN_SCRIPT_DIR'],
        motion_correction_main_function_name=os.path.basename(input_variables['MCORR_MAIN_SCRIPT'].replace('.m', '')),
        output_dir=input_variables['OUTPUT_DIRECTORY'],
        input_nifti_image_path=input_nifti_path,
        input_json_path=input_json_path,
        bgremoved_input_nifti_image_path=bgremoved_output_nifti_path,
        parameters_textfile_path=os.path.join(input_variables['OUTPUT_DIRECTORY'], "radian_parameters.txt"),
        recon_abruptmotion_output_filepath=os.path.join(input_variables['OUTPUT_DIRECTORY'], "recon_abruptmotion.nii.gz"),
        directliftandunliftcodes_dir=input_variables['MCORR_DIRECT_LIFTANDUNLIFT_CODES_DIR'],
        operators_dir=input_variables['MCORR_OPERATORS_DIR'],
        scrubbing_threshold=input_variables['MOTION_THRESHOLD'],
        matlab_installation_path=input_variables['MATLAB_INSTALLATION_PATH']
    )

    motion_correction_end_time = datetime.now()
    print(f'Motion Characterization End Time: {motion_correction_end_time.strftime("%A %B %d %Y at %I:%M:%S %p")}')
    print(f'Total Motion Characterization Time: {round((motion_correction_end_time - bg_remove_end_time).total_seconds() / 60, 2)} Minutes')


# 1. GET COMMAND LINE ARGUMENTS 
parser = argparse.ArgumentParser(
    description="This script automates the motion pipeline by calling the various software of the pipeline in order, one after the other."
)
parser.add_argument(
    "--configuration_files", 
    required=True, 
    nargs='+',
    help="The config.env file(s) containing your configurations."
)
parser.add_argument(
    "--serialize",
    action='store_true', 
    help="Flag if you dont want to run the config.env files in parallel."
)
args = parser.parse_args()

# 2. PARSE CONFIGURATION FILES AND VALIDATE VARIABLES
all_input_variables = parse_config_files(configuration_files=args.configuration_files)

# 3. RUN MAIN PIPELINE
start_time = datetime.now()
print(f'Pipeline Start Time: {start_time.strftime("%A %B %d %Y at %I:%M:%S %p")}')

n_jobs = 1 if args.serialize else min(len(args.configuration_files), 2)
print(f"Running Main Pipeline with {n_jobs} Worker(s)")
Parallel(n_jobs=n_jobs, verbose=11, backend='loky', max_nbytes='1G')(
    delayed(run_main_pipeline)(
        start_time=start_time,
        config_file=config_file_path,
        input_variables=input_variables
    ) for config_file_path, input_variables in all_input_variables.items()
)

# 4. RUN FMRIPREP
if any(input_variables['RUN_FMRIPREP'].lower() == 'true' for input_variables in all_input_variables.values()):
    RunfMRIPrep(configuration_files=list(all_input_variables.keys()))
   

print(f"\nPipeline is Done.")
end_time = datetime.now()
print(f'Start Time: {start_time.strftime("%A %B %d %Y at %I:%M:%S %p")}')
print(f'End Time: {end_time.strftime("%A %B %d %Y at %I:%M:%S %p")}')
print(f'Total Time: {round((end_time - start_time).total_seconds() / 60, 2)} Minutes (or {round(((end_time - start_time).total_seconds() / 60) / 60, 2)} Hours)')
