import os 
import sys
import yaml
import subprocess
from glob import glob 

# 
# Quick script for doing just fMRIPrep on the raw data 
# from each config file in the config_files/ directory
# 

project_dir = "/lab-share/Neuro-Cohen-e2/Groups/IRB-P00049401/intravolume_motion_pipeline/"
config_dir = os.path.join(project_dir, "run_pipeline/config_files")
fmriprep_script_path = os.path.join(project_dir, "scripts/SingleRunfMRIPrep/__main__.py")
slurm_script_path = os.path.join(project_dir, "run_pipeline/submit_command.sh")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from manage_configuration_files import Configurations

config_files = sorted(glob(os.path.join(config_dir, "sub-*.yaml")))
print(f"I Found {len(config_files)} Configuration Files:")
print('\n'.join(config_files))

for config_file in config_files:

    print(f"\nLoading Input Configuration File: {config_file}")
    configurations = None
    with open(config_file, mode='r') as file:
        configurations = Configurations(**yaml.safe_load(file))

    os.makedirs(configurations.WORKING_DIRECTORY_PATH, exist_ok=True)
    os.makedirs(configurations.OUTPUT_DIRECTORY_PATH, exist_ok=True)

    func_data = \
        configurations.FUNCTIONAL_DICOM_DIRECTORY \
        if configurations.FUNCTIONAL_DICOM_DIRECTORY \
        else f"{configurations.FUNCTIONAL_NIFTI_IMAGE_PATH} {configurations.FUNCTIONAL_JSON_FILE_PATH}" 
    anat_data = \
        configurations.ANATOMICAL_DICOM_DIRECTORY \
        if configurations.ANATOMICAL_DICOM_DIRECTORY \
        else f"{configurations.ANATOMICAL_NIFTI_IMAGE_PATH} {configurations.ANATOMICAL_JSON_FILE_PATH}" 
    
    python_command = \
        f"python {fmriprep_script_path} " + \
        f"--func_data {func_data} " + \
        f"--anat_data {anat_data} " + \
        f"--series_name {configurations.SERIES_NAME} " + \
        f"--subject_id {str(configurations.SUBJECT_ID)} " + \
        f"--session_num {configurations.SESSION_NUM} " + \
        f"--run_num {str(configurations.RUN_NUM)} " + \
        f"--working_directory {os.path.join(configurations.WORKING_DIRECTORY_PATH, 'just_fmriprep-working')} " + \
        f"--output_directory {os.path.join(configurations.OUTPUT_DIRECTORY_PATH, 'just_fmriprep-outputs')} " + \
        f"--n_jobs {str(configurations.N_JOBS)}"

    full_command = ["sbatch", slurm_script_path, config_file, python_command]

    print("Running Command: ")
    print(" ".join(full_command))

    subprocess.run(full_command)


