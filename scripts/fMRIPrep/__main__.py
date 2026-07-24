from datetime import datetime
from glob import glob
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import functions

class RunfMRIPrep:
    
    def __init__(self, configuration_files):
        
        if isinstance(configuration_files, str):
            configuration_files = [configuration_files]

        print(f"RunfMRIPrep Received the Following Configuration Files:") 
        print('\n'.join(configuration_files))
        
        across_configuration_file_vars = {} # will be extracted from the first config file
        participants = []
        fmriprep_pipeline_outputdir_pairs = []
        for configuration_file_num, configuration_file in enumerate(configuration_files):
            print(f"\nProcessing Configuration File: {configuration_file}")
            
            input_variables = functions.get_input_variables(config_file=configuration_file)

            for parent_directory in ('FMRIPREP_INPUT_DIRECTORY', 'FMRIPREP_OUTPUT_DIRECTORY', 'FMRIPREP_TMP_DIRECTORY', 'FMRIPREP_WORKING_DIRECTORY'):
                os.makedirs(input_variables[parent_directory], exist_ok=True) 
                print(f"{parent_directory}: {input_variables[parent_directory]}")
            
            # self.clear_directory(input_variables['FMRIPREP_WORKING_DIRECTORY'])

            if configuration_file_num == 0:
                across_configuration_file_vars = {
                    'FMRIPREP_CONTAINER_PATH':input_variables['FMRIPREP_CONTAINER_PATH'],
                    'FMRIPREP_TEMPLATEFLOW_DIRECTORY':input_variables['FMRIPREP_TEMPLATEFLOW_DIRECTORY'],
                    'FMRIPREP_LICENSE_PATH':input_variables['FMRIPREP_LICENSE_PATH'],
                    'FMRIPREP_INPUT_DIRECTORY':input_variables['FMRIPREP_INPUT_DIRECTORY'],
                    'FMRIPREP_DATASET_JSON_PATH':input_variables['FMRIPREP_DATASET_JSON_PATH'],
                    'FMRIPREP_OUTPUT_DIRECTORY':input_variables['FMRIPREP_OUTPUT_DIRECTORY'],
                    'FMRIPREP_WORKING_DIRECTORY':input_variables['FMRIPREP_WORKING_DIRECTORY']
                }
        
            subject_id_str = f"sub-{'{:02d}'.format(int(input_variables['FMRIPREP_PARTICIPANT_ID']))}"
            print(f"Formatted Subject ID: {subject_id_str}")
            if not subject_id_str.replace('sub-', '') in participants:
                participants.append(subject_id_str.replace('sub-', ''))

            session_id_str = f"ses-{'{:02d}'.format(int(input_variables['FMRIPREP_SESSION_ID']))}"
            print(f"Formatted Session ID: {session_id_str}")

            run_num_str = None
            if input_variables['FMRIPREP_RUN_NUM']:
                run_num_str = f"run-{'{:02d}'.format(int(input_variables['FMRIPREP_RUN_NUM']))}"
                print(f"Formatted Run Number: {run_num_str}")
            
            subject_directory = os.path.join(input_variables['FMRIPREP_INPUT_DIRECTORY'], subject_id_str)
            os.makedirs(subject_directory, exist_ok=True)
            print(f"Subject Directory: {subject_directory}")

            session_directory = os.path.join(subject_directory, f"{session_id_str}")
            os.makedirs(session_directory, exist_ok=True)
            print(f"Session Directory: {session_directory}")
            
            tmp_directory = os.path.join(input_variables['FMRIPREP_TMP_DIRECTORY'], f"{subject_id_str}_{session_id_str}{f'_{run_num_str}' if run_num_str else ''}") 
            os.makedirs(tmp_directory, exist_ok=True)
            print(f"Temporary Directory: {tmp_directory}")

            functional_directory = os.path.join(session_directory, "func")
            os.makedirs(functional_directory, exist_ok=True)
            print(f"Functional Directory: {functional_directory}")

            anatomical_directory = os.path.join(session_directory, "anat")
            os.makedirs(anatomical_directory, exist_ok=True)
            print(f"Anatomical Directory: {anatomical_directory}")

            if not os.path.exists(input_variables['FMRIPREP_DATASET_JSON_PATH']):
                self.make_dataset_json(input_variables['FMRIPREP_DATASET_JSON_PATH'])
            
            self.fill_anat_directory(
                input_variables=input_variables,
                anatomical_directory=anatomical_directory,
                tmp_directory=tmp_directory,
                subject_id_str=subject_id_str,
                session_id_str=session_id_str
            )
            self.fill_func_directory(
                input_variables=input_variables,
                functional_directory=functional_directory,
                tmp_directory=tmp_directory,
                subject_id_str=subject_id_str,
                session_id_str=session_id_str,
                run_num_str=run_num_str
            )
            
            print(f"\nBIDS Folder Layout Complete.")
            print(f"Input Data Tree:")
            subprocess.run(['tree', subject_directory])
            print('\n')

            fmriprep_pipeline_outputdir_pairs.append((
                os.path.join(input_variables['FMRIPREP_OUTPUT_DIRECTORY'], subject_id_str),
                os.path.join(input_variables['OUTPUT_DIRECTORY'], f'{subject_id_str}_fmriprep_outputs-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
            ))

            fs_lock_dir = os.path.join(
                subject_directory,
                "sourcedata",
                "freesurfer",
                subject_id_str,
                "scripts"
            )
            if os.path.exists(fs_lock_dir):
                self.clean_freesurfer_lock_files(fs_lock_dir)


        for i, participant in enumerate(participants, start=1):
            
            working_directory = os.path.join(
                    across_configuration_file_vars['FMRIPREP_WORKING_DIRECTORY'],
                    f'sub-{str(participant)}'
            )
            os.makedirs(working_directory, exist_ok=True)

            print(f"Running fMRIPrep on Participant {i} of {len(participants)}")
            self.run_fmriprep(
                participant=participant,
                input_data_directory=across_configuration_file_vars['FMRIPREP_INPUT_DIRECTORY'],
                templateflow_directory=across_configuration_file_vars['FMRIPREP_TEMPLATEFLOW_DIRECTORY'],
                output_directory=across_configuration_file_vars['FMRIPREP_OUTPUT_DIRECTORY'],
                working_directory=working_directory,
                license_path=across_configuration_file_vars['FMRIPREP_LICENSE_PATH'],
                fmriprep_container_path=across_configuration_file_vars['FMRIPREP_CONTAINER_PATH']
            )
        
        for fmriprep_dir, pipeline_dir in fmriprep_pipeline_outputdir_pairs:
            print(f"Copying fMRIPrep Output Directory: {fmriprep_dir} to Pipeline Output Directory: {pipeline_dir}")
            shutil.copytree(
                src=fmriprep_dir,
                dst=pipeline_dir
            )
    def clear_directory(self, directory_path):
        
        if os.path.exists(directory_path):
            print(f"Clearing Out: {directory_path}")
            shutil.rmtree(directory_path)

        print(f"Created: {directory_path}")
        os.makedirs(directory_path)
        

    def make_dataset_json(self, json_path):
        
        print(f"Making File: {json_path}")
        with open(file=json_path, mode='w') as f:
            json.dump(
                obj={
                    "Name": "fMRIPrep - fMRI PREProcessing workflow",
                    "BIDSVersion": "1.4.0",
                    "DatasetType": "derivative",
                    "GeneratedBy": [
                        {
                            "Name": "fMRIPrep",
                            "Version": "23.1.3",
                            "CodeURL": "https://github.com/nipreps/fmriprep/archive/23.1.3.tar.gz"
                        }
                    ],
                    "HowToAcknowledge": "Please cite our paper (https://doi.org/10.1038/s41592-018-0235-4), and include the generated citation boilerplate within the Methods section of the text."
                },
                fp=f
            )

    
    def fill_anat_directory(self, input_variables, anatomical_directory, tmp_directory, subject_id_str, session_id_str):
        
        anatomical_data = []
        if input_variables['INPUT_ANAT_DICOM_DIRECTORY']:
            
            print(f"Using Anatomical DICOM Directory: {input_variables['INPUT_ANAT_DICOM_DIRECTORY']}")

            functions.decompress_dicoms(
                dcmdjpeg_path=input_variables['DCMDJPEG_PATH'],
                dicom_directory=input_variables['INPUT_ANAT_DICOM_DIRECTORY'],
                output_directory=tmp_directory
            ) 

            functions.dicom_to_nifti(
                dcm2niix_path=input_variables['DCM2NIIX_PATH'],
                working_directory=tmp_directory,
                dicom_directory=tmp_directory,
                return_one_result=False
            )
            anatomical_data = [
                functions.search_for_file(
                    parent_directory=tmp_directory, 
                    filename_pattern=f"*{file_type}*{file_extension}", 
                    return_one_result=True,
                    not_exist_ok=True if file_type == "T2w" else False # we dont need t2w data
                )
                for file_type, file_extension in (("T1w", ".nii.gz"),  ("T1w", ".json"), ("T2w", ".nii.gz"), ("T2w", ".json"))
            ]
        
        else:
            anatomical_data = [
                input_variables['T1W_IMAGE_PATH'], 
                input_variables['T1W_JSON_PATH'],
                input_variables['T2W_IMAGE_PATH'],
                input_variables['T2W_JSON_PATH']
            ]
            
        anatomical_data = [os.path.abspath(file) for file in anatomical_data if file is not None and file.strip()]
        for file_path in anatomical_data:
            
            bids_filename = self.bids_rename_file(
                file_path=file_path,
                subject_id_str=subject_id_str,
                session_id_str=session_id_str,
                is_anat_image=True
            )
            
            try:
                print(f"Copying {os.path.basename(file_path)} to {anatomical_directory} with new filename: {bids_filename}")
                shutil.copy(
                    src=file_path,
                    dst=os.path.join(anatomical_directory, bids_filename)
                )
            except shutil.SameFileError:
                print(f"File already exists in {anatomical_directory}.")

    
    def fill_func_directory(self, input_variables, functional_directory, tmp_directory, subject_id_str, session_id_str, run_num_str = None):
        
        nifti_image, json_file, task_name = "", "", ""
        
        # Get nifti image 
        if input_variables['FMRIPREP_FUNC_IMAGE_PATH']:
            nifti_image = os.path.abspath(input_variables['FMRIPREP_FUNC_IMAGE_PATH'])
        else:
            nifti_image = functions.search_for_file(
                parent_directory=input_variables['OUTPUT_DIRECTORY'],
                filename_pattern=input_variables['MCORR_OUTPUT_FILENAME_PATTERN'],
                return_one_result=True,
                not_exist_ok=True
            )

            # if no file matches MCORR_OUTPUT_FILENAME_PATTERN 
            if not nifti_image:
                print(
                    f"\n\nIMPORTANT: No file matching {os.path.join(input_variables['OUTPUT_DIRECTORY'], input_variables['MCORR_OUTPUT_FILENAME_PATTERN'])} exists." \
                    "\nIf the inputted raw data has little or no motion, this is to be expected." \
                    "\nIf not, something has gone wrong." \
                    "\nWe will search for and input the raw, orignal data to fmriprep.\n\n"
                )  
                nifti_image = self.find_raw_data(input_variables)
                print(f"Running the Following NiFTI Image Through fMRIPrep: {nifti_image}\n\n") 

        
        # Get json image
        if input_variables['FMRIPREP_FUNC_JSON_PATH']:
            json_file = os.path.abspath(input_variables['FMRIPREP_FUNC_JSON_PATH'])
        else:
            json_file = functions.search_for_file(
                parent_directory=input_variables['OUTPUT_DIRECTORY'],
                filename_pattern="*json",
                return_one_result=True)

        # get task name
        if input_variables['TASK_NAME']:
            task_name = input_variables['TASK_NAME']
        else:
            # get from ProtocolName key in json file 
            with open(json_file, mode='r') as file:
                task_name = json.load(file)['ProtocolName'].replace('func-bold_task-', '')
        
        for file_path in (nifti_image, json_file):
            
            bids_filename = self.bids_rename_file(
                file_path=file_path,
                subject_id_str=subject_id_str,
                session_id_str=session_id_str,
                task_name=task_name,
                run_num_str=run_num_str
            )
            
            try:
                print(f"Copying {os.path.basename(file_path)} to {functional_directory} with new filename: {bids_filename}")
                shutil.copy(
                    src=file_path,
                    dst=os.path.join(functional_directory, bids_filename)
                )
            except shutil.SameFileError:
                print(f"File already exists in {functional_directory}.")


    def find_raw_data(self, input_variables):
             
        def exit_with_no_results():
            print(
                f"Script could not determine what nifti image to use for fmriprep." \
                "\nTry entering values for FMRIPREP_FUNC_IMAGE_PATH and FMRIPREP_FUNC_JSON_PATH," \
                "\nas we couldn't get the input data for fmriprep automatically."
            )
            sys.exit(1)

        found_nifti_images = glob(os.path.join(input_variables['OUTPUT_DIRECTORY'], "*nii.gz"))
        if not found_nifti_images:
            found_nifti_images = glob(os.path.join(input_variables['OUTPUT_DIRECTORY'], "*.nii"))
            if not found_nifti_images:
                exit_with_no_results()
        
        matching_nifti_images = [
            found_nifti_image
            for found_nifti_image in found_nifti_images
            if not "_bgremoved.nii" in os.path.basename(found_nifti_image) and \
            not input_variables['MCORR_ABRUPTMOTION_FILE_NAME'] ==  os.path.basename(found_nifti_image)
        ]
                
        if len(matching_nifti_images) == 1:
            return matching_nifti_images[0]
        
        elif len(matching_nifti_images) > 1:
            print(
                f"\n\nWARNING: More than one matching NiFTI File Found:" \
                "\n" + "\n".join(matching_nifti_images) + "\n" \
                "Returning last matching file."
            )
            return matching_nifti_images[-1]
        
        else:
            exit_with_no_results()
                

    def bids_rename_file(self, 
                         file_path,
                         subject_id_str, 
                         session_id_str,
                         is_anat_image=False,
                         task_name=None,
                         run_num_str=None):
        
        # BIDS Format: sub-<label>[_ses-<label>]_task-<label>[_acq-<label>][_run-<index>]_<suffix>.<extension>
        
        def get_anat_file_type(filepath):
            if "t1w" in os.path.basename(filepath).lower():
                return 'T1w'
            elif 't2w'in os.path.basename(filepath).lower():
                return 'T2w'
            else:
                print(f"\n\nERROR: Cannot determine if {filepath} is a T1w image/json or T2w image/json.") 
                print("The image type should be in the file names")
                sys.exit(1)
        

        def get_file_extension(filepath):
            if '.nii' in os.path.basename(filepath):
                if os.path.basename(filepath).endswith('.nii.gz'):
                    return '.nii.gz'
                else:
                    return '.nii'
            elif os.path.basename(filepath).endswith('.json'):
                return '.json'
            else:
                print(f"\n\nERROR: File {filepath} must end in '.json', '.nii.gz', or '.nii'") 
                sys.exit(1)
        

        filename_prefix = f"{subject_id_str}_{session_id_str}"
        if is_anat_image:
            filename_prefix += "_" + get_anat_file_type(file_path)
        
        else:
            if not task_name:
                print(f"\n\nERROR: Functional data must have an inputted task_name.")
                sys.exit(1)
            
            if run_num_str:
                filename_prefix += f"_task-{task_name}_{run_num_str}_bold"
            else:
                filename_prefix += f"_task-{task_name}_bold"

        filename_prefix += get_file_extension(file_path)
        
        return filename_prefix

    
    def clean_freesurfer_lock_files(self, freesurfer_subject_dir):
        lock_files = list(Path(freesurfer_subject_dir).rglob("IsRunning*"))
        for f in lock_files:
            try:
                print(f"Removing lock file: {f}")
                f.unlink()
            except Exception as e:
                print(e)
                
   
    def run_fmriprep(self,
                     participant,
                     input_data_directory, 
                     templateflow_directory, 
                     output_directory, 
                     working_directory,
                     license_path,
                     fmriprep_container_path, 
                     nprocs = 8,
                     omp_nthreads = 4,
                     mem = 48000):

        command = [
            "singularity", "run", "--cleanenv", \
            "-B", f"{templateflow_directory}:/templateflow",
            "-B", f"{input_data_directory}:/bids_dir",
            "-B", f"{working_directory}:/work_dir",
            "-B", f"{output_directory}:/out_dir",
            "-B", f"{license_path}:/license.txt",
            fmriprep_container_path,
            "/bids_dir", 
            "/out_dir", 
            "participant", "--participant-label", participant,
            "--fs-license-file", "/license.txt",
            "--nprocs", str(nprocs),
            "--omp-nthreads", str(omp_nthreads),
            "--mem", str(mem),
            "-w", "/work_dir",
            "--skip_bids_validation",
            "--output-spaces", "MNI152NLin2009cAsym:res-2",
        ] 
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print("fMRIPrep failed:")
            print(result.stdout)
            print(result.stderr)

        subprocess.run(command)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Run fMRIPrep"
    )
    parser.add_argument(
        "--configuration_files",
        required=True,
        nargs="+"
    )
    args = parser.parse_args()

    RunfMRIPrep(
        configuration_files=[
            os.path.abspath(config_file)
            for config_file in args.configuration_files
        ]
    ) 
