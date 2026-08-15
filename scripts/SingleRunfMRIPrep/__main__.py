import os 
import sys
import json
import shutil
import warnings
import subprocess
from glob import glob
from typing import Any
import SimpleITK as sitk 
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# in order of usage:
from decompress_dicoms import DecompressDicoms
from dicom_to_nifti import DicomToNifti

class StartSingleRunfMRIPrep:
    def __init__(self,
                 func_data: list[str],
                 anat_data: list[str],
                 series_name: str,
                 subject_id: int,  
                 session_num: int,  
                 run_num: int,      
                 working_directory: str = 'working',
                 output_directory: str = 'outputs',
                 dcmdjpeg_path: str = 'dcmdjpeg',
                 dcm2niix_path: str = 'dcm2niix',
                 FMRIPREP_CONTAINER_PATH: str | None = None,
                 FMRIPREP_TEMPLATEFLOW_DIRECTORY: str | None = None,
                 n_jobs: int | None = os.cpu_count(), # pyright: ignore[reportRedeclaration]
                 omp_nthreads: int = 8,
                 mem_mb: int = 24000  # in MB
                 ) -> None: 

        # Set up input configurations 
        if n_jobs == None:
            n_jobs: int = 8

        if n_jobs < 0:
            new_n_jobs: int = os.cpu_count() + (n_jobs  + 1) # pyright: ignore[reportOptionalOperand]
            if new_n_jobs < 0:
                raise ValueError(f"n_jobs value: {n_jobs} is not a valid number of jobs.")
            n_jobs: int = new_n_jobs

        print(f"fMRIPrep Max Number of Processes: {n_jobs}")
        print(f"fMRIPrep Max Number of Threads: {omp_nthreads}")
        print(f"fMRIPrep Max Memory: {mem_mb} MB / {round(mem_mb / 1000, 2)} GB")

        subject_id_str: str = 'sub-' + '{:02d}'.format(subject_id)
        session_num_str: str = 'ses-' + '{:02d}'.format(session_num)
        run_num_str: str = 'run-' + '{:02d}'.format(run_num)

        print(f"Subject: {subject_id_str}")
        print(f"Session: {session_num_str}")
        print(f"Run: {run_num_str}")

        # Set up input paths
        self.FMRIPREP_CONTAINER_PATH: str | None = FMRIPREP_CONTAINER_PATH # pyright: ignore[reportAttributeAccessIssue, reportRedeclaration]
        if self.FMRIPREP_CONTAINER_PATH == None:
            self.FMRIPREP_CONTAINER_PATH: str = "/lab-share/Neuro-Cohen-e2/Public/containers/fmriprep-23.1.3.simg"
        print(f"fMRIPrep Singularity Container Path: {self.FMRIPREP_CONTAINER_PATH}")

        self.FMRIPREP_TEMPLATEFLOW_DIRECTORY: str | None = FMRIPREP_TEMPLATEFLOW_DIRECTORY # pyright: ignore[reportAttributeAccessIssue, reportRedeclaration]
        if self.FMRIPREP_TEMPLATEFLOW_DIRECTORY == None:
            self.FMRIPREP_TEMPLATEFLOW_DIRECTORY: str = "/lab-share/Neuro-Cohen-e2/Public/templateflow/"
        print(f"fMRIPrep Templateflow Directory Path: {self.FMRIPREP_TEMPLATEFLOW_DIRECTORY}")

        self.dcmdjpeg_path: str = dcmdjpeg_path
        print(f"dcmdjpeg Path: {self.dcmdjpeg_path}")

        self.dcm2niix_path: str = dcm2niix_path
        print(f"dcm2niix Path: {self.dcm2niix_path}")

        os.makedirs(working_directory, exist_ok=True)
        print(f"Working Directory: {working_directory}")

        os.makedirs(output_directory, exist_ok=True)
        print(f"Output Directory: {output_directory}")

        fmriprep_output_directory: str = os.path.join(output_directory, f'fmriprep_output_directory_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
        os.makedirs(fmriprep_output_directory, exist_ok=True)
        print(f"fMRIPrep Output Directory: {fmriprep_output_directory}")

        fmriprep_main_directory: str = os.path.join(working_directory, f'fmriprep_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
        os.makedirs(fmriprep_main_directory, exist_ok=True)
        print(f"fMRIPrep Main Directory: {fmriprep_main_directory}")
        
        fmriprep_working_directory: str = os.path.join(fmriprep_main_directory, 'fmriprep_working_directory')
        os.makedirs(fmriprep_working_directory, exist_ok=True)
        print(f"fMRIPrep Working Directory: {fmriprep_working_directory}")

        fmriprep_input_directory: str = os.path.join(fmriprep_main_directory, "input_directory")
        os.makedirs(fmriprep_input_directory, exist_ok=True)
        print(f"fMRIPrep Input Directory: {fmriprep_input_directory}")

        subject_directory: str = os.path.join(fmriprep_input_directory, subject_id_str)
        os.makedirs(subject_directory, exist_ok=True)
        print(f"Subject Directory: {subject_directory}")

        session_directory: str = os.path.join(subject_directory, session_num_str)
        os.makedirs(session_directory, exist_ok=True)
        print(f"Session Directory: {session_directory}")

        functional_directory = os.path.join(session_directory, "func")
        os.makedirs(functional_directory, exist_ok=True)
        print(f"Functional Directory: {functional_directory}")

        anatomical_directory = os.path.join(session_directory, "anat")
        os.makedirs(anatomical_directory, exist_ok=True)
        print(f"Anatomical Directory: {anatomical_directory}")

        tmp_directory = os.path.join(fmriprep_main_directory, "fmriprep_tmp") 
        os.makedirs(tmp_directory, exist_ok=True)
        print(f"Temporary Directory: {tmp_directory}")

        # Copy fmriprep regulatory files
        fmriprep_script_directory = os.path.dirname(__file__)
        self.dataset_description_json: str = os.path.join(fmriprep_script_directory, "dataset_description.json")
        if not os.path.exists(self.dataset_description_json):
            raise FileNotFoundError(
                f"Could not find fMRIPrep Dataset Description JSON File at: {self.dataset_description_json}"
            )
        else:
            new_dataset_description_json: str = os.path.join(fmriprep_input_directory, os.path.basename(self.dataset_description_json))
            print(f"Copying: {self.dataset_description_json} to: {new_dataset_description_json}")
            shutil.copy(
                src=self.dataset_description_json,
                dst=new_dataset_description_json
            )
            self.dataset_description_json: str = new_dataset_description_json        
        self.license_path: str = os.path.join(fmriprep_script_directory, "license.txt")
        if not os.path.exists(self.license_path):
            raise FileNotFoundError(
                f"Could not find fMRIPrep License Path at: {self.license_path}"
            )
        else:
            new_license_path: str = os.path.join(fmriprep_input_directory, os.path.basename(self.license_path))
            print(f"Copying: {self.license_path} to: {new_license_path}")
            shutil.copy(
                src=self.license_path,
                dst=new_license_path
            )
            self.license_path: str = new_license_path

        # Get input data 
        func_nifti_image_path, \
        func_json_file_path, \
        anat_t1w_nifti_image_path,\
        anat_t1w_json_file_path, \
        anat_t2w_nifti_image_path, \
        anat_t2w_json_file_path = self.get_input_images(
            func_data, anat_data, working_directory=working_directory, series_name=series_name, n_jobs=n_jobs
        )

        # Convert Functional Data to BIDs
        for func_file_path in (func_nifti_image_path, func_json_file_path):
            
            bids_func_filename: str = self.bids_rename_file(
                func_file_path,
                subject_id_str=subject_id_str,
                session_id_str=session_num_str,
                run_num_str=run_num_str,
                series_name=series_name,
            )
            print(f"Renaming {os.path.basename(func_file_path)} to {bids_func_filename}")

            bids_func_file_path: str = os.path.join(functional_directory, bids_func_filename)
            if os.path.exists(bids_func_file_path):
                warnings.warn(
                    message=\
                        f"BIDs File: {bids_func_file_path} already exists. " + \
                        "We are going to overwrite it.",
                    category=UserWarning
                )
                os.remove(bids_func_file_path)
            print(f"Copying File: {func_file_path} to {bids_func_file_path}")
            shutil.copy(
                src=func_file_path,
                dst=bids_func_file_path
            )

        # Convert Anatomical Data to BIDs
        for anat_file_path in (anat_t1w_nifti_image_path, anat_t1w_json_file_path, 
                               anat_t2w_nifti_image_path, anat_t2w_json_file_path):

            if not anat_file_path:
                continue 

            bids_anat_filename: str = self.bids_rename_file(
                file_path=anat_file_path,
                subject_id_str=subject_id_str,
                session_id_str=session_num_str,
                run_num_str=run_num_str,
                is_anat_image=True
            )
            print(f"Renaming {os.path.basename(anat_file_path)} to {bids_anat_filename}")

            bids_anat_file_path: str = os.path.join(anatomical_directory, bids_anat_filename)
            if os.path.exists(bids_anat_file_path):
                warnings.warn(
                    message=\
                        f"BIDs File: {bids_anat_file_path} already exists. " + \
                        "We are going to overwrite it.",
                    category=UserWarning
                )
                os.remove(bids_anat_file_path)
            print(f"Copying File: {anat_file_path} to {bids_anat_file_path}")
            shutil.copy(
                src=anat_file_path,
                dst=bids_anat_file_path
            )

        # Start fMRIPRep
        print(f"Starting fMRIPrep")
        fmriprep_command: list[str] = [
            "singularity", "run", "--cleanenv", \
            "-B", f"{self.FMRIPREP_TEMPLATEFLOW_DIRECTORY}:/templateflow",
            "-B", f"{fmriprep_input_directory}:/bids_dir",
            "-B", f"{fmriprep_working_directory}:/work_dir",
            "-B", f"{fmriprep_output_directory}:/out_dir",
            "-B", f"{self.license_path}:/license.txt",
            self.FMRIPREP_CONTAINER_PATH,
            "/bids_dir", 
            "/out_dir", 
            "participant", "--participant-label", '{:02d}'.format(subject_id),
            "--fs-license-file", "/license.txt",
            "--nprocs", str(n_jobs),
            "--omp-nthreads", str(omp_nthreads),
            "--mem", str(mem_mb),
            "-w", "/work_dir",
            "--skip_bids_validation",
            "--output-spaces", "MNI152NLin2009cAsym:res-2", "func"
        ] 
        print(f"Running Command: {fmriprep_command}")
        result: subprocess.CompletedProcess = subprocess.run(fmriprep_command)
        if result.returncode != 0:
            raise CalledProcessError(
                returncode=result.returncode,
                cmd=result.args,
                output=result.stdout,
                stderr=result.stderr
            )

    def get_input_images(self, 
                         func_data: str | list[str],
                         anat_data: str | list[str],
                         working_directory: str, 
                         series_name: str,
                         n_jobs: int = -1
                         ) -> tuple[
                             str, str,
                             str, str,
                             str | None, str | None
                         ]:
        """
        =======================================================
        GET FUNCTIONAL NIFTI IMAGE
        =======================================================
        """
        func_nifti_image_path: str = ""
        func_json_file_path: str = ""
        if len(func_data) == 1 and os.path.isdir(func_data[0]):
            """
            =======================================================
            DECOMPRESS DICOMS 
            =======================================================
            """
            decompression_directory: str = os.path.join(working_directory, "func_decompressed_dicoms")
            print(f"Decompressing DICOMs into directory: {decompression_directory}")
            DecompressDicoms(
                dicom_directory=func_data[0],
                output_directory=decompression_directory,
                dcmdjpeg_path=self.dcmdjpeg_path,
                series_name=series_name,
                n_jobs=n_jobs
            )
    
            """
            =======================================================
            DO DICOM TO NIFTI
            =======================================================
            """
            dcm2niix_module: DicomToNifti = DicomToNifti(
                dicom_directory=decompression_directory,
                output_directory=working_directory,
                dcm2niix_path=self.dcm2niix_path,
            )
            func_nifti_image_path: str = dcm2niix_module.return_nifti_image() # pyright: ignore[reportAssignmentType]
            func_json_file_path: str = dcm2niix_module.return_json_file() # pyright: ignore[reportAssignmentType]
    
            dimensions: tuple[int, int, int, int] = sitk.ReadImage(func_nifti_image_path).GetSize()
            print(f"NiFTI Image Dimensions: {dimensions}")
            if len(dimensions) != 4:
                raise ValueError("Input data must be 4D.")
    
            
        elif len(func_data) == 2:
            for file in func_data:
                if file.endswith(".nii.gz") or file.endswith(".nii"):
                    func_nifti_image_path: str = file
                elif file.endswith(".json"):
                    func_json_file_path: str = file 
                else:
                    raise ValueError(
                        "For argument 'func_data', enter the path to the functional DICOM Directory "
                        "or give the paths to both the functional NiFTI image and JSON file. " 
                        "len(func_data) should be 1 or 2."
                    )

        else:
            raise ValueError(
                "For argument 'func_data', enter the path to the functional DICOM Directory "
                "or give the paths to both the functional NiFTI image and JSON file." 
                "len(func_data) should be 1 or 2."
            )

        if not func_nifti_image_path or not func_json_file_path:
            raise ValueError(
                "For argument 'func_data', enter the path to the functional DICOM Directory "
                "or give the paths to both the functional NiFTI image and JSON file." 
                "len(func_data) should be 1 or 2."
            )

        print(f"Functional NiFTI Image: {func_nifti_image_path}")
        print(f"Functional JSON Path: {func_json_file_path}")


        """
        =======================================================
        GET ANATOMICAL NIFTI IMAGE(S)
        =======================================================
        """
        anat_t1w_nifti_image_path: str = ""
        anat_t1w_json_file_path: str = ""
        anat_t2w_nifti_image_path: str | None = None
        anat_t2w_json_file_path: str | None = None
        if len(anat_data) == 1 and os.path.isdir(anat_data[0]):
            """
            =======================================================
            DECOMPRESS DICOMS 
            =======================================================
            """
            decompression_directory: str = os.path.join(working_directory, "anat_decompressed_dicoms")
            print(f"Decompressing DICOMs into directory: {decompression_directory}")
            DecompressDicoms(
                dicom_directory=anat_data[0],
                output_directory=decompression_directory,
                dcmdjpeg_path=self.dcmdjpeg_path
            )
    
            """
            =======================================================
            DO DICOM TO NIFTI
            =======================================================
            """
            dcm2niix_module: DicomToNifti = DicomToNifti(
                dicom_directory=decompression_directory,
                output_directory=working_directory,
                dcm2niix_path=self.dcm2niix_path,
                return_multiple_ok=True
            )
            anat_t1w_nifti_image_path, anat_t1w_json_file_path = self.find_anat_nifti_paths(
                parent_directory=working_directory, anat_type="t1w"
            ) # type: ignore
            print(f"T1W NIFTI Image: {anat_t1w_nifti_image_path}")
            print(f"T1W JSON Path: {anat_t1w_json_file_path}")

            anat_t2w_nifti_image_path, anat_t2w_json_file_path = self.find_anat_nifti_paths(
                parent_directory=working_directory, anat_type="t2w"
            )
            print(f"T2W NIFTI Image: {anat_t2w_nifti_image_path}")
            print(f"T2W JSON Path: {anat_t2w_json_file_path}")

        elif len(anat_data) == 2: 
            for file in anat_data:
                if 't1w' in file.lower().strip():
                    if file.endswith(".nii.gz") or file.endswith(".nii"):
                        anat_t1w_nifti_image_path: str = file
                    elif file.endswith(".json"):
                        anat_t1w_json_file_path: str = file 
                    else:
                        raise ValueError(
                            "For argument 'anat_data', enter the path to the anatomical DICOM Directory "
                            "or give the paths to both the anatomical T1w NiFTI image and T1w JSON file. " 
                            "You may also pass the T2w NiFTI Image and T2w Json File in addition."
                            "len(anat_data) should be 1, 2 or 4."
                        ) 
        elif len(anat_data) == 4:
            for file in anat_data:
                if 't1w' in file.lower().strip():
                    if file.endswith(".nii.gz") or file.endswith(".nii"):
                        anat_t1w_nifti_image_path: str = file
                    elif file.endswith(".json"):
                        anat_t1w_json_file_path: str = file 
                elif 't2w' in file.lower().strip():
                    if file.endswith(".nii.gz") or file.endswith(".nii"):
                        anat_t2w_nifti_image_path: str | None = file
                    elif file.endswith(".json"):
                        anat_t2w_json_file_path: str | None = file 
                else:
                    raise ValueError(
                        "For argument 'anat_data', enter the path to the anatomical DICOM Directory "
                        "or give the paths to both the anatomical T1w NiFTI image and T1w JSON file. " 
                        "You may also pass the T2w NiFTI Image and T2w Json File in addition."
                        "len(anat_data) should be 1, 2 or 4."
                    ) 
        else:
            raise ValueError(
                "For argument 'anat_data', enter the path to the anatomical DICOM Directory "
                "or give the paths to both the anatomical T1w NiFTI image and T1w JSON file. " 
                "You may also pass the T2w NiFTI Image and T2w Json File in addition."   
            )

        if not anat_t1w_nifti_image_path or not anat_t1w_json_file_path:
            raise ValueError(
                "For argument 'anat_data', enter the path to the anatomical DICOM Directory "
                "or give the paths to both the anatomical T1w NiFTI image and T1w JSON file. " 
                "You may also pass the T2w NiFTI Image and T2w Json File in addition."   
            )
        elif anat_t2w_nifti_image_path and not anat_t2w_json_file_path:
            raise ValueError(
                "For argument 'anat_data', enter the path to the anatomical DICOM Directory "
                "or give the paths to both the anatomical T1w NiFTI image and T1w JSON file. " 
                "You may also pass the T2w NiFTI Image and T2w Json File in addition."   
            )
        elif anat_t2w_json_file_path and not anat_t2w_nifti_image_path:
            raise ValueError(
                "For argument 'anat_data', enter the path to the anatomical DICOM Directory "
                "or give the paths to both the anatomical T1w NiFTI image and T1w JSON file. " 
                "You may also pass the T2w NiFTI Image and T2w Json File in addition."   
            )
        print(f"Anatomical T1w NiFTI Image Path: {anat_t1w_nifti_image_path}")
        print(f"Anatomical T1w JSON File Path: {anat_t1w_json_file_path}")
        print(f"Anatomical T2w NiFTI Image Path: {anat_t2w_nifti_image_path}")
        print(f"Anatomical T2w JSON File Path: {anat_t2w_json_file_path}")

        return \
            func_nifti_image_path, \
            func_json_file_path, \
            anat_t1w_nifti_image_path,\
            anat_t1w_json_file_path, \
            anat_t2w_nifti_image_path, \
            anat_t2w_json_file_path


    def find_anat_nifti_paths(self, 
                              parent_directory: str, 
                              anat_type: str, 
                              raise_if_not_found = False
                              ) -> tuple[str | None, str | None]:
        
        all_json_files: list[str] = sorted(
            glob(os.path.join(parent_directory, "*.json")), 
            key=os.path.getmtime
        )

        found_json_file: str = ""
        for json_file in all_json_files:
            with open(json_file, mode='r') as file:
                json_data: dict[str, Any] = json.load(file)
                if not 'SeriesDescription' in json_data:
                    raise ValueError(
                        f"Could not find 'SeriesDescription' in JSON File: {json_file}")
                else:
                    if anat_type.lower().strip() in json_data['SeriesDescription'].lower().strip():
                        found_json_file: str = json_file
                        break 
        else:
            if raise_if_not_found:
                raise ValueError(
                    f"Could not find matching {anat_type} JSON file when looking for post-dcm2niix anat data"
                )
            else:
                return None, None


        found_nifti_file: str = os.path.join(
            parent_directory, 
            os.path.basename(found_json_file.replace(".json", ".nii.gz"))
        )
        if not os.path.exists(found_nifti_file):
            if raise_if_not_found:
                raise ValueError(
                    f"Could not find the accompanying NiFTI File: {found_nifti_file} "
                    f"for the matching JSON File: {found_json_file}")
            else:
                warnings.warn(
                    f"We could find a {anat_type} anatomical JSON file: {found_json_file} "
                    f"but the accompanying NiFTI File: {found_nifti_file} does not exist."
                    f"We will not use {anat_type} anatomical data."
                )
                return None, None
                
        return found_nifti_file, found_json_file


    def bids_rename_file(self, 
                         file_path: str,
                         subject_id_str: str, 
                         session_id_str: str,
                         series_name: str | None = None,
                         is_anat_image: bool = False,
                         run_num_str: str | None = None
                         ) -> str:
            
        # BIDS Format: sub-<label>[_ses-<label>]_task-<label>[_acq-<label>][_run-<index>]_<suffix>.<extension>
        
        def get_anat_file_type(filepath: str) -> str:
            if "t1w" in os.path.basename(filepath).lower():
                return 'T1w'
            elif 't2w'in os.path.basename(filepath).lower():
                return 'T2w'
            else:
                raise ValueError(
                    f"Cannot determine if {filepath} is a T1w image/json or T2w image/json."
                    "The image type should be in the file names"  
                )

        def get_file_extension(filepath: str) -> str:
            if '.nii' in os.path.basename(filepath):
                if os.path.basename(filepath).endswith('.nii.gz'):
                    return '.nii.gz'
                else:
                    return '.nii'
            elif os.path.basename(filepath).endswith('.json'):
                return '.json'
            else:
                raise ValueError(
                    f"File {filepath} must end in '.json', '.nii.gz', or '.nii'"
                )

        filename = f"{subject_id_str}_{session_id_str}"
        if is_anat_image:
            filename += "_" + get_anat_file_type(file_path)
        else:
            filename += f"_task-{series_name}_{run_num_str}_bold"
            
        filename += get_file_extension(file_path)
        
        return filename

def get_fmriprep_func_outputs(fmriprep_directory: str, subj_id: int, ses_id: int, run_id: int, 
                              find_mask: bool = True, file_extension: str = ".nii.gz",
                              space_keyword: str = ""  # leave space_keyword blank for subj-space
                              ) -> str:
    
    fmriprep_func_directory: str = os.path.join(
        fmriprep_directory, 
        f"sub-{'{:02d}'.format(subj_id)}",
        f"ses-{'{:02d}'.format(ses_id)}",
        "func"
    )

    participant_str: str = f"sub-{'{:02d}'.format(subj_id)}_ses-{'{:02d}'.format(ses_id)}_task-func_run-{'{:02d}'.format(run_id)}"
    if space_keyword != "":
        participant_str += f"_space-{space_keyword}"

    file_ending: str = "_desc-brain_mask" + file_extension if find_mask else "_desc-preproc_bold" + file_extension

    return os.path.join(fmriprep_func_directory, participant_str + file_ending)

# more verbose than subprocess.CalledProcessError
class CalledProcessError(subprocess.CalledProcessError):
    def __str__(self) -> str:
        return (
            f"Command resulted in non-zero exit code:\n\n"
            f"Command: {self.cmd}\n\n"
            f"Exit code: {self.returncode}\n\n"
            f"Stdout: {self.output}\n\n"
            f"Stderr: {self.stderr}"
        )
if __name__ == "__main__":
    import argparse
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run fMRIPrep on Single Subject, Single Run Data."
    )
    parser.add_argument(
        "--func_data",
        required=True,
        nargs='+',
        help=\
            "Enter the path to the functional DICOM Directory or give the " + \
            "paths to both the functional NiFTI image and JSON file."
    )
    parser.add_argument(
        "--anat_data",
        required=True,
        nargs="+",
        help=\
            "Enter the path to the anatomical DICOM Directory or give the " + \
            "paths to both the T1W NiFTI image and it's JSON file. " + \
            "If you are inputting NiFTI data, you may also pass the " + \
            "T2W NiFTI image and it's JSON file."
    )
    parser.add_argument(
        "--series_name",
        required=True
    )
    parser.add_argument(
        "--subject_id",
        required=True,
        type=int
    )
    parser.add_argument(
        "--session_num",
        required=True,
        type=int
    )
    parser.add_argument(
        "--run_num",
        required=True,
        type=int
    )
    parser.add_argument(
        "--working_directory",
        required=False,
        default="working",
        help=\
            f"Default: {os.path.abspath('working')}"
    )
    parser.add_argument(
        "--output_directory",
        required=False,
        default='fmriprep_outputs',
        help=\
            f"Default: {os.path.abspath('outputs')}"
    )
    parser.add_argument(
        "--n_jobs",
        required=False,
        type=int,
        default=os.cpu_count(),
        help=f"Default: 1 Job per CPU Core (n_jobs = os.cpu_count())."
    )
    args: argparse.Namespace = parser.parse_args()
    StartSingleRunfMRIPrep(
        func_data=[
            os.path.abspath(file_path)
            for file_path in args.func_data
        ],
        anat_data=[
            os.path.abspath(file_path)
            for file_path in args.anat_data
        ],
        series_name=args.series_name,
        subject_id=args.subject_id,
        session_num=args.session_num,
        run_num=args.run_num,
        working_directory=os.path.abspath(args.working_directory),
        output_directory=os.path.abspath(args.output_directory),
        n_jobs=args.n_jobs
    )
    