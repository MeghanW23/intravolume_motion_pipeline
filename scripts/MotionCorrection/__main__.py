import os 
import subprocess
import sys 
import json
import warnings
from glob import glob
from typing import Any
import SimpleITK as sitk 

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# in order of usage:
from decompress_dicoms import DecompressDicoms
from dicom_to_nifti import DicomToNifti
from get_motion_threshold import GetMotionThreshold
from remove_background import RemoveBackground

class StartMotionCorrection:
    def __init__(self,
                 radian_parameters_path: str,
                 dicom_directory: str | None = None, 
                 series_name: str | None = None,
                 nifti_image_path: str | None = None, # pyright: ignore[reportRedeclaration]
                 json_file_path: str | None = None, # pyright: ignore[reportRedeclaration]
                 working_directory: str = os.path.abspath('working'),
                 output_directory: str = os.path.abspath('outputs'),
                 motion_threshold: int = 10,
                 dcmdjpeg_path: str = 'dcmdjpeg',
                 dcm2niix_path: str = 'dcm2niix',
                 matlab_path: str = 'matlab',
                 mcorr_output_filename_pattern: str = "recon_scrub_betainit1_*.nii.gz",
                 mcorr_abruptmotion_filename: str = "recon_abruptmotion.nii.gz",
                 n_jobs: int = - 1):

        print(f"\n========== Starting Intravolume Motion Correction ========== ")
        print(f"Radian Parameters Text File: {radian_parameters_path}")
        print(f"DICOM Directory: {dicom_directory}")
        print(f"Series Name: {series_name}")
        print(f"NiFTI Image Path: {nifti_image_path}")
        print(f"JSON File Path: {json_file_path}")
        print(f"Working Directory: {working_directory}")
        print(f"Output Directory: {output_directory}")
        print(f"Motion Threshold: {motion_threshold}")
        print(f"dcmdjpeg Path/Command: {dcmdjpeg_path}")
        print(f"dcm2niix Path/Command: {dcm2niix_path}")
        print(f"MATLAB Path/Command: {matlab_path}")
        print(f"Number of Jobs: {n_jobs}")
        print("======================================================================\n")
        print(
            "NOTE: The motion correction step requires quite a bit of memory. If your "
            "script dies suddenly check your memory allocation."
        )

        self.validate_inputted_data(dicom_directory, nifti_image_path, json_file_path)

        os.makedirs(working_directory, exist_ok=True)
        os.makedirs(output_directory, exist_ok=True)


        """
        =======================================================
        SET INPUT SCRIPT PATHS
        =======================================================
        """
        motion_correction_main_directory: str = os.path.abspath(os.path.dirname(__file__))
        print(f"Main Motion Correction Directory: {motion_correction_main_directory}")
        motion_correction_main_script_dir: str = \
            os.path.join(motion_correction_main_directory, "main_scripts")
        print(f"Motion Correction Main Scripts Directory: {motion_correction_main_script_dir}")
        motion_correction_main_function_name: str = "main_cameraparams"
        print(f"Motion Correction Main MATLAB Script Name: {motion_correction_main_function_name}")
        directliftandunliftcodes_dir: str = \
            os.path.join(motion_correction_main_directory, "direct-liftandunlift-codes")
        print(f"direct-liftandunlift-codes Scripts Directory: {directliftandunliftcodes_dir}")
        operators_dir: str = \
            os.path.join(motion_correction_main_directory, "operators")
        print(f"operators Script Directory: {operators_dir}")
        reconabruptmotion_file_path: str = \
            os.path.join(output_directory, "recon_abruptmotion.nii.gz")


        if dicom_directory != None: 
            """
            =======================================================
            DECOMPRESS DICOMS 
            =======================================================
            """
            decompression_directory: str = os.path.join(working_directory, "decompressed_dicoms")
            print(f"Decompressing DICOMs into directory: {decompression_directory}")
            DecompressDicoms(
                dicom_directory=dicom_directory,
                output_directory=decompression_directory,
                dcmdjpeg_path=dcmdjpeg_path,
                series_name=series_name
            )

            """
            =======================================================
            DO DICOM TO NIFTI
            =======================================================
            """
            print(f"Doing dcm2niix on the Decompressed Dicoms at: {decompression_directory}")
            dcm2niix_module: DicomToNifti = DicomToNifti(
                dicom_directory=decompression_directory,
                output_directory=working_directory,
                dcm2niix_path=dcm2niix_path
            )
            nifti_image_path: str = dcm2niix_module.return_nifti_image() # pyright: ignore[reportAssignmentType]
            json_file_path: str = dcm2niix_module.return_json_file() # pyright: ignore[reportAssignmentType]

        nifti_image: sitk.Image = sitk.ReadImage(nifti_image_path) # pyright: ignore[reportArgumentType]
        dimensions: tuple[int, int, int, int] = nifti_image.GetSize()
        print(f"NiFTI Image Dimensions: {dimensions}")
        if len(dimensions) != 4:
            raise ValueError("Input data must be 4D.")


        """
        =======================================================
        GET SLICE TIMING
        =======================================================
        """
        slice_timing: list[float] = self.get_slice_timing(json_file_path) # pyright: ignore[reportArgumentType]
        print(f"Slice Timing: {slice_timing}")

        sms_factor: int = self.get_sms_factor(slice_timing)
        print(f"SMS Factor: {sms_factor}")


        """
        =======================================================
        GET MOTION THRESHOLD IN MM
        =======================================================
        """
        threshold_in_mm: float = GetMotionThreshold(
            nifti_image=nifti_image_path, # pyright: ignore[reportArgumentType]
            threshold_as_percent=motion_threshold
        ).return_mm_threshold()
        print(f"Motion Threshold for Scrubbing in mm: {threshold_in_mm}")


        """
        =======================================================
        REMOVE BACKGROUND FROM 4D Image
        =======================================================
        """
        print(f"Removing the background from 4D NiFTI Image: {nifti_image_path}")
        bgremoved_input_nifti_image_path: str = os.path.join(
            output_directory, 
            f"{os.path.basename(nifti_image_path).replace('.nii.gz', '').replace('.nii', '')}_bgremoved.nii.gz" # type: ignore
        )
        RemoveBackground(
            nifti_file_path=nifti_image_path, # type: ignore
            output_file_path=bgremoved_input_nifti_image_path

        )


        """
        =======================================================
        SET ENVIRONMENT VARIABLES FOR SLURM
        =======================================================
        """
        slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK', '1') 
        print(f"Slurm CPUs per Task: {slurm_cpus}")

        for env_var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'MATLAB_NUM_THREADS']:
            print(f"{env_var}: {slurm_cpus}")
            os.environ[env_var] = slurm_cpus

        """
        =======================================================
        START MATLAB MAIN SCRIPT
        =======================================================
        """
        self.run_command([
            matlab_path, "-nosplash", "-nodesktop", "-r",
            f"addpath('{motion_correction_main_script_dir}'); \
            clear;close all;clc; \
            {motion_correction_main_function_name}( \
                '{output_directory}', \
                '{nifti_image_path}', \
                '{bgremoved_input_nifti_image_path}', \
                '{radian_parameters_path}', \
                '{reconabruptmotion_file_path}', \
                '{motion_correction_main_script_dir}', \
                '{directliftandunliftcodes_dir}', \
                '{operators_dir}', \
                '{str(sms_factor)}', \
                '{str(threshold_in_mm)}', \
                '{', '.join([str(val) for val in slice_timing])}', \
                '{str(n_jobs)}'); \
            exit;" 
        ])

    def validate_inputted_data(self,
                               dicom_directory: str | None = None,
                               nifti_image_path: str | None = None,
                               json_file_path: str | None = None):

        error_msg: str = \
            "Please enter either a value for 'dicom_directory' OR " + \
            "a value for BOTH: 'nifti_image_path' AND 'json_file_path'."
        

        dicom_mode: bool = dicom_directory is not None
        nifti_mode: bool = nifti_image_path is not None and json_file_path is not None
        nifti_partial: bool = nifti_image_path is not None or json_file_path is not None

        # valid only if exactly one mode is fully satisfied, and no overlap
        if dicom_mode and (nifti_partial):
            raise ValueError(error_msg)
        if not dicom_mode and not nifti_mode:
            raise ValueError(error_msg)


    def get_slice_timing(self, json_file_path: str) -> list[float]:
        with open(json_file_path, mode='r') as file:
            data: dict[str, Any] =  json.load(file)
            if not 'SliceTiming' in data:
                raise ValueError(
                    f"The key: 'Slice Timing' is not in your JSON file: {json_file_path}"
                )
            return data['SliceTiming']


    def get_sms_factor(self, slice_timing_list: list[float]) -> int:
        num_slices: int = len(slice_timing_list)
        print(f"Number of Slices in Each 3D Volume: {num_slices}")

        num_slice_groups: int = len(set(slice_timing_list))
        print(f"Number of Slice Groups in Each 3D Volume: {num_slice_groups}")

        sms_factor: float  = num_slices / num_slice_groups
        if not sms_factor.is_integer():
            warnings.warn(
                message=(
                    f"Your SMS Factor ({sms_factor}) is not an integer."
                    "sms_factor = num_slices / num_slice_groups."
                )
            )
        return int(sms_factor)


    def run_command(self, command: list[str], verbose: bool = True):
        if verbose:
            print(f"Running Command: {command}")

        subprocess.run(
            command
        )

        
def find_intravolume_corrected_data(output_directory_path: str, mcorr_output_filename_pattern: str, mcorr_abruptmotion_filename: str) -> str:
        corrected_image: str = ""
        matching_corrected_files: list[str] = glob(os.path.join(output_directory_path, mcorr_output_filename_pattern))
        if len(matching_corrected_files) == 0:
            corrected_image: str = os.path.join(output_directory_path, mcorr_abruptmotion_filename) 
            warnings.warn(
                message=(
                    f"Could not find scrubbed data matching pattern: {mcorr_output_filename_pattern}"
                    f"Using file: {corrected_image}. "
                    "If your data has no above-threshold motion, this is expected. "
                    "If not, something has gone wrong. "
                ),
                category=UserWarning
            )
            return corrected_image

        elif len(matching_corrected_files) > 1:
            corrected_image: str = sorted(matching_corrected_files, key=os.path.getmtime)[-1]
            warnings.warn(
                message=(
                    f"More than one scrubbed NiFTI Image Found: {matching_corrected_files}. "
                    f"Using the most recently modified image: {corrected_image}"
                ),
                category=UserWarning
            )
            return corrected_image
        
        else:
            return matching_corrected_files[0]


def find_binary_mask(output_directory_path: str, pattern: str = "*_bgremoved_MASK.nii.gz") -> str:
    matching_files: list[str] = sorted(glob(os.path.join(output_directory_path, pattern)), key=os.path.getmtime)
    matching_file: str = matching_files[-1]

    if len(matching_files) == 0:
        raise FileNotFoundError(
            f"Could not find file matching: {os.path.join(output_directory_path, pattern)}. " \
            "Is the class RemoveBackground's argument: 'save_binary_mask' set to True? "
        )
    
    elif len(matching_files) > 1:
        warnings.warn(
            message=\
                f"Found more than one file match the pattern: {os.path.join(output_directory_path, pattern)}. " \
                f"Returning the most recently modified file: {matching_file}.",
            category=UserWarning
            )

    return matching_file


if __name__ == "__main__":
    import argparse
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=\
            "Script for starting the motion correction pipeline."
    )
    parser.add_argument(
            "--dicom_directory",
            required=False,
            help=\
                "Please enter either a value for --dicom_directory OR " + \
                " a value for BOTH: --nifti_image_file_path and --json_file_path.",
            default=None
        )
    parser.add_argument(
        "--series_name",
        required=False,
        default=None,
        help=\
            "This pipeline runs on single-subject, single run data. " + \
            "If your DICOM Directory has more than one run, please provide " + \
            "the name of the task/series you are wanting to run the analysis on. " + \
            "Any inputted series name must match the string in the DICOM metadata " + \
            "key 'SeriesDescription' (not case sensitive)."
    )
    parser.add_argument(
        "--nifti_image_file_path",
        required=False,
        help=\
            "4D NiFTI image. " + \
            "Please enter either a value for --dicom_directory OR " + \
            " a value for BOTH: --nifti_image_file_path and --json_file_path.",
        default=None
    )
    parser.add_argument(
        "--json_file_path",
        required=False,
        help=\
            "Please enter either a value for --dicom_directory OR " + \
            " a value for BOTH: --nifti_image_file_path and --json_file_path.",
        default=None
    )
    parser.add_argument(
        "--radian_parameters_path",
        required=True,
        help=\
            "Input the motion parameters determined in the motion characterization step as " + \
            ".txt file where each row contains 6 parameters, one for each dimension in this order: " + \
            " x rotation, y rotation, z rotation, x translation, y translation, z translation. " + \
            "NOTE: These parameters must be in RADIANS."
    )
    parser.add_argument(
        "--output_directory_path",
        required=False,
        default='outputs',
        help=f"Default: {os.path.abspath('outputs')}"
    )
    parser.add_argument(
        "--working_directory_path",
        required=False,
        default='working',
        help=f"Default: {os.path.abspath('working')}"
    )
    parser.add_argument(
        "--motion_threshold",
        required=False,
        default=10,
        type=int,
        help=\
            "The percentage of the diameter of a single voxel. " + \
            "Any volumes with any motion about this threshold will be scrubbed. " + \
            "Default: 10 Percent."
    )
    args: argparse.Namespace = parser.parse_args()
    StartMotionCorrection(
        radian_parameters_path=os.path.abspath(args.radian_parameters_path),
        dicom_directory=\
            os.path.abspath(args.dicom_directory)
            if args.dicom_directory else None,
        series_name=args.series_name,
        nifti_image_path=\
            os.path.abspath(args.nifti_image_file_path)
            if args.nifti_image_file_path else None,
        json_file_path=\
            os.path.abspath(args.json_file_path)
            if args.json_file_path else None,
        working_directory=os.path.abspath(args.working_directory_path),
        output_directory=os.path.abspath(args.output_directory_path),
        motion_threshold=args.motion_threshold
    )
