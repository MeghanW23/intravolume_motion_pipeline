import os
import sys
import warnings
import subprocess
import SimpleITK as sitk
from collections import OrderedDict

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from decompress_dicoms import DecompressDicoms
from dicom_to_nifti import DicomToNifti
from get_slice_timing import GetSliceTiming
from determine_motion_threshold import DetermineMotionThreshold
from remove_background import RemoveBackground

class StartMotionCorrection:
    def __init__(self, 
                 radian_parameters_text_file: str, 
                 displacements_text_file: str,
                 nifti_image_path: str | None,  # pyright: ignore[reportRedeclaration]
                 json_file_path: str | None,  # pyright: ignore[reportRedeclaration]
                 dicom_directory: str | None = None, 
                 series_name: str | None = None,
                 motion_threshold: float | None = None, # pyright: ignore[reportRedeclaration]
                 percent_of_volumes_cutoff: int = 20,
                 output_directory_path: str = "outputs",
                 working_directory_path: str = "working",
                 matlab_main_script_path: str = os.path.abspath('main_cameraparams.m'),
                 noscrubbing_recon_filename_prefix: str = "no-scrubbing_motion-corrected_func_image",
                 scrubbed_recon_filename_prefix: str = "scrubbed_and_motion-corrected_func_image",
                 dcmdjpeg_path: str = 'dcmdjpeg',
                 dcm2niix_path: str = 'dcm2niix',
                 matlab_path: str = 'matlab') -> None:

        self.validate_inputted_data(dicom_directory, nifti_image_path, json_file_path)
        os.makedirs(output_directory_path, exist_ok=True)
        os.makedirs(working_directory_path, exist_ok=True)
        self.motion_threshold: float | None = motion_threshold # pyright: ignore[reportRedeclaration, reportAttributeAccessIssue]

        if dicom_directory != None: 
            """
            =======================================================
            DECOMPRESS DICOMS 
            =======================================================
            """
            decompression_directory: str = os.path.join(working_directory_path, "decompressed_dicoms")
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
                output_directory=working_directory_path,
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
        slice_ordering_path: str = os.path.join(output_directory_path, "slice_order.txt")
        slice_timing: OrderedDict[float, list[int]] = GetSliceTiming(
            json_data=json_file_path, # pyright: ignore[reportArgumentType]
            output_txt_slice_order_path=slice_ordering_path
        ).return_slice_timing()
        print(f"Slice Ordering Text File at: {slice_ordering_path}")

        sms_factor: int = len(list(slice_timing.values())[0])
        print(f"SMS Factor: {sms_factor}")

        """
        =======================================================
        GET MOTION THRESHOLD IN MM
        =======================================================
        """
        if self.motion_threshold == None:
            threshold_output_directory: str = os.path.join(output_directory_path, "threshold_selection_outputs")
            self.motion_threshold: float = DetermineMotionThreshold(
                displacements_text_file=displacements_text_file,
                nifti_image_path=nifti_image_path, # pyright: ignore[reportArgumentType]
                json_file_path=json_file_path, # pyright: ignore[reportArgumentType]
                output_directory=threshold_output_directory,
                percent_of_volumes_cutoff=percent_of_volumes_cutoff
            ).return_motion_threshold()
            print(f"Recevied Motion Threshold of: {self.motion_threshold} mm")
            print(f"See Threshold Selection Plots at: {threshold_output_directory}")

        """
        =======================================================
        REMOVE BACKGROUND FROM 4D Image
        =======================================================
        """
        print(f"Removing the background from 4D NiFTI Image: {nifti_image_path}")
        bgremoved_input_nifti_image_path: str = os.path.join(
            output_directory_path, 
            f"{os.path.basename(nifti_image_path).replace('.nii.gz', '').replace('.nii', '')}_bgremoved.nii.gz" # type: ignore
        )
        RemoveBackground(
            nifti_file_path=nifti_image_path, # type: ignore
            output_file_path=bgremoved_input_nifti_image_path,
            save_binary_mask=True
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
        matlab_function: str = (
            f"{os.path.basename(matlab_main_script_path)[:-2]}("
            f"'{nifti_image_path}', "
            f"'{bgremoved_input_nifti_image_path}', "
            f"'{output_directory_path}', "
            f"'{radian_parameters_text_file}', "
            f"'{displacements_text_file}', "
            f"'{slice_ordering_path}', "
            f"'{noscrubbing_recon_filename_prefix}', "
            f"'{scrubbed_recon_filename_prefix}', "
            f"{self.motion_threshold}, "
            f"{sms_factor});"
        )
        command: list[str] = [
            matlab_path,
            "-batch",
            f"addpath('{os.path.dirname(matlab_main_script_path)}'); "
            f"clear; close all; clc; "
            f"{matlab_function}"
        ]
        print(f"Running MATLAB Script: {command}")
        subprocess.run(
            command,
            check=True
        )
        print("Motion Correction Step Finished")
        

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

    
def find_intravolume_corrected_data(output_directory_path: str, scrubbed_data_filename_prefix: str, nonscrubbed_data_filename_prefix: str) -> str:

    corrected_image: str = os.path.join(output_directory_path, scrubbed_data_filename_prefix + ".nii.gz")
    if not os.path.exists(corrected_image):
        corrected_image: str = os.path.join(output_directory_path, nonscrubbed_data_filename_prefix + ".nii.gz") 
        warnings.warn(
            message=(
                f"Could not find scrubbed data at: {os.path.join(output_directory_path, scrubbed_data_filename_prefix + '.nii.gz')}"
                f"Using file: {corrected_image}. "
                "If your data has no above-threshold motion, this is expected. "
                "If not, something has gone wrong. "
            ),
            category=UserWarning
        )
        return corrected_image
    
    else:
        return corrected_image

    
if __name__ == "__main__":
    import argparse
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=\
            "Prepares inputs and starts the main MATLAB script " \
            "for the motion correction step. " \
            "The code for the motion correction step can be found at: " \
            "https://github.com/MeghanW23/rsfMRI_SMC_mc"
    )
    parser.add_argument(
        "--main_matlab_script_path",
        required=False,
        default="main_cameraparams.m",
        help=\
            "The path on your system to the main motion correction script, " \
            "likely named 'main_cameraparams.m'. Make sure this is the slice to volume version. " \
            f"Default: {os.path.abspath('main_cameraparams.m')}"
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
        "--nifti_image_path",
        required=False,
        help=\
            "The raw 4D NiFTI Image. " + \
            "Please enter either a value for --dicom_directory OR " + \
            " a value for BOTH: --nifti_image_file_path and --json_file_path."
    )
    parser.add_argument(
        "--json_file_path",
        required=False,
        help=\
            "The JSON sidecar for the raw 4D NiFTI Image. " + \
            "Please enter either a value for --dicom_directory OR " + \
            " a value for BOTH: --nifti_image_file_path and --json_file_path."
    )
    parser.add_argument(
        "--output_directory_path",
        required=False,
        default="outputs",
        help=f"Default: {os.path.abspath('outputs')}"
    )
    parser.add_argument(
        "--working_directory_path",
        required=False,
        default="working",
        help=os.path.abspath('working')
    )
    parser.add_argument(
        "--radian_parameters_text_file",
        required=True,
        help=\
            "The radian parameters text file outputted " \
            "during the motion characterization step"
    )
    parser.add_argument(
        "--displacements_text_file",
        required=True,
        help=\
            "The displacements text file outputted " \
            "during the motion characterization step"
    )
    parser.add_argument(
        "--motion_threshold",
        required=False,
        default=None,
        type=float,
        help=\
            "Volume where ANY slice group has a displacement value " \
            "> the motion threshold will be scrubbed. Takes in the threshold " \
            "in mm. Leave as None and we will calculate a threshold." \
            "Default: None."
    )
    args: argparse.Namespace = parser.parse_args()
    StartMotionCorrection(
        matlab_main_script_path=os.path.abspath(args.main_matlab_script_path),
        dicom_directory=os.path.abspath(args.dicom_directory) if args.dicom_directory else None,
        series_name=args.series_name,
        nifti_image_path=os.path.abspath(args.nifti_image_path) if args.nifti_image_path else None,
        json_file_path=os.path.abspath(args.json_file_path) if args.json_file_path else None,
        radian_parameters_text_file=os.path.abspath(args.radian_parameters_text_file),
        displacements_text_file=os.path.abspath(args.displacements_text_file),
        motion_threshold=args.motion_threshold,
        output_directory_path=os.path.abspath(args.output_directory_path),
        working_directory_path=os.path.abspath(args.working_directory_path)
    )
