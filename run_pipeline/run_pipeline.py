import os 
import sys
import yaml
import shutil
# add script directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))
from manage_configuration_files import Configurations
from decompress_dicoms import DecompressDicoms
from dicom_to_nifti import DicomToNifti
from MotionCharacterization import CharacterizeIntraVolumeMotion
from MotionCorrection import StartMotionCorrection
 
class RunPipeline:
    def __init__(self, configuration_file: str) -> None:
        if not os.path.exists(configuration_file):
            raise FileNotFoundError(
                f"Could not find configuration file: {configuration_file}"
            )
        """
        ========================================
        LOAD CONFIGURATION FILE 
        ========================================
        """
        print(f"Loading Input Configuration File: {configuration_file}")
        configurations = None
        with open(configuration_file, mode='r') as file:
            configurations = Configurations(**yaml.safe_load(file))

        """
        ========================================
        SET UP PATHS
        ========================================
        """
        # setup working and output directories
        os.makedirs(os.path.dirname(configurations.WORKING_DIRECTORY_PATH), exist_ok=True)
        os.makedirs(configurations.WORKING_DIRECTORY_PATH, exist_ok=True)
        print(f"Working Directory: {configurations.WORKING_DIRECTORY_PATH}")

        os.makedirs(os.path.dirname(configurations.OUTPUT_DIRECTORY_PATH), exist_ok=True)
        os.makedirs(configurations.OUTPUT_DIRECTORY_PATH, exist_ok=True)
        print(f"Output Directory: {configurations.OUTPUT_DIRECTORY_PATH}")

        print("Making a Copy of your Configuration File in the Output Directory")
        copied_configuration_file_path: str = os.path.join(configurations.OUTPUT_DIRECTORY_PATH, os.path.basename(configuration_file))
        shutil.copy(
            src=configuration_file,
            dst=copied_configuration_file_path
        )

        """
        ========================================
        DO DICOM TO NIFTI IF NEEDED 
        ========================================
        """
        func_nifti_image_path: str | None = configurations.FUNCTIONAL_NIFTI_IMAGE_PATH # pyright: ignore
        func_json_file_path: str | None = configurations.FUNCTIONAL_JSON_FILE_PATH # pyright: ignore
        if configurations.FUNCTIONAL_DICOM_DIRECTORY:
                """
                =======================================================
                DECOMPRESS DICOMS 
                =======================================================
                """
                decompression_directory: str = os.path.join(configurations.WORKING_DIRECTORY_PATH, "func_decompressed_dicoms")
                print(f"Decompressing DICOMs into directory: {decompression_directory}")
                DecompressDicoms(
                    dicom_directory=configurations.FUNCTIONAL_DICOM_DIRECTORY,
                    output_directory=decompression_directory,
                    dcmdjpeg_path=configurations.DCMDJPEG_PATH,
                    series_name=configurations.SERIES_NAME
                )
    
                """
                =======================================================
                DO DICOM TO NIFTI
                =======================================================
                """
                print(f"Doing dcm2niix on the Decompressed Dicoms at: {decompression_directory}")
                dcm2niix_module: DicomToNifti = DicomToNifti(
                    dicom_directory=decompression_directory,
                    output_directory=configurations.WORKING_DIRECTORY_PATH,
                    dcm2niix_path=configurations.DCM2NIIX_PATH
                )
                func_nifti_image_path: str = dcm2niix_module.return_nifti_image() # pyright: ignore[reportAssignmentType]
                func_json_file_path: str = dcm2niix_module.return_json_file() # pyright: ignore[reportAssignmentType]

        """
        ========================================
        STEP ONE: MOTION CHARACTERIZATION
        ========================================
        """
        if configurations.RUN_MOTION_CHARACTERIZATION:
            CharacterizeIntraVolumeMotion(
                nifti_image_path=func_nifti_image_path,
                json_file_path=func_json_file_path,
                working_directory=configurations.WORKING_DIRECTORY_PATH,
                output_directory=configurations.OUTPUT_DIRECTORY_PATH,
                run_environment=configurations.SMS_MI_REG_RUN_ENVIRONMENT,
                smsmireg_executable_path=configurations.SMS_MI_REG_EXECUTABLE_PATH,
                singularity_image_path=configurations.SMS_MI_REG_SINGULARITY_IMAGE_PATH,
                n_jobs=configurations.N_JOBS,
                motion_threshold=configurations.MOTION_THRESHOLD,
                reference_volume_index=configurations.REFERENCE_VOLUME_INDEX,
                limit_voxel_intensity=configurations.LIMIT_VOXEL_INTENSITY,
                voxel_intensity_lower_bound=configurations.VOXEL_LOWER_BOUND,
                voxel_intensity_upper_bound=configurations.VOXEL_UPPER_BOUND,
                dcmdjpeg_path=configurations.DCMDJPEG_PATH,
                dcm2niix_path=configurations.DCM2NIIX_PATH,
                upsample_reference_volume=configurations.UPSAMPLE_REFERENCE_VOLUME,
                reference_volume_spacing=configurations.REFERENCE_VOLUME_SPACING,
                head_radius=configurations.HEAD_RADIUS
            )
        if configurations.RUN_MOTION_CORRECTION:
            StartMotionCorrection(
                radian_parameters_path=os.path.join(configurations.OUTPUT_DIRECTORY_PATH, "radian-parameters.txt"),
                nifti_image_path=func_nifti_image_path,
                json_file_path=func_json_file_path,
                working_directory=configurations.WORKING_DIRECTORY_PATH,
                output_directory=configurations.OUTPUT_DIRECTORY_PATH,
                motion_threshold=configurations.MOTION_THRESHOLD,
                dcmdjpeg_path=configurations.DCMDJPEG_PATH,
                dcm2niix_path=configurations.DCM2NIIX_PATH,
                matlab_path=configurations.MATLAB_INSTALLATION_PATH
            )

if __name__ == "__main__":
    import argparse
    parser: argparse.ArgumentParser =  argparse.ArgumentParser(
        description="This script is the wrapper for running the whole intravolume motion pipeline."
    )
    parser.add_argument("--configuration_file", required=True, help=".yaml configuration file")
    args: argparse.Namespace = parser.parse_args()
    RunPipeline(configuration_file=os.path.abspath(args.configuration_file))