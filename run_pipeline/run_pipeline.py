import os 
import sys
import yaml
import shutil
import warnings
from glob import glob

# add script directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))
from manage_configuration_files import Configurations
from decompress_dicoms import DecompressDicoms
from dicom_to_nifti import DicomToNifti
from MotionCharacterization import CharacterizeIntraVolumeMotion
from MotionCorrection import StartMotionCorrection
from SingleRunfMRIPrep import StartSingleRunfMRIPrep
 
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
                    output_directory=os.path.join(configurations.WORKING_DIRECTORY_PATH, "func_images"),
                    dcm2niix_path=configurations.DCM2NIIX_PATH
                )
                self.func_nifti_image_path: str = dcm2niix_module.return_nifti_image() # pyright: ignore[reportAttributeAccessIssue]
                self.func_json_file_path: str = dcm2niix_module.return_json_file() # pyright: ignore[reportAttributeAccessIssue]

        # copy raw data into output directory
        if not os.path.basename(self.func_nifti_image_path) in configurations.OUTPUT_DIRECTORY_PATH:
            print(f"Copying Raw NiFTI Data into Output Directory: {configurations.OUTPUT_DIRECTORY_PATH}")
            shutil.copy(
                src=self.func_nifti_image_path,
                dst=os.path.join(configurations.OUTPUT_DIRECTORY_PATH, os.path.basename(self.func_nifti_image_path))
            )
        if not os.path.basename(self.func_json_file_path) in configurations.OUTPUT_DIRECTORY_PATH:
            print(f"Copying Raw JSON File into Output Directory: {configurations.OUTPUT_DIRECTORY_PATH}")
            shutil.copy(
                    src=self.func_json_file_path,
                    dst=os.path.join(configurations.OUTPUT_DIRECTORY_PATH, os.path.basename(self.func_json_file_path))
                )


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
                run_environment=configurations.OPTIMIZER_RUN_ENVIRONMENT,
                smsmireg_executable_path=configurations.OPTIMIZER_EXECUTABLE_PATH,
                singularity_image_path=configurations.OPTIMIZER_SINGULARITY_IMAGE_PATH,
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

        """
        ========================================
        STEP TWO: MOTION CORRECTION
        ========================================
        """
        if configurations.RUN_MOTION_CORRECTION:
            self.motion_correction_step = StartMotionCorrection(
                radian_parameters_path=os.path.join(configurations.OUTPUT_DIRECTORY_PATH, "radian-parameters.txt"),
                nifti_image_path=func_nifti_image_path,
                json_file_path=func_json_file_path,
                working_directory=configurations.WORKING_DIRECTORY_PATH,
                output_directory=configurations.OUTPUT_DIRECTORY_PATH,
                motion_threshold=configurations.MOTION_THRESHOLD,
                dcmdjpeg_path=configurations.DCMDJPEG_PATH,
                dcm2niix_path=configurations.DCM2NIIX_PATH,
                matlab_path=configurations.MATLAB_INSTALLATION_PATH,
                n_jobs=configurations.N_JOBS
            )
        from MotionCorrection.__main__ import find_intravolume_corrected_data
        self.motion_corrected_image_path: str = find_intravolume_corrected_data(
            output_directory_path=configurations.OUTPUT_DIRECTORY_PATH,
            mcorr_output_filename_pattern=configurations.MCORR_OUTPUT_FILENAME_PATTERN,
            mcorr_abruptmotion_filename=configurations.MCORR_ABRUPTMOTION_FILE_NAME
        )
        
        """
        ========================================
        STEP THREE: FMRIPREP
        ========================================
        """
        if configurations.RUN_FMRIPREP:

            

            StartSingleRunfMRIPrep(
                func_data=[self.motion_corrected_image_path, self.func_json_file_path],
                anat_data=\
                    [configurations.ANATOMICAL_DICOM_DIRECTORY] 
                    if configurations.ANATOMICAL_DICOM_DIRECTORY
                    else [configurations.ANATOMICAL_NIFTI_IMAGE_PATH, configurations.ANATOMICAL_JSON_FILE_PATH], # pyright: ignore[reportArgumentType]
                series_name=configurations.SERIES_NAME, # pyright: ignore[reportArgumentType]
                subject_id=configurations.SUBJECT_ID, # pyright: ignore[reportArgumentType]
                session_num=configurations.SESSION_NUM, # pyright: ignore[reportArgumentType]
                run_num=configurations.RUN_NUM, # pyright: ignore[reportArgumentType]
                working_directory=configurations.WORKING_DIRECTORY_PATH,
                output_directory=configurations.OUTPUT_DIRECTORY_PATH,
                dcmdjpeg_path=configurations.DCMDJPEG_PATH,
                dcm2niix_path=configurations.DCM2NIIX_PATH,
                FMRIPREP_CONTAINER_PATH=configurations.FMRIPREP_CONTAINER_PATH,
                FMRIPREP_TEMPLATEFLOW_DIRECTORY=configurations.FMRIPREP_TEMPLATEFLOW_DIRECTORY,
                n_jobs=configurations.N_JOBS,
                omp_nthreads=configurations.OMP_NTHREADS, # pyright: ignore[reportArgumentType],
                mem_mb=configurations.MEM_MB # pyright: ignore[reportArgumentType]
            )        


if __name__ == "__main__":
    import argparse
    parser: argparse.ArgumentParser =  argparse.ArgumentParser(
        description="This script is the wrapper for running the whole intravolume motion pipeline."
    )
    parser.add_argument("--configuration_file", required=True, help=".yaml configuration file")
    args: argparse.Namespace = parser.parse_args()
    RunPipeline(configuration_file=os.path.abspath(args.configuration_file))