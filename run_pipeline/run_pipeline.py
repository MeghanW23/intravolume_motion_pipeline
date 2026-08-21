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
from determine_motion_threshold import DetermineMotionThreshold
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
        if os.path.exists(copied_configuration_file_path):
            if copied_configuration_file_path != configuration_file:
                os.remove(copied_configuration_file_path)
        shutil.copy(
            src=configuration_file,
            dst=copied_configuration_file_path
        )

        """
        ========================================
        DO DICOM TO NIFTI IF NEEDED 
        ========================================
        """
        self.func_nifti_image_path: str | None = configurations.FUNCTIONAL_NIFTI_IMAGE_PATH # pyright: ignore
        self.func_json_file_path: str | None = configurations.FUNCTIONAL_JSON_FILE_PATH # pyright: ignore
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
        copied_nifti_path: str = os.path.join(configurations.OUTPUT_DIRECTORY_PATH, os.path.basename(self.func_nifti_image_path))
        if not os.path.exists(copied_nifti_path):
            print(f"Copying Raw NiFTI Data into Output Directory: {configurations.OUTPUT_DIRECTORY_PATH}")
            shutil.copy(
                src=self.func_nifti_image_path,
                dst=copied_nifti_path
            )
        copied_json_path: str = os.path.join(configurations.OUTPUT_DIRECTORY_PATH, os.path.basename(self.func_json_file_path))
        if not os.path.exists(copied_json_path):
            print(f"Copying Raw JSON File into Output Directory: {configurations.OUTPUT_DIRECTORY_PATH}")
            shutil.copy(
                    src=self.func_json_file_path,
                    dst=copied_json_path
                )


        """
        ========================================
        STEP ONE: MOTION CHARACTERIZATION
        ========================================
        """
        if configurations.RUN_MOTION_CHARACTERIZATION:
            self.motion_char_step = CharacterizeIntraVolumeMotion(
                nifti_image_path=self.func_nifti_image_path,
                json_file_path=self.func_json_file_path,
                working_directory=configurations.WORKING_DIRECTORY_PATH,
                output_directory=configurations.OUTPUT_DIRECTORY_PATH,
                run_environment=configurations.OPTIMIZER_RUN_ENVIRONMENT,
                smsmireg_executable_path=configurations.OPTIMIZER_EXECUTABLE_PATH,
                singularity_image_path=configurations.OPTIMIZER_SINGULARITY_IMAGE_PATH,
                n_jobs=configurations.N_JOBS,
                motion_threshold=configurations.MOTION_THRESHOLD,
                percent_of_volumes_cutoff=configurations.PERCENT_OF_VOLUMES_CUTOFF,
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
        self.motion_correction_output_directory: str = os.path.join(configurations.OUTPUT_DIRECTORY_PATH, "motion_correction_outputs")
        if configurations.RUN_MOTION_CORRECTION:
            self.motion_correction_step = StartMotionCorrection(
                nifti_image_path=self.func_nifti_image_path,
                json_file_path=self.func_json_file_path,
                radian_parameters_text_file=os.path.join(configurations.OUTPUT_DIRECTORY_PATH, "radian-parameters.txt"),
                displacements_text_file=os.path.join(configurations.OUTPUT_DIRECTORY_PATH, "displacements.txt"),
                motion_threshold=self.motion_char_step.motion_threshold if configurations.RUN_MOTION_CHARACTERIZATION else configurations.MOTION_THRESHOLD,
                percent_of_volumes_cutoff=configurations.PERCENT_OF_VOLUMES_CUTOFF,
                matlab_main_script_path=configurations.MAIN_MOTION_CORRECTION_MATLAB_SCRIPT,
                noscrubbing_recon_filename_prefix=configurations.NON_SCRUBBED_DATA_FILENAME_PREFIX,
                scrubbed_recon_filename_prefix=configurations.SCRUBBED_DATA_FILENAME_PREFIX,
                output_directory_path=self.motion_correction_output_directory,
                working_directory_path=configurations.WORKING_DIRECTORY_PATH,
                dcmdjpeg_path=configurations.DCMDJPEG_PATH,
                dcm2niix_path=configurations.DCM2NIIX_PATH,
                matlab_path=configurations.MATLAB_INSTALLATION_PATH
            )
        
        """
        ========================================
        STEP THREE: FMRIPREP
        ========================================
        """
        if configurations.RUN_FMRIPREP:
            from MotionCorrection.__main__ import find_intravolume_corrected_data
            self.motion_corrected_image_path: str = find_intravolume_corrected_data(
                output_directory_path=self.motion_correction_output_directory,
                scrubbed_data_filename_prefix=configurations.SCRUBBED_DATA_FILENAME_PREFIX,
                nonscrubbed_data_filename_prefix=configurations.NON_SCRUBBED_DATA_FILENAME_PREFIX
            )
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

        """
        ========================================
        STEP FOUR: RUN POST-ANALYSIS SCRIPTS
        ========================================
        """
        if configurations.RUN_POST_PIPELINE_ANALYSES:
            self.post_pipeline_analyses(configurations)

        
    def post_pipeline_analyses(self, configurations: Configurations):

        def get_fmriprep_dir(output_directory_path: str) -> str:

            fmriprep_directories: list[str] = sorted(glob(os.path.join(output_directory_path, "fmriprep*")))
            if len(fmriprep_directories) == 0:
                raise ValueError(
                    f"No fMRIPrep Output Directories (matching: {os.path.join(output_directory_path, 'fmriprep*')}" 
                    "found in output directory.")
            elif len(fmriprep_directories) > 1:
                warnings.warn(
                    message=f"Multiple fMRIPrep Output Directories Found. Using: {fmriprep_directories[-1]}",
                    category=UserWarning
                )
            return fmriprep_directories[-1]

        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "PostPipelineAnalysis"))
        from PostPipelineAnalysis import PostRunAnalysis

        # Get the main preprocessed data directories for fMRIPrep
        fmriprep_directory: str = get_fmriprep_dir(configurations.OUTPUT_DIRECTORY_PATH)
        fmriprep_func_directory: str = glob(os.path.join(fmriprep_directory, "*", "*", "func"))[-1]
        fmriprep_anat_directory: str = glob(os.path.join(fmriprep_directory, "*", "*", "anat"))[-1]

        # If the RUN_MOTION_CHARACTERIZATION or RUN_MOTION_CORRECTION steps 
        # already got a threshold, or if one was given in the .yaml file, 
        # just use that. Else, calculate the threshold here 
        motion_threshold: float = 0.6
        if configurations.MOTION_THRESHOLD != None:
            motion_threshold: float = configurations.MOTION_THRESHOLD
        elif configurations.RUN_MOTION_CHARACTERIZATION:
            motion_threshold: float = self.motion_char_step.motion_threshold
        elif configurations.RUN_MOTION_CORRECTION:
            motion_threshold: float = self.motion_correction_step.motion_threshold
        else:
            threshold_output_directory: str = os.path.join(configurations.OUTPUT_DIRECTORY_PATH, "threshold_selection_outputs")
            motion_threshold: float = DetermineMotionThreshold(
                displacements_text_file=os.path.join(configurations.OUTPUT_DIRECTORY_PATH, "displacements.txt"),
                nifti_image_path=self.func_nifti_image_path,
                json_file_path=self.func_json_file_path,
                output_directory=threshold_output_directory,
                percent_of_volumes_cutoff=configurations.PERCENT_OF_VOLUMES_CUTOFF
            ).return_motion_threshold()
            print(f"Recevied Motion Threshold of: {motion_threshold} mm")
            print(f"See Threshold Selection Plots at: {threshold_output_directory}")

        # If it doesnt already exist, find the motion corrected image via
        # MotionCorrection's func: 'find_intravolume_corrected_data()'
        from MotionCorrection.__main__ import find_intravolume_corrected_data
        self.motion_corrected_image_path: str = find_intravolume_corrected_data(
            output_directory_path=self.motion_correction_output_directory,
            scrubbed_data_filename_prefix=configurations.SCRUBBED_DATA_FILENAME_PREFIX,
            nonscrubbed_data_filename_prefix=configurations.NON_SCRUBBED_DATA_FILENAME_PREFIX
        )

        # Make sure all inputs exist
        inputs: dict = {
            "raw_func_image": self.func_nifti_image_path,
            "motion_corrected_func_image": self.motion_corrected_image_path,
            "json_file_path": self.func_json_file_path,
            "transform_directory": os.path.join(configurations.OUTPUT_DIRECTORY_PATH, "transforms"),
            "fmriprep_func_image": glob(os.path.join(fmriprep_func_directory, f"*run-{'{:02d}'.format(configurations.RUN_NUM)}_desc-preproc_bold.nii.gz"))[-1],
            "fmriprep_func_image_mask": glob(os.path.join(fmriprep_func_directory, f"*run-{'{:02d}'.format(configurations.RUN_NUM)}_desc-brain_mask.nii.gz"))[-1],
            "fmriprep_confounds_path": glob(os.path.join(fmriprep_func_directory, f"*run-{'{:02d}'.format(configurations.RUN_NUM)}_desc-confounds_timeseries.tsv"))[-1],
            "fmriprep_anat_image": glob(os.path.join(fmriprep_anat_directory, f"sub-{'{:02d}'.format(configurations.SUBJECT_ID)}" + f"_ses-{'{:02d}'.format(configurations.SESSION_NUM)}" + "_desc-preproc_T1w.nii.gz"))[-1],
            "reference_volume_path": sorted(glob(os.path.join(configurations.OUTPUT_DIRECTORY_PATH, "refvol*.nii")))[-1]
        }
        for file_key, file_path in inputs.items():
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Could not find file path for '{file_key}': {file_path}"
                )
            
        PostRunAnalysis(
            raw_func_nifti_image=inputs['raw_func_image'],
            corrected_func_nifti_image=inputs['motion_corrected_func_image'],
            fmriprep_func_nifti_image=inputs["fmriprep_func_image"],
            json_file=inputs['json_file_path'],
            brain_mask_path=inputs["fmriprep_func_image_mask"],
            anatomical_nifti_image=inputs["fmriprep_anat_image"],
            confounds_file_path=inputs['fmriprep_confounds_path'],
            reference_volume_path=inputs['reference_volume_path'],
            transform_directory=inputs['transform_directory'],
            head_radius=configurations.HEAD_RADIUS,
            mm_displacement_threshold=motion_threshold,
            working_directory=configurations.WORKING_DIRECTORY_PATH,
            output_directory=os.path.join(configurations.OUTPUT_DIRECTORY_PATH, "post-pipeline_analyses")
        )

if __name__ == "__main__":
    import argparse
    parser: argparse.ArgumentParser =  argparse.ArgumentParser(
        description="This script is the wrapper for running the whole intravolume motion pipeline."
    )
    parser.add_argument("--configuration_file", required=True, help=".yaml configuration file")
    args: argparse.Namespace = parser.parse_args()
    RunPipeline(configuration_file=os.path.abspath(args.configuration_file))