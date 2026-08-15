import yaml
from pathlib import Path
from typing import Literal, Sequence
from pydantic import BaseModel, model_validator, Field, PositiveInt

class Configurations(BaseModel):
    """
    This script manages and validates the values of the input .yaml 
    configuration files using the 'pydantic' module.
    """
    RUN_MOTION_CHARACTERIZATION: bool
    RUN_MOTION_CORRECTION: bool
    RUN_FMRIPREP: bool

    WORKING_DIRECTORY_PATH: str
    OUTPUT_DIRECTORY_PATH: str

    FUNCTIONAL_DICOM_DIRECTORY: str | None
    FUNCTIONAL_NIFTI_IMAGE_PATH: str | None
    FUNCTIONAL_JSON_FILE_PATH: str | None

    SERIES_NAME: str | None

    ANATOMICAL_DICOM_DIRECTORY: str | None
    ANATOMICAL_NIFTI_IMAGE_PATH: str | None
    ANATOMICAL_JSON_FILE_PATH: str | None

    SUBJECT_ID: int | None
    SESSION_NUM: int | None 
    RUN_NUM: int | None

    OPTIMIZER_RUN_ENVIRONMENT: Literal[
        "singularity",
        "docker",
        "local"
    ]
    OPTIMIZER_EXECUTABLE_PATH: str | None 
    OPTIMIZER_SINGULARITY_IMAGE_PATH: str | None

    MCORR_OUTPUT_FILENAME_PATTERN: str
    MCORR_ABRUPTMOTION_FILE_NAME: str 

    CONDA_ENV_NAME: str
    CONDA_INIT_PATH: str
    CONDA_ENV_PYTHON_PATH: str
    MATLAB_INSTALLATION_PATH: str
    DCM2NIIX_PATH: str
    BIOGRIDS_PATH: str
    DCMDJPEG_PATH: str
    FSLDIR_PATH: str
    FMRIPREP_CONTAINER_PATH: str
    FMRIPREP_TEMPLATEFLOW_DIRECTORY: str
    FMRIPREP_LICENSE_PATH: str

    N_JOBS: int

    MOTION_THRESHOLD: int = Field(ge=0, le=100)

    REFERENCE_VOLUME_INDEX: int | None

    UPSAMPLE_REFERENCE_VOLUME: bool
    REFERENCE_VOLUME_SPACING: Sequence[float] | None

    LIMIT_VOXEL_INTENSITY: bool
    VOXEL_LOWER_BOUND: int | None 
    VOXEL_UPPER_BOUND: int | None 

    HEAD_RADIUS: int 

    OMP_NTHREADS: int | None
    MEM_MB: int | None

    RUN_POST_PIPELINE_ANALYSIS: bool

    @model_validator(mode='after')
    def validate_functional_inputs(self):
        has_dicom: bool = self.FUNCTIONAL_DICOM_DIRECTORY is not None
        has_nifti: bool = self.FUNCTIONAL_NIFTI_IMAGE_PATH is not None
        has_json: bool = self.FUNCTIONAL_JSON_FILE_PATH is not None
        if has_dicom:
            # Don't allow mixing input types
            if has_nifti or has_json:
                raise ValueError(
                    "Provide either FUNCTIONAL_DICOM_DIRECTORY "
                    "or both FUNCTIONAL_NIFTI_IMAGE_PATH and "
                    "FUNCTIONAL_JSON_FILE_PATH, not both."
                )
            elif not self.SERIES_NAME:
                raise ValueError(
                    "Provide SERIES_NAME if you are using FUNCTIONAL_DICOM_DIRECTORY"
                )
        else:
            # No DICOM directory, so require both files
            if not (has_nifti and has_json):
                raise ValueError(
                    "Either provide FUNCTIONAL_DICOM_DIRECTORY, "
                    "or provide BOTH FUNCTIONAL_NIFTI_IMAGE_PATH "
                    "and FUNCTIONAL_JSON_FILE_PATH."
                )

            if not self.FUNCTIONAL_NIFTI_IMAGE_PATH.endswith(".nii.gz") and not self.FUNCTIONAL_NIFTI_IMAGE_PATH.endswith(".nii"):  # pyright: ignore[reportOptionalMemberAccess]
                raise ValueError(
                    "FUNCTIONAL_NIFTI_IMAGE_PATH must end in '.nii.gz' or '.nii'"
                )

            if not self.FUNCTIONAL_JSON_FILE_PATH.endswith(".json"): # pyright: ignore[reportOptionalMemberAccess]
                  raise ValueError(
                        "FUNCTIONAL_JSON_FILE_PATH must end in '.json'"
                    )
            

        return self

    @model_validator(mode='after')
    def validate_anatomical_inputs(self):
        if self.RUN_FMRIPREP:
            has_dicom: bool = self.ANATOMICAL_DICOM_DIRECTORY is not None
            has_nifti: bool = self.ANATOMICAL_NIFTI_IMAGE_PATH is not None
            has_json: bool = self.ANATOMICAL_JSON_FILE_PATH is not None
            if has_dicom:
                # Don't allow mixing input types
                if has_nifti or has_json:
                    raise ValueError(
                        "Provide either ANATOMICAL_DICOM_DIRECTORY "
                        "or both ANATOMICAL_NIFTI_IMAGE_PATH and "
                        "ANATOMICAL_JSON_FILE_PATH, not both."
                    )
                elif not self.SERIES_NAME:
                    raise ValueError(
                        "Provide SERIES_NAME if you are using ANATOMICAL_DICOM_DIRECTORY"
                    )
            else:
                # No DICOM directory, so require both files
                if not (has_nifti and has_json):
                    raise ValueError(
                        "Either provide ANATOMICAL_DICOM_DIRECTORY, "
                        "or provide BOTH ANATOMICAL_NIFTI_IMAGE_PATH "
                        "and ANATOMICAL_JSON_FILE_PATH."
                    )

                if not self.ANATOMICAL_NIFTI_IMAGE_PATH.endswith(".nii.gz") and not self.ANATOMICAL_NIFTI_IMAGE_PATH.endswith(".nii"):  # pyright: ignore[reportOptionalMemberAccess]
                    raise ValueError(
                        "ANATOMICAL_NIFTI_IMAGE_PATH must end in '.nii.gz' or '.nii'"
                    )

                if not self.ANATOMICAL_JSON_FILE_PATH.endswith(".json"): # pyright: ignore[reportOptionalMemberAccess]
                        raise ValueError(
                            "ANATOMICAL_JSON_FILE_PATH must end in '.json'"
                        )
        return self

    @model_validator(mode='after')
    def validate_fmriprep_inputs(self):
        if self.RUN_FMRIPREP:
            if not self.SERIES_NAME:
                raise ValueError(
                    "Provide SERIES_NAME if RUN_FMRIPREP = True."
                )
        return self

    @model_validator(mode='after')
    def validate_smsmireg_run_env(self):
        if self.OPTIMIZER_RUN_ENVIRONMENT == "singularity":
            if not self.OPTIMIZER_SINGULARITY_IMAGE_PATH:
                raise ValueError(
                    "If OPTIMIZER_RUN_ENVIRONMENT = 'singularity': "
                    "please also provide the singularity .sif path via: OPTIMIZER_SINGULARITY_IMAGE_PATH"
                )
        elif self.OPTIMIZER_RUN_ENVIRONMENT == 'local':
            if not self.OPTIMIZER_EXECUTABLE_PATH:
                raise ValueError(
                    "If OPTIMIZER_RUN_ENVIRONMENT =  'local': "
                    "please also provide the path to the compiled CPP code via: OPTIMIZER_EXECUTABLE_PATH"
                )
        return self


    @model_validator(mode='after')
    def validate_voxel_intensity_args(self):
        if self.LIMIT_VOXEL_INTENSITY:
            if not self.VOXEL_LOWER_BOUND or not self.VOXEL_UPPER_BOUND:
                raise ValueError(
                    "If LIMIT_VOXEL_INTENSITY == True, please enter integer values for both: "
                    "VOXEL_LOWER_BOUND and VOXEL_UPPER_BOUND."
                )
        return self

    @model_validator(mode='after')
    def validate_refvol_upsampling(self):
        if self.UPSAMPLE_REFERENCE_VOLUME:
            if not self.REFERENCE_VOLUME_SPACING:
                raise ValueError(
                    "If UPSAMPLE_REFERENCE_VOLUME = True, "
                    "give the new spacing as 3 float values (in mm) "
                    "via: REFERENCE_VOLUME_SPACING"
                )
            elif len(self.REFERENCE_VOLUME_SPACING) != 3:
                raise ValueError(
                    "If UPSAMPLE_REFERENCE_VOLUME = True, "
                    "give the new spacing as 3 float values (in mm) "
                    "via: REFERENCE_VOLUME_SPACING"
                )
            elif not all(isinstance(new_mm_val, float) for new_mm_val in self.REFERENCE_VOLUME_SPACING):
                raise ValueError(
                    "If UPSAMPLE_REFERENCE_VOLUME = True, "
                    "give the new spacing as 3 float values (in mm) "
                    "via: REFERENCE_VOLUME_SPACING"
                )
        return self

    @model_validator(mode='after')
    def add_backslash(self):
        if self.WORKING_DIRECTORY_PATH[-1].strip() != "/":
            self.WORKING_DIRECTORY_PATH += "/"
        elif self.OUTPUT_DIRECTORY_PATH[-1].strip() != "/":
            self.OUTPUT_DIRECTORY_PATH += "/"

        return self
    @model_validator(mode='after')
    def check_for_fmriprep_ids(self):
        if self.RUN_FMRIPREP:
            if not self.SUBJECT_ID:
                raise ValueError(
                    "If you are running fMRIPrep, please enter the following variables with integer values: "
                    "SUBJECT_ID, SESSION_NUM, RUN_NUM")
            if not self.SESSION_NUM:
                raise ValueError(
                    "If you are running fMRIPrep, please enter the following variables with integer values: "
                    "SUBJECT_ID, SESSION_NUM, RUN_NUM")
            if not self.RUN_NUM:
                raise ValueError(
                    "If you are running fMRIPrep, please enter the following variables with integer values: "
                    "SUBJECT_ID, SESSION_NUM, RUN_NUM"
                )
        return self
if __name__ == "__main__":
    import argparse

    parser: argparse.ArgumentParser= argparse.ArgumentParser(
        description="Validate config json file."
    )
    parser.add_argument(
        "--config_file",
        required=True
    )
    args: argparse.Namespace = parser.parse_args()
    with open(args.config_file, mode='r') as file:
        Configurations(**yaml.safe_load(file))
