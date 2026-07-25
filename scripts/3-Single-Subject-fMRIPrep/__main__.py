import os 
import sys 

# in order of usage:
from decompress_dicoms import DecompressDicoms
from dicom_to_nifti import DicomToNifti

class SingleSubjectfMRIPrep:

    def __init__(self, 
                 dicom_directory: str, 
                 series_name: str, 
                 nifti_image_path: str,  # pyright: ignore[reportRedeclaration]
                 json_file_path: str, # pyright: ignore[reportRedeclaration]
                 output_directory: str,
                 working_directory: str,
                 subject_id: int,
                 session_id: int,
                 run_num: int,
                 dont_clear_subject_directory: bool = True,
                 dcmdjpeg_path: str = 'dcmdjpeg',
                 dcm2niix_path: str = 'dcm2niix') -> None:

        self.validate_inputted_data(dicom_directory, nifti_image_path, json_file_path)
        
        os.makedirs(working_directory, exist_ok=True)
        os.makedirs(output_directory, exist_ok=True)

        fmriprep_main_directory: str = os.path.join(output_directory, "fMRIPrep")
        os.makedirs(fmriprep_main_directory, exist_ok=True)
        print(f"Main Directory")

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
            nifti_image_path: str = dcm2niix_module.return_nifti_image()
            json_file_path: str = dcm2niix_module.return_json_file()


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


if __name__ == "__main__":
    import argparse
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run fMRIPrep on a Single Run + Single Participant."
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
        "--working_directory_path",
        required=False,
        default='outputs',
        help=f"Default: {os.path.abspath('outputs')}"
    )
    parser.add_argument(
        "--output_directory_path",
        required=False,
        default='outputs',
        help=f"Default: {os.path.abspath('outputs')}"
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
        "--do_not_clear_subject_dir",
        action='store_true',
        help=\
            "By default, the script will clear the participant directory " + \
            "if it already exists. Use this flag to prevent this. "
    )
    args: argparse.Namespace = parser.parse_args()
    SingleSubjectfMRIPrep(
        dicom_directory=\
            os.path.abspath(args.dicom_directory)
            if args.dicom_directory else None, # pyright: ignore[reportArgumentType]
        series_name=args.series_name,
        nifti_image_path=\
            os.path.abspath(args.nifti_image_file_path)
            if args.nifti_image_file_path else None, # pyright: ignore[reportArgumentType]
        json_file_path=\
            os.path.abspath(args.json_file_path)
            if args.json_file_path else None, # pyright: ignore[reportArgumentType]
        output_directory=os.path.abspath(args.output_directory_path),
        working_directory=os.path.abspath(args.working_directory_path),
        subject_id=args.subject_id,
        session_id=args.session_num,
        run_num=args.run_num,
        dont_clear_subject_directory=args.do_not_clear_subject_dir
    )