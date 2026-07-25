import os 

class SingleSubjectfMRIPrep:

    def __init__(self, 
                 dicom_directory: str, 
                 series_name: str, 
                 nifti_image_path: str, 
                 json_file_path: str,
                 output_directory: str,
                 subject_id: int,
                 session_id: int,
                 run_num: int) -> None:

        fmriprep_main_dir: str = os.path.abspath(os.path.dirname(__file__))

if __name__ == "__main__":
    import argparse
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run fMRIPrep on a Single Run + Single Participant."
    )
    parser.add_argument(
        "--d"
    )