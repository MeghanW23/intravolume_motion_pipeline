from glob import glob
import subprocess
import os
from decompress_dicoms import DecompressDicoms

class DecompressAndDcm2niix:
    
    def __init__(self, dicom_directory, target_sequence_number = None, working_directory = None):
        
        if not working_directory:
            working_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working")
        os.makedirs(working_directory, exist_ok=True)

        print(f"Processing Data at: {dicom_directory}")
        print(f"Working Directory: {working_directory}")

        # usage: parallel_decompress.py [-h] --dicom_dir DICOM_DIR [--output_dir OUTPUT_DIR] [--n_jobs N_JOBS] [--series_name SERIES_NAME]
        print(f"Decompressing DICOMs using parallel_decompress.py")
        DecompressDicoms(
            dicom_directory=dicom_directory,
            output_directory=working_directory
        )

        print(f"Running dcm2niix")
        subprocess.run([
            "dcm2niix", dicom_directory, ".", "-z", "y", "-b","y","-ba","n", "-w", "1", working_directory
        ])

        print("Done.")

            
if __name__ == '__main__':
         
    default_working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working")
    
    import argparse 
    parser = argparse.ArgumentParser(description="To decompress and run dcm2niix on a DICOM directory." )
    parser.add_argument("--dicom_directory", 
                        required=True)
    parser.add_argument("--working_directory", 
                        required=False, 
                        default=default_working_dir, 
                        help=f"Default location: {default_working_dir}")
    args = parser.parse_args()

    DecompressAndDcm2niix(
        dicom_directory=os.path.abspath(args.dicom_directory),
        working_directory=os.path.abspath(args.working_directory)
    )
