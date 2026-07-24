import os 
import pydicom
import warnings
import subprocess
from glob import glob 
from joblib import Parallel, delayed

class DecompressDicoms:
    
    def __init__(self, 
                 dicom_directory: str, 
                 output_directory: str | None = None,  # pyright: ignore[reportRedeclaration]
                 dcmdjpeg_path = 'dcmdjpeg',
                 series_name: str | None = None, # pyright: ignore[reportRedeclaration]
                 n_jobs: int = -1):

        print(f"Decompressing DICOM Directory: {dicom_directory} With {n_jobs} Jobs")
        
        output_directory: str = os.path.join(dicom_directory, "decompressed_dicoms") if not output_directory else output_directory
        os.makedirs(output_directory, exist_ok=True)
        print(f"Output Directory: {output_directory}")

        if series_name:
            series_name: str = str(series_name.strip())
            print(f"Decompressing DICOMS from Series: {series_name}")

        all_dicoms: list[str] = sorted(glob(os.path.join(dicom_directory, "*.dcm")))
        print(f"DICOMs in DICOM Directory: {len(all_dicoms)}")
        if len(all_dicoms) == 0:
            raise FileNotFoundError(f"\nERROR: No files ending in *.dcm in directory: {dicom_directory}\n")

        Parallel(n_jobs=n_jobs)(
            delayed(self.decompress_dicom)(
                i=i,
                dicom_path=dicom_path,
                num_dicoms=len(all_dicoms),
                output_dir=output_directory,
                series_name=series_name,
                dcmdjpeg_path=dcmdjpeg_path
            )
            for i, dicom_path in enumerate(all_dicoms, start=1)
        )
        print(f"Done. Compressed Dicoms at: {output_directory}")


    def is_in_series(self, 
                     series_string: str, 
                     dicom_path: str, 
                     stdout_string: str) -> bool:
        
        try:
            print(f"{stdout_string} Series Data:" + str(pydicom.dcmread(dicom_path)['SeriesDescription']).split("LO:")[-1].strip())
        
            return series_string.strip().lower() in str(pydicom.dcmread(dicom_path)['SeriesDescription']).strip().lower()
        
        except pydicom.errors.InvalidDicomError: # pyright: ignore[reportAttributeAccessIssue]

            warnings.warn(
                message=(
                    f"File {os.path.basename(dicom_path)} is missing DICOM File "
                    "Metadata Information header or the 'DICM' prefix is missing "
                    "from the header. Use force=True to force reading. "
                    "This DICOM will not be included."
                ),
                category=UserWarning
            )
            
            return False 


    def decompress_dicom(self, i: int, dicom_path: str, num_dicoms: int, 
                         output_dir: str, series_name: str, dcmdjpeg_path: str = 'dcmdjpeg'):
        
        stdout_string: str = f"File {'{:04d}'.format(i)} of {num_dicoms} - {os.path.basename(dicom_path)} -"
        print(f"{stdout_string} Starting Now...")

        if series_name:
            
            print(f"{stdout_string} Checking if Dicom is in Series...")
            
            if not self.is_in_series(dicom_path=dicom_path, series_string=series_name, stdout_string=stdout_string):
                
                print(f"{stdout_string} DICOM not in Series. Skipping Dicom.")
                
                return 

        decompression_command: list[str] = [dcmdjpeg_path, dicom_path, os.path.join(output_dir, f"{os.path.basename(dicom_path).replace('.dcm', '')}_decompressed.dcm")]
        print(f"{stdout_string} Decompressing the DICOM via: {decompression_command}")

        result: subprocess.CompletedProcess = subprocess.run(decompression_command, capture_output=True, text=True)
        if result.returncode != 0:
            raise CalledProcessError(
                returncode=result.returncode,
                cmd=result.args,
                output=result.stdout,
                stderr=result.stderr
            )
        else:
            print(f"{stdout_string} Decompression Sucessful.")

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
        description="Decompress a DICOM Directory."
    )
    parser.add_argument(
        "--dicom_directory",
        required=True
    )
    parser.add_argument(
        "--output_directory",
        required=False,
        default='outputs',
        help=f"Default: {os.path.abspath('outputs')}."
    )
    parser.add_argument(
        "--dcmdjpeg_path",
        required=False,
        default='dcmdjpeg',
        help="The path or terminal command for the software 'dcmdjpeg'. Default = 'dcmdjpeg'."
    )
    parser.add_argument(
        "--series_name",
        required=False,
        default=None,
        help=\
            "If a series name is given, we will only decompress DICOMs with that series name. " + \
            "If series_name = None, we will decompress all DICOMs in the directory. " + \
            "Any inputted series name must match the string in the DICOM metadata key 'SeriesDescription' " + \
            "(not case sensitive)."
    )
    parser.add_argument(
        "--n_jobs",
        required=False,
        default=-1,
        type=int,
        help="Number of DICOMs to decompress in parallel. " + \
             "-1 = Run a DICOM per CPU core. " + \
             "Default: -1."
    )
    args: argparse.Namespace = parser.parse_args()
    DecompressDicoms(
        dicom_directory = os.path.abspath(args.dicom_directory),
        output_directory = \
            os.path.abspath(args.output_directory) \
            if args.output_directory else None,
        dcmdjpeg_path=args.dcmdjpeg_path,
        series_name=args.series_name,
        n_jobs=args.n_jobs
    )