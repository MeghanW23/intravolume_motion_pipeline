import os
import pydicom
import subprocess
from glob import glob
from joblib import Parallel, delayed

class DicomToNiFTI:
    
    def __init__(
        self, 
        dicom_directory, 
        working_directory = None, 
        decompress = True, 
        dcmdjpeg_path = 'dcmdjpeg', 
        dcm2niix_path = 'dcm2niix', 
        target_sequence_number = None, 
        series_name = None, 
        serialize = False):
        
        if not working_directory:
            working_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working")
        os.makedirs(working_directory, exist_ok=True)
        
        dcm2niix_input_directory = dicom_directory
        if decompress:
            print(f"Decompressing DICOMs at: {dicom_directory}")
            DecompressDicoms(
                dicom_directory=dicom_directory,
                output_directory=working_directory,
                series_name=series_name,
                dcmdjpeg_path=dcmdjpeg_path,
                n_jobs = None if serialize else -1
            )
            dcm2niix_input_directory = working_directory

        dcm2niix_commmand = [
            dcm2niix_path, 
            "-o", working_directory, 
            "-z", "y", 
            "-b","y",
            "-ba","n", 
            "-w", "1", 
            dcm2niix_input_directory
        ]
        print(f"Running dcm2niix: {dcm2niix_commmand}")
        subprocess.run(dcm2niix_commmand)

        self.found_nifti_image = self.get_file(".nii.gz", working_directory, series_name, target_sequence_number)
        self.found_json_file = self.get_file(".json", working_directory, series_name, target_sequence_number)

        print("Done with dcm2niix.")

    
    def get_file(self, file_extension, directory, series_name = None, target_sequence_number = None):
        
        if series_name:
            series_name = series_name.lower().strip()
        
        if target_sequence_number:
            target_sequence_number = str(target_sequence_number).strip()
        

        for filename in os.listdir(directory):
            filename_lowercase = filename.lower().strip()

            if not filename_lowercase.endswith(file_extension):
                continue 

            if not "func-bold_task" in filename:
                continue 

            elif series_name and target_sequence_number:
                if series_name in filename_lowercase and filename_lowercase.endswith(f"_{target_sequence_number}{file_extension}"):
                    return os.path.join(directory, filename)

            elif series_name and series_name in filename_lowercase:
                return os.path.join(directory, filename)

            elif target_sequence_number and filename_lowercase.endswith(f"_{target_sequence_number}{file_extension}"):
                return os.path.join(directory, filename)
            
            else:
                continue 
        
        print("\nERROR: Could not find any files matching these criteria after dcm2niix:")
        print(f"Inputted file extension: {file_extension}")
        print(f"Inputted parent directory: {directory}")
        print(f"Inputted series name: {series_name}")
        print(f"Inputted target sequence number: {target_sequence_number}")
        exit(1)
        
    def return_nifti_image(self):
        return self.found_nifti_image
    
    def return_json_file(self):
        return self.found_json_file


class DecompressDicoms:
    
    def __init__(self, dicom_directory, output_directory = None, dcmdjpeg_path = 'dcmdjpeg', series_name = None, n_jobs = -1):

        print(f"Running With {n_jobs} Jobs")

        if not output_directory:
            output_directory = os.path.join(dicom_directory, "decompressed_dicoms")
        os.makedirs(output_directory, exist_ok=True)

        if series_name:
            series_name = str(series_name.strip())
            print(f"Decompressing DICOMS from Series: {series_name}")

        all_dicoms = sorted(glob(os.path.join(dicom_directory, "*.dcm")))

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


    def is_in_series(self, series_string, dicom_path, stdout_string):
        
        try:
            print(f"{stdout_string} Series Data:" + str(pydicom.dcmread(dicom_path)['SeriesDescription']).split("LO:")[-1].strip())
        
            return series_string.strip().lower() in str(pydicom.dcmread(dicom_path)['SeriesDescription']).strip().lower()
        
        except pydicom.errors.InvalidDicomError:
            
            print(f"\nWARNING: File {os.path.basename(dicom_path)} is missing DICOM File Meta Information header or the 'DICM' prefix is missing from the header. Use force=True to force reading.")
            print(f"This DICOM will not be included\n.")
            
            return False 

    def decompress_dicom(self, i, dicom_path, num_dicoms, output_dir, series_name, dcmdjpeg_path = 'dcmdjpeg'):
        
        stdout_string = f"File {'{:04d}'.format(i)} of {num_dicoms} - {os.path.basename(dicom_path)} -"
        print(f"{stdout_string} Starting Now...")

        if series_name:
            
            print(f"{stdout_string} Checking if Dicom is in Series...")
            
            if not self.is_in_series(dicom_path=dicom_path, series_string=series_name, stdout_string=stdout_string):
                
                print(f"{stdout_string} DICOM not in Series. Skipping Dicom.")
                
                return 

        decompression_command = [dcmdjpeg_path, dicom_path, os.path.join(output_dir, f"{os.path.basename(dicom_path).replace('.dcm', '')}_decompressed.dcm")]
        print(f"{stdout_string} Decompressing the DICOM via: {decompression_command}")

        result = subprocess.run(decompression_command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR Decompressing DICOM Via: {decompression_command}")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            exit(0)
        else:
            print(f"{stdout_string} Decompression Sucessful.")

            
if __name__ == '__main__':
         
    default_working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working")
    
    import argparse 
    parser = argparse.ArgumentParser(description="To decompress and run dcm2niix on a DICOM directory." )
    parser.add_argument(
        "--dicom_directory",
        required=True
    )
    parser.add_argument(
        "--working_directory", 
        required=False, 
        default=default_working_dir,
        help=f"Default location: {default_working_dir}"
    )
    parser.add_argument(
        "--series_name",
        required=False,
        default=None,
        help="To only decompress and dcm2niix on one series/task. NOTE: must match the value of the DICOM metadata key: 'SeriesDescription'."
    )
    parser.add_argument(
        "--target_sequence_number",
        required=False,
        type=int,
        default=None,
        help="To only decompress and dcm2niix on the series/task matching this target_sequence_number in the filename. Ex: '15' for dicom_dir_func-bold_task-preRIFG_20250827181314_15.nii.gz"
    )
    parser.add_argument(
        "--decompress",
        action='store_true'
    )
    args = parser.parse_args()

    DicomToNiFTI(
        dicom_directory=os.path.abspath(args.dicom_directory),
        working_directory=os.path.abspath(args.working_directory),
        decompress=args.decompress,
        series_name=args.series_name,
        target_sequence_number=args.target_sequence_number
        
    )
