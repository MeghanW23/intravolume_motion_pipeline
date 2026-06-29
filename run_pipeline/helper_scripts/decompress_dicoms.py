import os
import joblib 
import pydicom 
import subprocess 
from glob import glob

class DecompressDicoms:
    
    def __init__(self, dicom_directory, output_directory = None, series_name = None, n_jobs = -1):

        print(f"Running With {n_jobs} Jobs")

        if not output_directory:
            output_directory = os.path.join(dicom_directory, "decompressed_dicoms")
        os.makedirs(output_directory, exist_ok=True)
        print(f"Making Output Directory: {output_directory}")

        if series_name:
            series_name = str(series_name.strip())
            print(f"Decompressing DICOMS from Series: {series_name}")

        all_dicoms = sorted(glob(os.path.join(dicom_directory, "*.dcm")))

        joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(self.decompress_dicom)(
                i=i,
                dicom_path=dicom_path,
                num_dicoms=len(all_dicoms),
                output_dir=output_directory,
                series_name=series_name
            )
            for i, dicom_path in enumerate(all_dicoms, start=1)
        )
        print(f"Done. Compressed Dicoms at: {output_directory}")


    def is_in_series(self, series_string, dicom_path,stdout_string):
        
        print(f"{stdout_string} Series Data:" + str(pydicom.dcmread(dicom_path)['SeriesDescription']).split("LO:")[-1].strip())
        
        return series_string in str(pydicom.dcmread(dicom_path)['SeriesDescription'])


    def decompress_dicom(self, i, dicom_path, num_dicoms, output_dir, series_name):
        
        stdout_string = f"File {'{:04d}'.format(i)} of {num_dicoms} - {os.path.basename(dicom_path)} -"
        print(f"{stdout_string} Starting Now...")

        if series_name:
            
            print(f"{stdout_string} Checking if Dicom is in Series...")
            
            if not self.is_in_series(dicom_path=dicom_path, series_string=series_name, stdout_string=stdout_string):
                
                print(f"{stdout_string} DICOM not in Series. Skipping Dicom.")
                
                return 

        print(f"{stdout_string} Decompressing the DICOM")
        subprocess.run([
            'dcmdjpeg', dicom_path, os.path.join(output_dir, f"{os.path.basename(dicom_path).replace('.dcm', '')}_decompressed.dcm")
        ])
        print(f"{stdout_string} Decompression Sucessful.")


if __name__ == '__main__':
    
    import argparse
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Decompress a DICOM directory (using dcmdjpeg) and parallelize the conversions (via Joblib)")
    parser.add_argument("--dicom_directory", required=True)
    parser.add_argument("--output_directory", required=False)
    parser.add_argument("--series_name", required=False)
    parser.add_argument("--n_jobs", required=False, type=int, default=-1)
    args: argparse.Namespace = parser.parse_args()

    DecompressDicoms(
        dicom_directory=os.path.abspath(args.dicom_directory),
        output_directory=os.path.abspath(args.output_directory) if args.output_directory else None,
        series_name=args.series_name,
        n_jobs=args.n_jobs
    )


