import os 
import joblib
from glob import glob
import SimpleITK as sitk 

class CutUpTimeseries:
    
    def __init__(self, nifti_image_path, output_directory, volume_output_file_prefix = "volume_outputs", slice_output_file_prefix = "slice_outputs", n_jobs = None):
        """
        Cut a 4d Nifti Image into 3d Volumes and 2d Slices
        """
        print("\nNOTE: This is script requires a lot of memory. Make sure you have allocated a lot of memory. If the script is very slow, there may be a memory issue.\n")
        
        os.makedirs(output_directory, exist_ok=True)
        dimensions = sitk.ReadImage(nifti_image_path).GetSize()
        num_dimensions = len([dim for dim in dimensions if dim != 1])

        print(f"Input Nifti Image: {nifti_image_path}")
        print(f"Output Directory Path: {output_directory}")
        print(f"Input Image Dimensions: {dimensions}")
        print(f"Input Image is {num_dimensions}D.")

        if num_dimensions == 4: 
            
            # Extract Volumes
            joblib.Parallel(n_jobs=n_jobs if n_jobs else  min(dimensions[3], os.cpu_count()))(
                joblib.delayed(self.extract_volume)(
                    volume_num=volume_index,
                    nifti_image_path=nifti_image_path,
                    working_directory=output_directory,
                    num_volumes=dimensions[3],
                    output_file_prefix=volume_output_file_prefix
                )
                for volume_index in range(dimensions[3])
            )

            # Extract Slices
            for volume_index, volume_path in enumerate(sorted(glob(os.path.join(output_directory, f"{volume_output_file_prefix}*")))):
                joblib.Parallel(n_jobs=n_jobs if n_jobs else min(dimensions[2], os.cpu_count()))(
                    joblib.delayed(self.extract_slices)(
                        slice_num=slice_index,
                        num_slices=dimensions[2],
                        volume_image_path=volume_path,
                        working_directory=output_directory,
                        volume_num=volume_index,
                        output_file_prefix=slice_output_file_prefix
                    )
                    for slice_index in range(dimensions[2])
                )
                
        
        elif num_dimensions == 3:  
            
            # Extract Slices              
            joblib.Parallel(n_jobs=n_jobs if n_jobs else min(dimensions[2], os.cpu_count()))(
                joblib.delayed(self.extract_slices)(
                    slice_num=slice_index,
                    num_slices=dimensions[2],
                    volume_image_path=nifti_image_path,
                    working_directory=output_directory,
                    output_file_prefix=slice_output_file_prefix
                )
                for slice_index in range(dimensions[2])
            )

        else:
            
            print(f"Your input image must be 3D or 4D.")
            exit(0)
    

    def extract_volume(self, volume_num, num_volumes, nifti_image_path, working_directory, output_file_prefix = "volume_outputs"):

        output_file = os.path.join(working_directory, f"{output_file_prefix}-{'{:04d}'.format(volume_num)}.nii")
        print(f"Extracting Volume {'{:04d}'.format(volume_num)} of {'{:04d}'.format(num_volumes)} as: {output_file}")

        nifti_image = sitk.ReadImage(nifti_image_path)
        extract = sitk.ExtractImageFilter()
        extract.SetSize(list(nifti_image.GetSize())[:3] + [0])
        extract.SetIndex((0,0,0,volume_num))
        volume_image = extract.Execute(nifti_image)

        writer = sitk.ImageFileWriter()
        writer.SetFileName(output_file)
        writer.Execute(volume_image)
        

    def extract_slices(self, slice_num, num_slices, volume_image_path, working_directory, volume_num=None, output_file_prefix = "slice_outputs"):
        
        output_file = os.path.join(working_directory, f"{output_file_prefix}-{'{:03d}'.format(slice_num)}.nii")
        if volume_num != None:
            output_file = os.path.join(working_directory, f"{output_file_prefix}-{'{:04d}'.format(volume_num)}-{'{:03d}'.format(slice_num)}.nii")
            print(f"Extracting Volume {'{:04d}'.format(volume_num)}'s Slice {'{:04d}'.format(slice_num)} of {'{:04d}'.format(num_slices)} as: {output_file}")
        else:
            print(f"Extracting Slice {'{:04d}'.format(slice_num)} of {'{:04d}'.format(num_slices)} as: {output_file}")

        volume_image = sitk.ReadImage(volume_image_path)
        slice_size = list(volume_image.GetSize())[:2] + [1]
        slice_volume = sitk.RegionOfInterest(
            volume_image, 
            slice_size, 
            (0, 0, slice_num) 
        )
        writer = sitk.ImageFileWriter()
        writer.SetFileName(output_file)
        writer.Execute(slice_volume)
        

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Cut a 4d Nifti Image into 3d Volumes and 2d Slices")
    parser.add_argument(
        "--nifti_image_path", 
        required=True, 
        help= \
            "If image is 4D, we will cut it into 3D volumes and 2D slices. " \
            "If Image is 3D, we will cut it into 3D slices."
    )
    parser.add_argument(
        "--output_directory", 
        required=True
    )
    parser.add_argument(
        "--volume_output_file_prefix",
        required=False,
        default="volume_outputs"
    )
    parser.add_argument(
        "--slice_output_file_prefix",
        required=False,
        default="slice_outputs"
    )
    args = parser.parse_args()

    CutUpTimeseries(
        nifti_image_path=os.path.abspath(args.nifti_image_path),
        output_directory=os.path.abspath(args.output_directory),
        volume_output_file_prefix=args.volume_output_file_prefix,
        slice_output_file_prefix=args.slice_output_file_prefix
    )
