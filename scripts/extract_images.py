import os
import SimpleITK as sitk 
from joblib import Parallel, delayed


class ExtractNiFTIImage:
    def __init__(self, 
                 input_nifti_image_path: str,
                 output_directory_path: str | None = None,
                 file_prefix: str = 'output',
                 index_to_extract: int = -1,
                 n_jobs: int = -1): # pyright: ignore[reportRedeclaration]

        dimensions: tuple[int, ...] = sitk.ReadImage(input_nifti_image_path).GetSize()
        print(f"Input Image Dimensions: {dimensions}")
        num_dimensions: int = len([
            dimension_num
            for dimension_num in dimensions
            if dimension_num > 1
        ])

        if output_directory_path:
            print(f"Saving image(s) to: {output_directory_path}")
            os.makedirs(output_directory_path, exist_ok=True)
        
        if num_dimensions == 4:
            volumes_to_extract = list(range(dimensions[-1])) if index_to_extract == -1 else [index_to_extract]
            print(f"Extracting {len(volumes_to_extract)} 3D Volume(s) from 4D Timeseries with {n_jobs} Jobs: {volumes_to_extract}")
            
            if index_to_extract != -1 and index_to_extract >= dimensions[-1]:
                raise ValueError(
                    f"Inputted (zero-indexed) 3D volume index ({index_to_extract}) is greater than the number of volumes ({dimensions[-1]})."
                )

            self.images: list[str] | list[tuple[int, sitk.Image]] = Parallel(
                n_jobs=n_jobs, verbose=11, return_as="list")(
                delayed(self.extract_single_volume)(
                    volume_num,
                    input_nifti_image_path=input_nifti_image_path,
                    output_directory_path=output_directory_path,
                    file_prefix=file_prefix
                )
                for volume_num in volumes_to_extract
            ) # pyright: ignore[reportAttributeAccessIssue]
            self.images.sort()

        
        elif num_dimensions == 3:
            slices_to_extract: list[int] = list(range(dimensions[-1])) if index_to_extract == -1 else [index_to_extract]
            print(f"Extracting {len(slices_to_extract)} Slice(s): {slices_to_extract} with {n_jobs} jobs")

            if index_to_extract != -1 and index_to_extract >= dimensions[-1]:
                raise ValueError(
                    f"Inputted (zero-indexed) 2D slice index ({index_to_extract}) is greater than the number of slices ({dimensions[-1]})."
                )
            self.images: list[str] | list[tuple[int, sitk.Image]] = Parallel(
                n_jobs=n_jobs, verbose=11, return_as="list")(
                delayed(self.extract_single_slice)(
                    slice_num,
                    input_nifti_image_path=input_nifti_image_path,
                    output_directory_path=output_directory_path,
                    file_prefix=file_prefix
                )
                for slice_num in slices_to_extract
            )  # pyright: ignore[reportAttributeAccessIssue]
            self.images.sort()

        else:
            raise ValueError(
                "\nERROR: The input nifti image must be 3D or 4D" +
                f" but your image has {num_dimensions} dimensions.\n"
            )
       

    def extract_single_volume(self, 
                              volume_num: int, 
                              input_nifti_image_path: str,
                              output_directory_path: str | None = None,
                              file_prefix: str = 'outputs') -> str | tuple[int, sitk.Image]:
        
        # cannot parallelize if passing a sitk.Image, must pass the path and load image
        input_nifti_image: sitk.Image = sitk.ReadImage(input_nifti_image_path)
        
        timeseries_dimensions: tuple[int, int, int, int] = input_nifti_image.GetSize()

        extractor: sitk.ExtractImageFilter = sitk.ExtractImageFilter()
        extractor.SetSize([
            timeseries_dimensions[0],
            timeseries_dimensions[1],
            timeseries_dimensions[2],
            0
        ])
        extractor.SetIndex([0, 0, 0, volume_num])

        volume_image: sitk.Image = extractor.Execute(input_nifti_image)

        if output_directory_path:
            output_file_path: str = os.path.join(output_directory_path, f"{file_prefix}-{'{:04d}'.format(volume_num)}.nii")
            sitk.WriteImage(
                volume_image,
                fileName=output_file_path
            )
            return output_file_path
        else:
            return (volume_num, volume_image)


    def extract_single_slice(self, 
                             slice_num: int, 
                             input_nifti_image_path: str, 
                             output_directory_path: str | None = None,
                             file_prefix: str = 'outputs') -> str | tuple[int, sitk.Image]:
        
        # cannot parallelize if passing a sitk.Image, must pass the path and load image
        input_nifti_image: sitk.Image = sitk.ReadImage(input_nifti_image_path)

        volume_dimensions: tuple[int, int, int] = input_nifti_image.GetSize()
        
        slice_image: sitk.Image = sitk.RegionOfInterest(
            input_nifti_image,
            size=[volume_dimensions[0], volume_dimensions[1], 1],
            index=[0, 0, slice_num]
        )
        
        if output_directory_path:
            output_file_path: str = os.path.join(output_directory_path, f"{file_prefix}-{'{:03d}'.format(slice_num)}.nii")
            sitk.WriteImage(
                slice_image,
                fileName=output_file_path
            )
            return output_file_path
        else:
            return (slice_num, slice_image)
    

    def return_images(self) -> list[str] | list[tuple[int, sitk.Image]]:
        return self.images 


if __name__ == "__main__":
    import argparse
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=\
            "Extract 3D Volumes or 2D Slices from " + \
            "an Input 3D or 4D NiFTI Image. " + \
            "If your image is 4D, we will extract 3D volumes. "  + \
            "If your image is 3D, we will extract 2D slices."
    )
    parser.add_argument(
        "--input_nifti_image_path",
        required=True,
        help="Must be either 3D or 4D."
    )
    parser.add_argument(
        "--output_directory_path",
        required=False,
        default=None,
        help="Default = None: Will only save as SimpleITK images in memory."
    )
    parser.add_argument(
        "--file_prefix",
        required=False,
        default='output',
        help="If saving to file, this is the filename prefix. Default: 'output'." \
        " Ex. 'output-<index_to_extract>.nii'."
    )
    parser.add_argument(
        "--index_to_extract",
        required=False,
        default=-1,
        type=int,
        help=\
            "The index of the 3D or 2D image to extract." + \
            " All Indices = -1." + \
            " Default = -1."
    )
    parser.add_argument(
        "--n_jobs",
        required=False,
        default=-1,
        type=int,
        help=\
            "Number of extractions to run in parallel. " + \
            "All CPU cores = -1. Default = -1." 
    )
    args: argparse.Namespace = parser.parse_args()
    ExtractNiFTIImage(
        input_nifti_image_path = os.path.abspath(args.input_nifti_image_path),
        output_directory_path = \
                os.path.abspath(args.output_directory_path)
                if args.output_directory_path else None,
        file_prefix=args.file_prefix,
        index_to_extract = args.index_to_extract,
        n_jobs = args.n_jobs
    )
