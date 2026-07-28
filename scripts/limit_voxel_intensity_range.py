import os 
from glob import glob
import SimpleITK as sitk
from joblib import Parallel, delayed

class LimitVoxelIntensityRange:
    def __init__(self, 
                 nifti_image_paths: list[str],
                 output_directory: str = 'outputs',
                 lower_bound: int = 10,
                 upper_bound: int = 1000,
                 n_jobs: int = -1) -> None:
        
        os.makedirs(output_directory, exist_ok=True)

        print(f"Clamping images to voxel intensity range: {lower_bound} - {upper_bound}")
        Parallel(n_jobs=n_jobs)(
            delayed(self.single_voxel_clamp)(
                nifti_image_path,
                output_file_path=os.path.join(
                    output_directory,
                    f"clamped_{os.path.basename(nifti_image_path)}"
                ),
                lower_bound=lower_bound,
                upper_bound=upper_bound
            )
            for nifti_image_path in nifti_image_paths
        )
        self.output_image_paths: list[str] = sorted(glob(os.path.join(output_directory, "clamped_*")))

    def single_voxel_clamp(self, 
                           nifti_image_path: str, 
                           output_file_path: str,
                           lower_bound: int  = 10, 
                           upper_bound: int = 1000) -> None:

        nifti_image: sitk.Image = sitk.ReadImage(nifti_image_path)
        if len(nifti_image.GetSize()) != 3:
            raise ValueError(
                "All input nifti image paths must be 3D."
            )

        output_image: sitk.Image = sitk.Clamp(
            nifti_image,
            outputPixelType=sitk.sitkFloat32,
            lowerBound=lower_bound,
            upperBound=upper_bound
        )

        if output_file_path:
            sitk.WriteImage(
                output_image,
                fileName=output_file_path
            )
            print(f"Saved clamped image to: {output_file_path}") 

    def return_output_image_paths(self) -> list[str]:
        return self.output_image_paths

if __name__ == "__main__":
    import argparse
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Limit the voxel intensity range of a NiFTI image."
    )
    parser.add_argument(
        "--nifti_image_paths",
        required=True,
        nargs="+",
        help="Please enter one or more 3D NiFTI image paths."
    )
    parser.add_argument(
        "--output_directory",
        required=False,
        default='outputs',
        help=f"Default: {os.path.abspath('outputs')}"
    )
    parser.add_argument(
        "--lower_bound",
        required=False,
        type=int,
        default=10,
        help="Minimum voxel intensity. Default: 10."
    )
    parser.add_argument(
        "--upper_bound",
        required=False,
        type=int,
        default=1000,
        help="Maximum voxel intensity. Default = 1000."
    )
    args: argparse.Namespace = parser.parse_args()
    LimitVoxelIntensityRange(
        nifti_image_paths=[
            os.path.abspath(image_path)
            for image_path in args.nifti_image_paths
        ],
        output_directory=os.path.abspath(args.output_directory),
        lower_bound=args.lower_bound,
        upper_bound=args.upper_bound
    )