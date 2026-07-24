import os 
import math 
import SimpleITK as sitk 

class GetMotionThreshold:

    def __init__(self, 
                 nifti_image: str | sitk.Image, 
                 threshold_as_percent: float = 10) -> None:
        
        if not isinstance(threshold_as_percent, float | int):
            raise ValueError("'threshold_as_percent' must be an integer or float value.")
        
        if threshold_as_percent < 0 or threshold_as_percent > 100:
            raise ValueError("'threshold_as_percent' must be between 0 to 100.")

        self.mm_threshold: float = self.get_threshold_in_mm(
            nifti_image=\
                sitk.ReadImage(nifti_image) if isinstance(nifti_image, str) else nifti_image,
            threshold_as_percent=threshold_as_percent
        )
        print(f"Threshold: {self.mm_threshold} mm")

    def get_threshold_in_mm(self, 
                            nifti_image: sitk.Image,
                            threshold_as_percent: float) -> float:
        """

        To get the diagonal of a rectangular prism (3D rectangle):
        d = sqrt(dim_x^2 + dim_y^2 + dim_z^2)
        
        To get the threshold in mm:
        mm_threshold = d * (threshold_as_percent_of_voxel / 100)
        """
        spacing: tuple[float, float, float] | tuple[float, float, float] = nifti_image.GetSpacing()
        if len(spacing) < 3:
            raise ValueError("Image must be at least 3D.")

        d_x, d_y, d_z = spacing[:3]
        print(f"Voxel Spacng: {d_x, d_y, d_z}")
        
        diagonal: float = math.sqrt(d_x ** 2 + d_y ** 2 + d_z ** 2)
        print(f"Voxel diagonal: {diagonal} mm")

        return diagonal * (threshold_as_percent / 100)

    def return_mm_threshold(self) -> float:
        return self.mm_threshold
    
if __name__ == "__main__":
    import argparse
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=\
            "Calculate the motion threshold in mm given the " + \
            "desired percent threshold of the diagonal of a single voxel."
    )
    parser.add_argument(
        "--nifti_image_file_path",
        required=True,
        help="The 3D or 4D input NiFTI image."
    )
    parser.add_argument(
        "--threshold_as_percent",
        type=float,
        default=10,
        required=False,
        help=\
            "The threshold as the percent of the diagonal of a single voxel. " +\
            "Default = 10 Percent."
    )
    args: argparse.Namespace = parser.parse_args()
    GetMotionThreshold(
        nifti_image=os.path.abspath(args.nifti_image_file_path),
        threshold_as_percent=args.threshold_as_percent
    )