import os 
import SimpleITK as sitk


class MakeIdentityTransform:
    def __init__(self, 
                 input_nifti_image: str | sitk.Image, 
                 output_file_path: str | None = None) -> None:
        
        sitk_image: sitk.Image = sitk.ReadImage(input_nifti_image) if isinstance(input_nifti_image, str) else input_nifti_image
        if sitk_image.GetDimension() != 3:
            raise ValueError(f"The input NiFTI image must be 3D. Your image has {sitk_image.GetDimension()} dimensions.")

        image_center: tuple[float, float, float] = sitk_image.TransformContinuousIndexToPhysicalPoint([
            (index - 1) / 2.0 
            for index in sitk_image.GetSize()
        ])
        print(f"Rotation Center: {image_center}")

        self.transform: sitk.AffineTransform = sitk.AffineTransform(3)
        self.transform.SetIdentity()
        self.transform.SetCenter(image_center)

        if output_file_path:
            sitk.WriteTransform(
                self.transform,
                output_file_path
            )

    def return_loaded_transform(self) -> sitk.AffineTransform:
        return self.transform


if __name__ == "__main__":
    import argparse
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description= \
            "Create a SimpleITK AffineTransform identity transform " + \
            "at the center of an input 3D NiFTI volume." 
    )
    parser.add_argument(
        "--input_nifti_image_path",
        required=True,
        help="Must be a 3D NiFTI image."
    )
    parser.add_argument(
        "--output_file_path",
        required=False,
        default=None,
        help="Do not save to file: None. Default: None."
    )
    args: argparse.Namespace = parser.parse_args()
    MakeIdentityTransform(
        input_nifti_image=os.path.abspath(args.input_nifti_image_path),
        output_file_path=\
            os.path.abspath(args.output_file_path) 
            if args.output_file_path else None
    )