import os
import numpy as np 
import SimpleITK as sitk 

class UpsampleReferenceVolume: 
    def __init__(self, 
                 input_nifti_image: str | sitk.Image,
                 output_file_path: str | None = None,
                 new_spacing: tuple[float, float, float] = (1.236, 1.236, 1.236)) -> None:
        
        sitk_image: sitk.Image = sitk.ReadImage(input_nifti_image) if isinstance(input_nifti_image, str) else input_nifti_image
        if sitk_image.GetDimension() != 3:
            raise ValueError(f"The input NiFTI image must be 3D. Your image has {sitk_image.GetDimension()} dimensions.")

        if any(s <= 0 for s in new_spacing):
            raise ValueError("All spacing values must be positive.")

        new_size: np.ndarray = self.get_new_size(
            sitk_image,
            new_spacing=new_spacing
        )

        print(f"Upsampling with Spacing: {new_spacing}")
        self.resampled_image: sitk.Image = self.resample_image(
            sitk_image,
            spacing=new_spacing,
            volume_size=new_size.tolist()
        )

        if output_file_path:
            sitk.WriteImage(
                self.resampled_image,
                fileName=output_file_path
            )
            print(f"Upsampled Volume at: {output_file_path}")
       
        
    def resample_image(self,
                       image: sitk.Image, 
                       spacing: tuple[float, float, float],
                       volume_size: list[int],
                       interpolator: int = sitk.sitkLinear) -> sitk.Image:
        
        r: sitk.ResampleImageFilter = sitk.ResampleImageFilter()
        r.SetInterpolator(interpolator)
        r.SetOutputPixelType( image.GetPixelID() )
        r.SetDefaultPixelValue(0)
        r.SetOutputOrigin(image.GetOrigin())
        r.SetOutputSpacing(spacing)
        r.SetOutputDirection(image.GetDirection())
        r.SetSize(volume_size)

        return r.Execute(image)
    

    def get_new_size(self,
                     image: sitk.Image,
                     new_spacing: tuple[float, float, float]) -> np.ndarray:
        
        spacing: np.ndarray = np.array(image.GetSpacing())
        size: np.ndarray = np.array(image.GetSize())
        new_size: np.ndarray = np.floor(spacing / new_spacing * size).astype(np.uint32)
        new_size: np.ndarray = 2*np.floor((new_size+1)/2).astype(np.uint32)
        print(f"New Size: {new_size}")

        return new_size


    def return_resampled_image(self) -> sitk.Image:
        return self.resampled_image
    

if __name__ == "__main__":
    import argparse
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description= "Upsample a 3D NiFTI image."
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
        help="If you don't want to save to file: None. Default: None."
    )
    parser.add_argument(
        "--new_spacing",
        required=False,
        type=float,
        nargs=3,
        default=(1.236, 1.236, 1.236),
        help="mm voxel size in X, Y, and Z. Default: (1.236, 1.236, 1.236)."
    )
    args: argparse.Namespace = parser.parse_args()
    UpsampleReferenceVolume(
        input_nifti_image=os.path.abspath(args.input_nifti_image_path),
        output_file_path=\
            os.path.abspath(args.output_file_path)
            if args.output_file_path else
            None,
        new_spacing=args.new_spacing
    )