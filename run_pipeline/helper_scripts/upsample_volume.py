import os
import subprocess
import numpy as np
import SimpleITK as sitk 

class UpsampleVolume:
    
    def __init__(self, nifti_image_path, voxel_size, output_image_path = None):
        
        if not output_image_path:
            output_image_path = os.path.join(os.path.dirname(nifti_image_path), f"upsampled_{args.voxel_size}mm_{os.path.basename(nifti_image_path)}")

        upsampled_volume = self.resample_img_new_spacing(
            sitk.ReadImage(nifti_image_path),
            new_spacing=(voxel_size, voxel_size, voxel_size)
        )
        
        sitk.WriteImage(
            upsampled_volume, 
            output_volume_path
        )

    def resample_img(self, img, spacing, sz, interpolator = sitk.sitkLinear):
        # interpolator could be sitk.sitkLinear
        # interpolator could be sitk.sitkBSpline
        r = sitk.ResampleImageFilter()
        r.SetInterpolator(interpolator)
        r.SetOutputPixelType( img.GetPixelID() )
        r.SetDefaultPixelValue(0)
        r.SetOutputOrigin(img.GetOrigin())
        r.SetOutputSpacing(spacing)
        r.SetOutputDirection(img.GetDirection())
        r.SetSize(sz)
        return r.Execute(img)

        
    def resample_img_new_spacing(self, img, new_spacing):
        spacing = np.array(img.GetSpacing())
        sz = np.array(img.GetSize())
        new_sz = np.floor(spacing / new_spacing * sz).astype(np.uint32)
        new_sz = 2*np.floor((new_sz+1)/2).astype(np.uint32)

        return self.resample_img(img, new_spacing, new_sz.tolist())

        
if __name__ == '__main__':

    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nifti_image_path", required=True)
    parser.add_argument("--output_image_path", required=False, default=None)
    parser.add_argument("--voxel_size", required=True, type=float)
    args = parser.parse_args()

    UpsampleVolume(
        nifti_image_path=os.path.abspath(args.nifti_image_path),
        output_image_path=os.path.abspath(args.output_image_path) if args.output_image_path else None,
        voxel_size=args.voxel_size
    )
