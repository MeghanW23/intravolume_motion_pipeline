import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from nilearn import image
from nilearn import masking
from nilearn.maskers import NiftiMasker

class RemoveBackground:
    """
    Code adapted from MaskOutBackground.py script (author: ch20807) in the rsfMRI_SMC_mc Repo:
    https://github.com/bchimagine/rsfMRI_SMC_mc/ 
    """
    def __init__(self, nifti_file_path, output_path = None):

        if not output_path:
            output_path = os.path.join(
                os.path.dirname(nifti_file_path),
                f"{os.path.basename(nifti_file_path).replace('.nii.gz', '').replace('.nii', '')}_bgremoved.nii.gz"
            )
        
        sitk_fmri_image = sitk.ReadImage(
            nifti_file_path,
            sitk.sitkFloat64)
        origin_4d = sitk_fmri_image.GetOrigin()
        spacing_4d = sitk_fmri_image.GetSpacing()
        direction_4d = sitk_fmri_image.GetDirection()
        dim1, dim2, num_slices, num_volumes = sitk_fmri_image.GetSize()
        
        nib_fmri_image = nib.load(nifti_file_path)
        mean_nib_fmri_image = image.mean_img(nib_fmri_image)
        mask = masking.compute_epi_mask(
            mean_nib_fmri_image, 
            lower_cutoff=0.25,
            upper_cutoff=0.75
        )
        masker = NiftiMasker(mask_img=mask)
        masker.fit(nib_fmri_image)
        mask_fname = masker.mask_img_

        data1 = masker.fit_transform(nib_fmri_image)
        data2 = masker.inverse_transform(data1)
        A = data2.get_fdata()
        B = nib_fmri_image.get_fdata()
        data3 = np.transpose(A,(3,2,1,0))
        
        sitk_image = self.Numpy4d_to_SITK(
            data3,
            origin_4d,
            direction_4d,
            spacing_4d,
            num_volumes)

        sitk.WriteImage(
            sitk_image,
            output_path
        )
        print(f"Background Removed Image at: {output_path}")

    def Numpy4d_to_SITK(self, nd_array_data, origin, direction, spacing, num_volumes):
    
        loaded_volumes = []
        
        for volume_num in range(0, num_volumes):
            loaded_volumes.append(sitk.GetImageFromArray(nd_array_data[volume_num,:,:,:]))
            
        timeseries = sitk.JoinSeries(loaded_volumes)
        timeseries.SetOrigin(origin)
        timeseries.SetSpacing(spacing)
        timeseries.SetDirection(direction) 
        
        return timeseries

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="To remove the background/skull from a 4d NiFTI file")
    parser.add_argument("--nifti_file_path", required=True)
    parser.add_argument("--output_file_path", required=False, help="Default is <nifti_file_path>_bgremoved.nii.gz")
    args = parser.parse_args()

    RemoveBackground(
        nifti_file_path=os.path.abspath(args.nifti_file_path),
        output_path=os.path.abspath(args.output_file_path) if args.output_file_path else None
    )


      
