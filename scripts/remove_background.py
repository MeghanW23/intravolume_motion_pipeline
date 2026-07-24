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
    def __init__(self, 
                 nifti_file_path: str, 
                 output_file_path: str | None = None):

        self.output_file_path: str = output_file_path # pyright: ignore[reportAttributeAccessIssue]
        if not self.output_file_path:
            self.output_file_path = os.path.join(
                os.path.dirname(nifti_file_path),
                f"{os.path.basename(nifti_file_path).replace('.nii.gz', '').replace('.nii', '')}_bgremoved.nii.gz"
            )
            print(f"Output File Path Set to: {self.output_file_path}")
        
        print(f"Reading Nifti image: {nifti_file_path} as a SimpleITK Image")
        sitk_fmri_image: sitk.Image = sitk.ReadImage(
            nifti_file_path,
            sitk.sitkFloat64
        )
        if sitk_fmri_image.GetDimension() != 4:
            raise ValueError(f"The input nifti_file_path must be 4D.")
        
        origin_4d: np.ndarray = sitk_fmri_image.GetOrigin()
        spacing_4d: np.ndarray = sitk_fmri_image.GetSpacing()
        direction_4d: np.ndarray = sitk_fmri_image.GetDirection()
        dim1, dim2, num_slices, num_volumes = sitk_fmri_image.GetSize()
        
        print(f"Reading Nifti image: {nifti_file_path} as a nib.Nifti1Image")
        nib_fmri_image: nib.nifti1.Nifti1Image = nib.load(nifti_file_path) # pyright: ignore[reportAssignmentType, reportPrivateImportUsage]

        print("Computing the mean image")
        mean_nib_fmri_image: nib.nifti1.Nifti1Image = image.mean_img(nib_fmri_image, copy_header=True) # pyright: ignore[reportPrivateImportUsage, reportAssignmentType]

        print("Computing the mask")
        mask: nib.nifti1.Nifti1Image = masking.compute_epi_mask(
            mean_nib_fmri_image, 
            lower_cutoff=0.25,
            upper_cutoff=0.75
        ) # pyright: ignore[reportAssignmentType]
        masker = NiftiMasker(mask_img=mask)
        masker.fit()

        print("Creating the transforms")
        data1: np.ndarray = masker.transform(nib_fmri_image)
        data2: np.ndarray = masker.inverse_transform(data1) # type: ignore
        A: np.ndarray = data2.get_fdata() # pyright: ignore[reportAttributeAccessIssue]
        data3: np.ndarray = np.transpose(A,(3,2,1,0))

        print("Converting the Numpy Array to a SimpleITK Image")
        sitk_image: sitk.Image = self.Numpy4d_to_SITK(
            data3,
            origin_4d,
            direction_4d,
            spacing_4d,
            num_volumes
        )

        sitk.WriteImage(
            sitk_image,
            self.output_file_path
        )
        print(f"Background Removed Image at: {self.output_file_path}")


    def Numpy4d_to_SITK(self, 
                        nd_array_data: np.ndarray, 
                        origin: np.ndarray, 
                        direction: np.ndarray, 
                        spacing: np.ndarray, 
                        num_volumes: int) -> sitk.Image:
    
        loaded_volumes: list[sitk.Image] = [
            sitk.GetImageFromArray(nd_array_data[volume_num,:,:,:])
            for volume_num in range(0, num_volumes)
        ]
            
        timeseries: sitk.Image = sitk.JoinSeries(loaded_volumes)
        timeseries.SetOrigin(origin)
        timeseries.SetSpacing(spacing)
        timeseries.SetDirection(direction) 
        
        return timeseries


    def return_output_file_path(self) -> str:
        return self.output_file_path

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="To remove the background/skull from a 4d NiFTI file")
    parser.add_argument(
        "--nifti_file_path", 
        required=True,
        help="The raw, 4D NiFTI Image."
    )
    parser.add_argument(
        "--output_file_path", 
        required=False, 
        help="Default is <nifti_file_path>_bgremoved.nii.gz"
    )
    args = parser.parse_args()
    RemoveBackground(
        nifti_file_path=os.path.abspath(args.nifti_file_path),
        output_file_path=\
            os.path.abspath(args.output_file_path) 
            if args.output_file_path else None
    )


      
