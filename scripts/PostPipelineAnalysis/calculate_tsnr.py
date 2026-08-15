import os 
import numpy as np
import nibabel as nib 
from nilearn.image import mean_img 
from nilearn.plotting import plot_stat_map


class CalculateTSNR:
    def __init__(self, nifti_image_path: str, nifti_brain_mask_path: str, output_directory: str = "outputs", plot_title: str = "tSNR Plot") -> None:


        os.makedirs(output_directory, exist_ok=True)


        print("Loading Data")
        img: nib.Nifti1Image = nib.load(nifti_image_path) # pyright: ignore[reportAssignmentType, reportPrivateImportUsage]
        data: np.ndarray = img.get_fdata()
        print(f"Image Dimensions: {data.shape}")
        if len(data.shape) != 4:
            raise ValueError("Your input image must be 4D.")


        mask_img: nib.Nifti1Image = nib.load(nifti_brain_mask_path) # pyright: ignore[reportAssignmentType, reportPrivateImportUsage]
        mask_img_data: np.ndarray = mask_img.get_fdata() > 0
        print(f"Mask Image Dimensions: {mask_img_data.shape}")
        if len(mask_img_data.shape) != 3:
            raise ValueError("Your input mask image must be 3D.")

        if not np.allclose(img.affine, mask_img.affine): # pyright: ignore[reportArgumentType]
            raise ValueError("fMRI image and mask are not in the same spatial space.")

        print("Meaning the Signal Over Time")
        mean_signal: np.ndarray = np.mean(data, axis=3)


        print("Getting Temporal Standard Deviation")
        temporal_std: np.ndarray = np.std(data, axis=3, ddof=1)


        print("Getting tSNR")
        tsnr: np.ndarray = mean_signal / temporal_std


        print("Masking tSNR Image")
        tsnr_masked: np.ndarray = np.where(mask_img_data, tsnr, np.nan)


        print("Creating tSNR NiFTI Map")
        tsnr_img: nib.Nifti1Image = nib.Nifti1Image( # pyright: ignore[reportPrivateImportUsage]
            tsnr_masked.astype(np.float32),
            img.affine,
            img.header
        )
        tsnr_img_path: str = os.path.join(output_directory, "tsnr_map.nii.gz")
        nib.save(tsnr_img, tsnr_img_path) # pyright: ignore[reportPrivateImportUsage]
        print(f"tSNR NiFTI Map Saved to: {tsnr_img_path}")


        print("Calculating tSNR Statistics")
        mean_tsnr: float = np.nanmean(tsnr_masked) # pyright: ignore[reportAssignmentType]
        median_tsnr = np.nanmedian(tsnr_masked)

        print(f"Mean tSNR: {mean_tsnr:.2f}")
        print(f"Median tSNR: {median_tsnr}")

        print(f"Plotting Map to .png Image")
        tsnr_img_plot: str = os.path.join(output_directory, "tsnr_map_plot.png")
        plot_stat_map(
            tsnr_img,
            bg_img=mean_img(img),
            title="tSNR Plot",
            display_mode='z',
            cut_coords=3,
            output_file=tsnr_img_plot
        )
        print(f"tSNR Map Plot Saved to: {tsnr_img_plot}")


if __name__ == "__main__":
    import argparse
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=\
            "Calculate Temporal Signal To Noise Ratio (tSNR) of a 4D " \
            "NiFTI Image"
    )
    parser.add_argument(
        "--nifti_image_path",
        required=True,
        help="Must be 4D"
    )
    parser.add_argument(
        "--nifti_brain_mask_path",
        required=True,
        help="Must be 3D and Binary."
    )
    parser.add_argument(
        "--output_directory",
        required=False,
        default='outputs',
        help=f"Default: {os.path.abspath('outputs')}"
    )
    args: argparse.Namespace = parser.parse_args()
    CalculateTSNR(
        nifti_image_path=os.path.abspath(args.nifti_image_path),
        nifti_brain_mask_path=os.path.abspath(args.nifti_brain_mask_path),
        output_directory=os.path.abspath(args.output_directory)
    )
