import os 
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
from nilearn.plotting import plot_stat_map
from nilearn.image import crop_img, smooth_img
from nilearn.plotting.displays import OrthoSlicer


class CompareTSNRPlots:
    def __init__(self, 
                 framewise_corrected_tsnr_nifti_image: str, 
                 intravolume_corrected_tsnr_nifti_image: str, 
                 background_image: str,
                 output_file_path: str = "fd-vs-sd_tsnr_plots.png",
                 plot_title: str = "Temporal Signal to Noise Ratio (tSNR)",
                 smoothing_fwhm: float | None = 6) -> None:

        loaded_fd_image: nib.Nifti1Image = nib.load(framewise_corrected_tsnr_nifti_image) # pyright: ignore[reportAssignmentType, reportPrivateImportUsage]
        loaded_sd_image: nib.Nifti1Image = nib.load(intravolume_corrected_tsnr_nifti_image) # pyright: ignore[reportAssignmentType, reportPrivateImportUsage]
        loaded_bg_image: nib.Nifti1Image = nib.load(background_image) # pyright: ignore[reportAssignmentType, reportPrivateImportUsage]
        loaded_bg_image = crop_img(nib.load(background_image), pad=2, copy_header=True) # type: ignore

        if smoothing_fwhm is not None:
            loaded_fd_image = smooth_img(
                loaded_fd_image,
                fwhm=smoothing_fwhm
            ) # pyright: ignore[reportAssignmentType]
            loaded_sd_image = smooth_img(
                loaded_sd_image,
                fwhm=smoothing_fwhm
            ) # pyright: ignore[reportAssignmentType]
            
        diff_data: np.ndarray = loaded_sd_image.get_fdata() - loaded_fd_image.get_fdata()
        diff_image: nib.Nifti1Image =  nib.Nifti1Image(diff_data, loaded_sd_image.affine, loaded_sd_image.header) # pyright: ignore[reportPrivateImportUsage]

        max_val: float = max(np.nanmax(loaded_sd_image.get_fdata()), np.nanmax(loaded_fd_image.get_fdata())) # pyright: ignore[reportAssignmentType]
        min_val: float = min(np.nanmin(loaded_sd_image.get_fdata()), np.nanmin(loaded_fd_image.get_fdata())) # pyright: ignore[reportAssignmentType]

        vmin: float = min_val + ((max_val - min_val) / 4)
        print(f"Value Range: {min_val} to {max_val}")

        fig, axes = plt.subplots(nrows=3, ncols=1, facecolor='white', figsize=(6, 4))
        fig.suptitle(plot_title, fontweight="bold", fontsize=8)

        
        coords: list[int] = [0, 30, 60]
        plt.rcParams.update({'font.size': 4})

        fd_map: OrthoSlicer = plot_stat_map(
            loaded_fd_image,
            bg_img=loaded_bg_image,
            display_mode='z',
            cut_coords=coords,
            cmap='hot',
            vmin=vmin,
            vmax=max_val,
            annotate=False,
            black_bg=False,
            # threshold=min_val + ((max_val - min_val) / 1.5),
            axes=axes[0], # pyright: ignore[reportIndexIssue, reportAssignmentType]
        )
        fd_map.annotate(left_right=False, positions=False, size=5)
        fd_map.title(
            "Framewise Motion-Corrected",
            size=7,
            color='black',
            bgcolor='white',
            alpha=1,
        )

        sd_map: OrthoSlicer = plot_stat_map(
            loaded_sd_image,
            bg_img=loaded_bg_image,
            display_mode='z',
            cut_coords=coords,
            cmap='hot',
            vmin=vmin,
            vmax=max_val,
            annotate=False,
            black_bg=False,
            # threshold=min_val + ((max_val - min_val) / 1.5),
            axes=axes[1] # pyright: ignore[reportIndexIssue, reportAssignmentType]
        ) 
        sd_map.title(
            "Intra-Frame Motion-Corrected",
            size=7,
            color='black',
            bgcolor='white',
            alpha=1,
        )

        diff_map: OrthoSlicer = plot_stat_map(
            diff_image,
            bg_img=loaded_bg_image,
            display_mode='z',
            cut_coords=coords,
            annotate=False,
            black_bg=False,
            threshold=5,
            axes=axes[2] # pyright: ignore[reportIndexIssue, reportAssignmentType]
        )
        diff_map.annotate(left_right=False, positions=True, size=5)
        diff_map.title(
            "Difference: Intra-Frame - Framewise",
            size=7,
            color='black',
            bgcolor='white',
            alpha=1
        )
        plt.savefig(
            output_file_path,
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.2,
            facecolor='white',
        )
        print(f"Plot at: {output_file_path}")

        
if __name__ == "__main__":
    import argparse
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Compare the tSNR Plots from Framewise vs. Intra-Frame Motion Corrected Data"
    )
    parser.add_argument(
        "--framewise_corrected_tsnr_map",
        required=True,
        help="The 3D tSNR Map NiFTI Image for the Framewise-Motion Corrected Data."
    )
    parser.add_argument(
        "--intraframe_corrected_tsnr_map",
        required=True,
        help="The 3D tSNR Map NiFTI Image for the Intra-Frame-Motion Corrected Data."
    )
    parser.add_argument(
        "--background_image_path",
        required=True,
        help="The background 3D NiFTI Image."
    )
    parser.add_argument(
        "--output_file_path",
        required=False,
        default="fd-vs-sd_tsnr_plots.png",
        help=f"Default: {os.path.abspath('fd-vs-sd_tsnr_plots.png')}"
    )
    parser.add_argument(
        "--plot_title",
        required=False,
        help="Default: 'Temporal Signal to Noise Ratio (tSNR)'",
        default="Temporal Signal to Noise Ratio (tSNR)"
    )
    args: argparse.Namespace = parser.parse_args()
    CompareTSNRPlots(
        framewise_corrected_tsnr_nifti_image=os.path.abspath(args.framewise_corrected_tsnr_map),
        intravolume_corrected_tsnr_nifti_image=os.path.abspath(args.intraframe_corrected_tsnr_map),
        background_image=os.path.abspath(args.background_image_path),
        output_file_path=os.path.abspath(args.output_file_path),
        plot_title=args.plot_title
    )