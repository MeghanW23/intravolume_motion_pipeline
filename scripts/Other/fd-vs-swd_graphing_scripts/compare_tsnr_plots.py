import os 
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
from nilearn.plotting import plot_stat_map
from nilearn.plotting.displays import OrthoSlicer

class CompareTSNRPlots:
    def __init__(self, 
                 framewise_corrected_tsnr_nifti_image: str, 
                 intravolume_corrected_tsnr_nifti_image: str, 
                 background_image: str,
                 output_file_path: str = "fd-vs-sd_tsnr_plots.png",
                 plot_title: str = "tSNR Maps: Framewise vs. Intra-Frame Motion Corrected Preproccessed Data") -> None:

        loaded_fd_image: nib.Nifti1Image = nib.load(framewise_corrected_tsnr_nifti_image) # pyright: ignore[reportAssignmentType, reportPrivateImportUsage]
        loaded_sd_image: nib.Nifti1Image = nib.load(intravolume_corrected_tsnr_nifti_image) # pyright: ignore[reportAssignmentType, reportPrivateImportUsage]
        loaded_bg_image: nib.Nifti1Image = nib.load(background_image) # pyright: ignore[reportAssignmentType, reportPrivateImportUsage]

        
        diff_data: np.ndarray = loaded_sd_image.get_fdata() - loaded_fd_image.get_fdata()
        diff_image: nib.Nifti1Image =  nib.Nifti1Image(diff_data, loaded_sd_image.affine, loaded_sd_image.header) # pyright: ignore[reportPrivateImportUsage]

        max_val: float = max(np.nanmax(loaded_sd_image.get_fdata()), np.nanmax(loaded_fd_image.get_fdata())) # pyright: ignore[reportAssignmentType]
        min_val: float = min(np.nanmin(loaded_sd_image.get_fdata()), np.nanmin(loaded_fd_image.get_fdata())) # pyright: ignore[reportAssignmentType]
        print(f"Value Range: {min_val} to {max_val}")

        fig, axes = plt.subplots(nrows=3, ncols=1, facecolor='white', figsize=(14, 8.5))
        fig.suptitle(plot_title, fontweight="bold")

        num_z_slices: int = loaded_bg_image.shape[-1]
        divisor: int = num_z_slices // 5
        coords: list[int] = [0, divisor, divisor * 2, divisor * 3, divisor * 4]
        
        fd_map: OrthoSlicer = plot_stat_map(
            loaded_fd_image,
            bg_img=loaded_bg_image,
            display_mode='z',
            cut_coords=coords,
            vmin=min_val,
            vmax=max_val,
            black_bg=False,
            dim=-1,
            annotate=False,
            axes=axes[0],
        ) # pyright: ignore[reportAssignmentType]
        fd_map.annotate(left_right=True, positions=False)
        fd_map.title(
            "Framewise Motion-Corrected",
            size=12,
            color='black',
            bgcolor='white',
            alpha=1,
        )

        sd_map: OrthoSlicer = plot_stat_map(
            loaded_sd_image,
            bg_img=loaded_bg_image,
            display_mode='z',
            cut_coords=coords,
            vmin=min_val,
            vmax=max_val,
            black_bg=False,
            annotate=False,
            dim=-1,
            axes=axes[1]
        ) # pyright: ignore[reportAssignmentType]
        sd_map.title(
            "Intra-Frame Motion-Corrected",
            size=12,
            color='black',
            bgcolor='white',
            alpha=1,
        )

        diff_map: OrthoSlicer = plot_stat_map(
            diff_image,
            bg_img=loaded_bg_image,
            display_mode='z',
            cut_coords=coords,
            black_bg=False,
            dim=-1,
            annotate=False,
            axes=axes[2]
        ) # pyright: ignore[reportAssignmentType]
        diff_map.annotate(left_right=False, positions=True)
        diff_map.title(
            "Intra-Frame Motion-Corrected - Framewise Motion-Corrected",
            size=12,
            color='black',
            bgcolor='white',
            alpha=1,
        )
        plt.savefig(
            output_file_path,
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.2,   # was 0.05 — give it more room
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
        help="Default: 'tSNR Maps: Framewise vs. Intra-Frame Motion Corrected Preproccessed Data'",
        default="tSNR Maps: Framewise vs. Intra-Frame Motion Corrected Preproccessed Data"
    )
    args: argparse.Namespace = parser.parse_args()
    CompareTSNRPlots(
        framewise_corrected_tsnr_nifti_image=os.path.abspath(args.framewise_corrected_tsnr_map),
        intravolume_corrected_tsnr_nifti_image=os.path.abspath(args.intraframe_corrected_tsnr_map),
        background_image=os.path.abspath(args.background_image_path),
        output_file_path=os.path.abspath(args.output_file_path),
        plot_title=args.plot_title
    )