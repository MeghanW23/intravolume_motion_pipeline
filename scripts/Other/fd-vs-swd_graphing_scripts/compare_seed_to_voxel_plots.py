import os 
import numpy as np
import nibabel as nib 
from matplotlib import pyplot as plt 
from nilearn.plotting import plot_stat_map
from nilearn.masking import compute_epi_mask
from nilearn.image import crop_img, smooth_img, math_img
from nilearn.image import new_img_like

from nilearn.plotting.displays import OrthoSlicer

class CompareSeedToVoxelPlots:
    def __init__(self, 
                 framewise_corrected_map_nifti_image: str, 
                 intravolume_corrected_map_nifti_image: str, 
                 framewise_background_image: str,
                 intravolume_background_image: str,
                 output_file_path: str = "fd-vs-sd_seed-to-vox_plots.png",
                 plot_title: str = "Single Participant Seed to Voxel Correlation",
                 seed_coords: tuple[int, int, int] = (0, -52, 18),
                 smoothing_fwhm: float | None = 6):

        loaded_fd_image: nib.Nifti1Image = nib.load(framewise_corrected_map_nifti_image) # pyright: ignore[reportAssignmentType, reportPrivateImportUsage]
        loaded_sd_image: nib.Nifti1Image = nib.load(intravolume_corrected_map_nifti_image) # pyright: ignore[reportAssignmentType, reportPrivateImportUsage]
        loaded_fd_bg_image: nib.Nifti1Image = nib.load(framewise_background_image) # pyright: ignore[reportAssignmentType, reportPrivateImportUsage]
        loaded_sd_bg_image: nib.Nifti1Image = nib.load(intravolume_background_image)  # pyright: ignore[reportAssignmentType, reportPrivateImportUsage]

        if smoothing_fwhm is not None:
            loaded_fd_image = smooth_img(
                loaded_fd_image,
                fwhm=smoothing_fwhm
            ) # pyright: ignore[reportAssignmentType]
            loaded_sd_image = smooth_img(
                loaded_sd_image,
                fwhm=smoothing_fwhm
            ) # pyright: ignore[reportAssignmentType]

        max_val: float = max(np.nanmax(loaded_sd_image.get_fdata()), np.nanmax(loaded_fd_image.get_fdata())) # pyright: ignore[reportAssignmentType]
        min_val: float = min(np.nanmin(loaded_sd_image.get_fdata()), np.nanmin(loaded_fd_image.get_fdata())) # pyright: ignore[reportAssignmentType]

        vmin: float = min_val + ((max_val - min_val) / 4)
        print(f"Value Range: {min_val} to {max_val}")

        fig, axes = plt.subplots(nrows=2, ncols=1, facecolor='white', figsize=(6, 4))
        fig.suptitle(plot_title, fontweight="bold", fontsize=8)
        fig.text(
            0.5, 0.93,                      # x=centered, y=just below suptitle
            f"Seed at: {seed_coords}",
            ha='center',
            fontsize=7,
            fontweight='normal',
            color='dimgray',
        )
                
        plt.rcParams.update({'font.size': 4})

        fd_map: OrthoSlicer = plot_stat_map(
            loaded_fd_image,
            bg_img=loaded_fd_bg_image,
            cut_coords=seed_coords,
            vmin=vmin,
            vmax=max_val,
            annotate=False,
            black_bg=False, # type: ignore
            axes=axes[0], # type: ignore
            threshold=0.15,
        )
        fd_map.annotate(left_right=False, positions=False, size=5)
        fd_map.title(
            "Framewise Motion-Corrected",
            size=7,
            color='black',
            bgcolor='white',
            alpha=0,
            y=1.11,
            clip_on=False,
        )
        fd_map.add_markers(
            marker_coords=[seed_coords], marker_color="g", marker_size=30
        )
        fd_map.annotate(left_right=True, positions=True, size=5)

        sd_map: OrthoSlicer = plot_stat_map(
            loaded_sd_image,
            bg_img=loaded_sd_bg_image,
            cut_coords=seed_coords,
            vmin=vmin,
            vmax=max_val,
            annotate=False,
            black_bg=False, # type: ignore
            axes=axes[1], # type: ignore
            threshold=0.15
        ) 
        sd_map.title(
            "Intra-Frame Motion-Corrected",
            size=7,
            color='black',
            bgcolor='white',
            alpha=0,
            y=1.11,
            clip_on=False,
        )
        sd_map.add_markers(
            marker_coords=[seed_coords], marker_color="g", marker_size=30
        )
        sd_map.annotate(left_right=True, positions=True, size=5)

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
        description="Compare the Seed-to-Voxel Correlation Plots from Framewise vs. Intra-Frame Motion Corrected Data"
    )
    parser.add_argument(
        "--framewise_corrected_map",
        required=True,
        help="The 3D Correlation NiFTI Map for the Framewise-Motion Corrected Data."
    )
    parser.add_argument(
        "--intraframe_corrected_map",
        required=True,
        help="The 3D Correlation NiFTI Map for the Intra-Frame-Motion Corrected Data."
    )
    parser.add_argument(
        "--framewise_background_image_path",
        required=True,
        help="The background 3D NiFTI Image for the Framewise-Motion Corrected Data."
    )
    parser.add_argument(
        "--intravolume_background_image_path",
        required=True,
        help="The background 3D NiFTI Image for the Intra-Frame-Motion Corrected Data."
    )
    parser.add_argument(
        "--seed_coords",
        required=False,
        default=(0, -52, 18),
        type=int,
        nargs=3,
        help="Default: (0, -52, 18)"
    )
    parser.add_argument(
        "--output_file_path",
        required=False,
        default="fd-vs-sd_seed-to-vox_plots.png",
        help=f"Default: {os.path.abspath('fd-vs-sd_seed-to-vox_plots.png')}"
    )
    parser.add_argument(
        "--plot_title",
        required=False,
        help="Default: 'Single Participant Seed to Voxel Correlation'",
        default="Single Participant Seed to Voxel Correlation"
    )
    
    args: argparse.Namespace = parser.parse_args()
    CompareSeedToVoxelPlots(
        framewise_corrected_map_nifti_image=os.path.abspath(args.framewise_corrected_map),
        intravolume_corrected_map_nifti_image=os.path.abspath(args.intraframe_corrected_map),
        framewise_background_image=os.path.abspath(args.framewise_background_image_path),
        intravolume_background_image=os.path.abspath(args.intravolume_background_image_path),
        seed_coords=args.seed_coords,
        output_file_path=os.path.abspath(args.output_file_path),
        plot_title=args.plot_title
    )