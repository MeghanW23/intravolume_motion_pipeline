import os 
import sys
from PIL import Image
from matplotlib import pyplot as plt 
from plotly import graph_objects as go 
from plotly.subplots import make_subplots

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "PostPipelineAnalysis"))
from graph_carpet_plot import CarpetPlot

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from calculate_displacements import CalculateDisplacements

class CompareCarpetPlots: 
    def __init__(self, 
                 anatomical_image: str, 
                 reference_volume_image: str,
                 uncorrected_func_image: str,
                 fd_corrected_func_image: str,
                 sd_corrected_func_image: str, 
                 fd_corrected_transform_directory: str,
                 sd_corrected_transform_directory: str,
                 json_file: str, 
                 displacement_threshold: float = 0.4157,
                 output_directory: str = "outputs",
                 output_file_path: str = "carpet_plot_comparison.png",
                 ) -> None:

        print(f"\nCreating FD-Corrected Carpet Plot for: {os.path.basename(fd_corrected_func_image)}...")
        fdcorr_output_directory: str = os.path.join(output_directory, "fd-corrected_results")
        fdcorr_heatmaps: dict[str, go.Heatmap] = CarpetPlot(
            anatomical_image=anatomical_image,
            functional_image=fd_corrected_func_image,
            json_file=json_file,
            reference_volume_image=reference_volume_image,
            transform_directory=fd_corrected_transform_directory,
            output_directory=fdcorr_output_directory,
            output_file_path=os.path.join(fdcorr_output_directory, "carpet_plot.html"),
            displacement_threshold=displacement_threshold,
            also_save_png_file=False
        ).heatmaps
        self.heatmaps_to_png_fig(fdcorr_heatmaps, transform_directory=fd_corrected_transform_directory, output_file_path=os.path.join(fdcorr_output_directory, "stacked_carpet.png"))

        print(f"\nCreating Uncorrected Carpet Plot for: {os.path.basename(uncorrected_func_image)}...")
        uncorr_output_directory: str = os.path.join(output_directory, "uncorrected_results")
        uncorr_heatmaps: dict[str, go.Heatmap] = CarpetPlot(
            anatomical_image=anatomical_image,
            functional_image=uncorrected_func_image,
            json_file=json_file,
            reference_volume_image=reference_volume_image,
            transform_directory=fd_corrected_transform_directory, # doesnt matter what u put here for uncorrected
            output_directory=uncorr_output_directory,
            output_file_path=os.path.join(uncorr_output_directory, "carpet_plot.html"),
            displacement_threshold=displacement_threshold,
            also_save_png_file=False
        ).heatmaps
        self.heatmaps_to_png_fig(uncorr_heatmaps, output_file_path=os.path.join(uncorr_output_directory, "stacked_carpet.png"))
        
        
        print(f"\nCreating SD-Corrected Carpet Plot for: {os.path.basename(sd_corrected_func_image)}...")
        sdcorr_output_directory: str = os.path.join(output_directory, "sd-corrected_results")
        sdcorr_heatmaps: dict[str, go.Heatmap] = CarpetPlot(
            anatomical_image=anatomical_image,
            functional_image=sd_corrected_func_image,
            json_file=json_file,
            reference_volume_image=reference_volume_image,
            transform_directory=sd_corrected_transform_directory,
            output_directory=sdcorr_output_directory,
            output_file_path=os.path.join(sdcorr_output_directory, "carpet_plot.html"),
            displacement_threshold=displacement_threshold,
            also_save_png_file=False
        ).heatmaps
        self.heatmaps_to_png_fig(sdcorr_heatmaps, transform_directory=sd_corrected_transform_directory, output_file_path=os.path.join(sdcorr_output_directory, "stacked_carpet.png"))

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 10),  gridspec_kw={'height_ratios': [1.75, 2, 2]})
        fig.suptitle(
            f"Raw vs. Frame-Wise Motion Correction vs. Intra-Frame-Wise Motion Correction",
            fontsize=8,
            fontweight='bold')
        for row_num, output_dir, condition_name in [
                                                    (0, uncorr_output_directory, "Uncorrected Data"),
                                                    (1, fdcorr_output_directory, "Frame-Wise Corrected Data"),
                                                    (2, sdcorr_output_directory, "Intra-Frame-Wise Corrected Data")
                                                    ]:

            axes[row_num].set_title(condition_name, fontsize=8, loc='left')
            axes[row_num].imshow(Image.open(os.path.join(output_dir, "stacked_carpet.png")), aspect='auto')
            axes[row_num].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_file_path)
        print(f"Figure Saved to: {output_file_path}")
        plt.show()

    def heatmaps_to_png_fig(self, heatmaps: dict[str, go.Heatmap], output_file_path: str, transform_directory: str | None = None):
        labels: dict[str, str] = {'Gray Matter': 'GM Voxels', 'White Matter': 'WM Voxels', 'Cerebrospinal Fluid': 'CSF Voxels'}
        if transform_directory:
            displacements: list[float] = CalculateDisplacements(transform_directory=transform_directory, verbose=False).return_displacements()
            fig = make_subplots(
                rows=4, cols=1,  
                vertical_spacing=0.01, horizontal_spacing=0.0,
                row_heights=[0.4, 0.25, 0.2, 0.15]
            )
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(displacements))),
                    y=displacements
                ),
                row=1,
                col=1
            )
            fig.update_yaxes(title_text="Displacement (mm)", title_font=dict(size=15), tickfont=dict(size=10), row=1, col=1)
            for row_num, (matter_type, heatmap) in enumerate(heatmaps.items(), start=2):
                print(f"Plotting {matter_type}")
                fig.add_trace(heatmap, row=row_num, col=1)
                fig.update_yaxes(title_text=labels[matter_type], title_font=dict(size=15), tickfont=dict(size=10), row=row_num, col=1)

            fig.update_xaxes(title_text="Aquisition Number", title_font=dict(size=15), tickfont=dict(size=10), row=4, col=1)     
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0)) 
            fig.write_image(output_file_path, width=1400, height=500, scale=2)

        else:
            fig = make_subplots(
                rows=3, cols=1, 
                vertical_spacing=0.01, horizontal_spacing=0.0, 
                row_heights=[0.5, 0.3, 0.2]
            )
            for row_num, (matter_type, heatmap) in enumerate(heatmaps.items(), start=1):
                print(f"Plotting {matter_type}")
                fig.add_trace(heatmap, row=row_num, col=1)
                fig.update_yaxes(title_text=labels[matter_type], title_font=dict(size=15), tickfont=dict(size=10), row=row_num, col=1)

            fig.update_xaxes(title_text="Aquisition Number", title_font=dict(size=15), tickfont=dict(size=10), row=3, col=1) 
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0)) 
            fig.write_image(output_file_path, width=1400, height=500, scale=2)
        
        print(f"Stacked Carpet .png File at: {output_file_path}")
         

if __name__ == "__main__":
    
    # python compare_carpet_plots.py \
    #    --anatomical_image ../../../participant_data/carpet_plot_comparison_inputs/sub-04_ses-04_desc-preproc_T1w.nii.gz \
    #    --reference_volume_image ../../../participant_data/carpet_plot_comparison_inputs/refvol_upsampled_clamped_volume_outputs-0103.nii \
    #    --uncorrected_func_image ../../../participant_data/carpet_plot_comparison_inputs/p004-ses-04_func-bold_task-NFB2_20250827181314_24.nii.gz \
    #    --fd_corrected_func_image ../../../participant_data/carpet_plot_comparison_inputs/FD-ONLY_sub-04_ses-04_task-func_run-01_desc-preproc_bold.nii.gz \
    #    --sd_corrected_func_image ../../../participant_data/carpet_plot_comparison_inputs/SD_sub-04_ses-04_task-func_run-01_desc-preproc_bold.nii.gz \
    #    --fd_corrected_transform_directory ../../../participant_data/carpet_plot_comparison_inputs/FD-transforms \
    #    --sd_corrected_transform_directory ../../../participant_data/carpet_plot_comparison_inputs/SD-transforms \
    #    --json_file ../../../participant_data/carpet_plot_comparison_inputs/p004-ses-04_func-bold_task-NFB2_20250827181314_24.json
    
    
    import argparse
    parser = argparse.ArgumentParser(
        description="Compare Raw vs. FD-Corrected vs. SD-Corrected Carpet Plots of NiFTI Data."
    )
    parser.add_argument("--anatomical_image", required=True)
    parser.add_argument("--reference_volume_image", required=True)
    parser.add_argument("--uncorrected_func_image", required=True)
    parser.add_argument("--fd_corrected_func_image", required=True)
    parser.add_argument("--sd_corrected_func_image", required=True)
    parser.add_argument("--fd_corrected_transform_directory", required=True)
    parser.add_argument("--sd_corrected_transform_directory", required=True)
    parser.add_argument("--json_file", required=True)
    parser.add_argument("--displacement_threshold", required=False, default=0.4157, 
                        type=float, help="Default: 0.4157 mm")
    parser.add_argument("--output_directory", required=False, default="carpet_comparison_outputs", 
                        help=f"Default: {os.path.abspath('carpet_comparison_outputs')}")
    parser.add_argument("--output_file_path", required=False, default="carpet_comparison.png",
                         help=f"Default: {os.path.abspath('carpet_comparison.png')}")
    args = parser.parse_args()
    CompareCarpetPlots(
        anatomical_image=os.path.abspath(args.anatomical_image),
        reference_volume_image=os.path.abspath(args.reference_volume_image),
        uncorrected_func_image=os.path.abspath(args.uncorrected_func_image),
        fd_corrected_func_image=os.path.abspath(args.fd_corrected_func_image),
        sd_corrected_func_image=os.path.abspath(args.sd_corrected_func_image),
        fd_corrected_transform_directory=os.path.abspath(args.fd_corrected_transform_directory),
        sd_corrected_transform_directory=os.path.abspath(args.sd_corrected_transform_directory),
        json_file=os.path.abspath(args.json_file),
        output_directory=os.path.abspath(args.output_directory),
        output_file_path=os.path.abspath(args.output_file_path),
        displacement_threshold=args.displacement_threshold
    )