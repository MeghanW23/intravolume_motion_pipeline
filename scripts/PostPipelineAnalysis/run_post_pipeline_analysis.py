import os 
import sys
# add post-pipeline analysis directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "PostPipelineAnalysis"))
from calculate_tsnr import CalculateTSNR
from graph_carpet_plot import CarpetPlot
from seed_to_voxel_correlation import SeedToVoxelCorrelation
 
class PostRunAnalysis:
    def __init__(self,
                json_file: str,
                anatomical_nifti_image: str,
                transform_directory: str,
                mm_displacement_threshold: float,
                brain_mask_path: str,
                reference_volume_path: str,
                confounds_file_path: str,
                raw_func_nifti_image: str | None = None,
                corrected_func_nifti_image: str | None = None,
                fmriprep_func_nifti_image: str | None = None,
                output_directory: str = "outputs"):

        os.makedirs(output_directory, exist_ok=True)

        if raw_func_nifti_image:
            raw_data_output_dir: str = os.path.join(output_directory, "raw_data_outputs")
            os.makedirs(raw_data_output_dir, exist_ok=True)

            print(f"\nCreating Raw Data Carpet Plot")
            CarpetPlot(
                anatomical_image=anatomical_nifti_image,
                functional_image=raw_func_nifti_image,
                json_file=json_file,
                reference_volume_image=reference_volume_path,
                transform_directory=transform_directory,
                displacement_threshold=mm_displacement_threshold,
                transform_suffix=".tfm",
                plot_title="Raw Data: Voxel Percent Signal Change Carpet Plot + Displacements",
                output_directory=os.path.join(raw_data_output_dir, "segmentations"),
                output_file_path=os.path.join(raw_data_output_dir, "raw_data_carpet_plot.html"),
                also_save_png_file=True  
            )

            print(f"\nCalculating TSNR")
            CalculateTSNR(
                nifti_image_path=raw_func_nifti_image,
                nifti_brain_mask_path=brain_mask_path,
                output_directory=raw_data_output_dir,
                plot_title="Raw Data: tSNR Plot"
            )

            print(f"\nDoing Seed to Voxel Correlation")
            SeedToVoxelCorrelation(
                nifti_image_path=raw_func_nifti_image,
                nifti_image_mask_path=brain_mask_path,
                json_file_path=json_file,
                confounds_file_path=confounds_file_path,
                output_directory_path=raw_data_output_dir,
                plot_title="Raw Data: Seed to Voxel Correlation"
            )


        if corrected_func_nifti_image:
            corrected_data_output_dir: str = os.path.join(output_directory, "motion_corrected_data_outputs")
            os.makedirs(corrected_data_output_dir, exist_ok=True)

            print(f"\nCreating Corrected Data Carpet Plot")
            CarpetPlot(
                anatomical_image=anatomical_nifti_image,
                functional_image=corrected_func_nifti_image,
                json_file=json_file,
                reference_volume_image=reference_volume_path,
                transform_directory=transform_directory,
                displacement_threshold=mm_displacement_threshold,
                transform_suffix=".tfm",
                plot_title="Intravolume Motion Corrected Data: Voxel Percent Signal Change Carpet Plot + Displacements",
                output_directory=os.path.join(corrected_data_output_dir, "segmentations"),
                output_file_path=os.path.join(corrected_data_output_dir, "corrected_data_carpet_plot.html"),
                also_save_png_file=True   
            )

            print(f"\nCalculating TSNR")
            CalculateTSNR(
                nifti_image_path=corrected_func_nifti_image,
                nifti_brain_mask_path=brain_mask_path,
                output_directory=corrected_data_output_dir,
                plot_title="Intravolume Motion Corrected Data: tSNR Plot"
            )

            print(f"\nDoing Seed to Voxel Correlation")
            SeedToVoxelCorrelation(
                nifti_image_path=corrected_func_nifti_image,
                nifti_image_mask_path=brain_mask_path,
                json_file_path=json_file,
                confounds_file_path=confounds_file_path,
                output_directory_path=corrected_data_output_dir,
                plot_title="Intravolume Motion Corrected Data: Seed to Voxel Correlation"
            )


        if fmriprep_func_nifti_image:
            fmriprep_data_output_dir: str = os.path.join(output_directory, "fmriprep_data_outputs")
            os.makedirs(fmriprep_data_output_dir, exist_ok=True)


            print(f"\nCreating fMRIPrep Data Carpet Plot")
            CarpetPlot(
                anatomical_image=anatomical_nifti_image,
                functional_image=fmriprep_func_nifti_image,
                json_file=json_file,
                reference_volume_image=reference_volume_path,
                transform_directory=transform_directory,
                displacement_threshold=mm_displacement_threshold,
                transform_suffix=".tfm",
                plot_title="fMRIPrep + Intravolume Motion Corrected Data: Voxel Percent Signal Change Carpet Plot + Displacements",
                output_directory=os.path.join(fmriprep_data_output_dir, "segmentations"),
                output_file_path=os.path.join(fmriprep_data_output_dir, "fmriprep_data_carpet_plot.html"),
                also_save_png_file=True   
            )

            print(f"\nCalculating TSNR")
            CalculateTSNR(
                nifti_image_path=fmriprep_func_nifti_image,
                nifti_brain_mask_path=brain_mask_path,
                output_directory=fmriprep_data_output_dir,
                plot_title="fMRIPrep + Intravolume Motion Corrected Data: tSNR Plot"
            )

            print(f"\nDoing Seed to Voxel Correlation")
            SeedToVoxelCorrelation(
                nifti_image_path=fmriprep_func_nifti_image,
                nifti_image_mask_path=brain_mask_path,
                json_file_path=json_file,
                confounds_file_path=confounds_file_path,
                output_directory_path=fmriprep_data_output_dir,
                plot_title="fMRIPrep + Intravolume Motion Corrected Data: Seed to Voxel Correlation"
            )
if __name__ == "__main__":
    import argparse
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Wrapper post-analysis scripts.")
    parser.add_argument("--raw_func_nifti_image_path", required=True, help="Raw, unprocessed nifti image.")
    parser.add_argument("--corrected_nifti_image_path", required=True, help="The nifti image outputted by the motion correction step.")
    parser.add_argument("--fmriprep_nifti_image_path", required=True, help="The subject-space nifti image outputted by fmriPrep.")
    parser.add_argument("--json_file_path", required=True, help="The dcm2niix-outputted json sidecar.")
    parser.add_argument("--anatomical_image_path", required=True)
    parser.add_argument("--transform_directory", required=True, help="Transform directory outputted by the motion characterization step.")
    parser.add_argument("--mm_displacement_threshold", required=True, help="in mm", type=float)
    parser.add_argument("--brain_mask_path", required=True, help="Must be in subject-space.")
    parser.add_argument("--reference_volume_path", required=True)
    parser.add_argument("--output_directory", required=False, default="outputs", help=f"Default: {os.path.abspath('outputs')}")
    parser.add_argument("--confounds_file_path", required=True, help="fMRIPrep desc-confounds_timeseries.tsv file.")
    args: argparse.Namespace = parser.parse_args()

    PostRunAnalysis(
        json_file=os.path.abspath(args.json_file_path),
        anatomical_nifti_image=os.path.abspath(args.anatomical_image_path),
        transform_directory=os.path.abspath(args.transform_directory),
        mm_displacement_threshold=args.mm_displacement_threshold,
        brain_mask_path=os.path.abspath(args.brain_mask_path),
        reference_volume_path=os.path.abspath(args.reference_volume_path),
        confounds_file_path=os.path.abspath(args.confounds_file_path),
        raw_func_nifti_image=os.path.abspath(args.raw_func_nifti_image_path) if args.raw_func_nifti_image_path else None,
        corrected_func_nifti_image=os.path.abspath(args.corrected_nifti_image_path) if args.corrected_nifti_image_path else None,
        fmriprep_func_nifti_image=os.path.abspath(args.fmriprep_nifti_image_path) if args.fmriprep_nifti_image_path else None,
        output_directory=os.path.abspath(args.output_directory)
    )