import os 
import sys
# add script directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from manage_configuration_files import Configurations
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
                 slice_times_text_file: str, 
                 mm_displacement_threshold: int,
                 brain_mask_path: str,
                 reference_volume_path: str,
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
                output_directory=raw_data_output_dir,
                slice_times_text_file=slice_times_text_file,
                also_save_png_file=True   
            )

            print(f"\nCalculating TSNR")
            CalculateTSNR(
                nifti_image_path=raw_func_nifti_image,
                nifti_brain_mask_path=brain_mask_path,
                output_directory=raw_data_output_dir,
                plot_title="Raw Data: tSNR Plot"
            )

            print(f"\nSeed to Voxel Correlation")
            SeedToVoxelCorrelation(
                nifti_image_path=raw_func_nifti_image,
                nifti_image_mask_path=brain_mask_path,
                json_file_path=json_file,
                output_directory_path=raw_data_output_dir,
                plot_title="Raw Data: Seed to Voxel Correlation"
            )


        if corrected_func_nifti_image:
            corrected_data_output_dir: str = os.path.join(output_directory, "motion_corrected_data_outputs")
            os.makedirs(corrected_data_output_dir, exist_ok=True)

            print(f"\nCreating Raw Data Carpet Plot")
            CarpetPlot(
                anatomical_image=anatomical_nifti_image,
                functional_image=corrected_func_nifti_image,
                json_file=json_file,
                reference_volume_image=reference_volume_path,
                transform_directory=transform_directory,
                displacement_threshold=mm_displacement_threshold,
                transform_suffix=".tfm",
                plot_title="Intravolume Motion Corrected Data: Voxel Percent Signal Change Carpet Plot + Displacements",
                output_directory=corrected_data_output_dir,
                slice_times_text_file=slice_times_text_file,
                also_save_png_file=True   
            )

            print(f"\nCalculating TSNR")
            CalculateTSNR(
                nifti_image_path=corrected_func_nifti_image,
                nifti_brain_mask_path=brain_mask_path,
                output_directory=corrected_data_output_dir,
                plot_title="Intravolume Motion Corrected Data: tSNR Plot"
            )

            print(f"\nSeed to Voxel Correlation")
            SeedToVoxelCorrelation(
                nifti_image_path=corrected_func_nifti_image,
                nifti_image_mask_path=brain_mask_path,
                json_file_path=json_file,
                output_directory_path=corrected_data_output_dir,
                plot_title="Intravolume Motion Corrected Data: Seed to Voxel Correlation"
            )


        if fmriprep_func_nifti_image:
            fmriprep_data_output_dir: str = os.path.join(output_directory, "fmriprep_data_outputs")
            os.makedirs(fmriprep_data_output_dir, exist_ok=True)


            print(f"\nCreating Raw Data Carpet Plot")
            CarpetPlot(
                anatomical_image=anatomical_nifti_image,
                functional_image=fmriprep_func_nifti_image,
                json_file=json_file,
                reference_volume_image=reference_volume_path,
                transform_directory=transform_directory,
                displacement_threshold=mm_displacement_threshold,
                transform_suffix=".tfm",
                plot_title="fMRIPrep + Intravolume Motion Corrected Data: Voxel Percent Signal Change Carpet Plot + Displacements",
                output_directory=fmriprep_data_output_dir,
                slice_times_text_file=slice_times_text_file,
                also_save_png_file=True   
            )

            print(f"\nCalculating TSNR")
            CalculateTSNR(
                nifti_image_path=fmriprep_func_nifti_image,
                nifti_brain_mask_path=brain_mask_path,
                output_directory=fmriprep_data_output_dir,
                plot_title="fMRIPrep + Intravolume Motion Corrected Data: tSNR Plot"
            )

            print(f"\nSeed to Voxel Correlation")
            SeedToVoxelCorrelation(
                nifti_image_path=fmriprep_func_nifti_image,
                nifti_image_mask_path=brain_mask_path,
                json_file_path=json_file,
                output_directory_path=fmriprep_data_output_dir,
                plot_title="fMRIPrep + Intravolume Motion Corrected Data: Seed to Voxel Correlation"
            )


