import os 
import numpy as np 
import SimpleITK as sitk 
from collections import OrderedDict
from joblib import Parallel, delayed
from plotly import graph_objects as go

from extract_images import ExtractNiFTIImage
from get_slice_timing import GetSliceTiming
from upsample_volume import UpsampleReferenceVolume
from limit_voxel_intensity_range import LimitVoxelIntensityRange
from make_identity_transform import MakeIdentityTransform
from run_alignment import RunAlignments
from calculate_displacements import CalculateDisplacements


class IdentifyReferenceVolume:
    def __init__(self, 
                 nifti_image_path: str, 
                 json_file_path: str, 
                 run_environment: str, 
                 smsmireg_executable_path: str | None = None,
                 singularity_image_file: str | None = None,
                 output_directory_path: str = "outputs", 
                 new_spacing: tuple[float, float, float] = (1.236, 1.236, 1.236),
                 voxel_intensity_lower_bound: int = 50,
                 voxel_intensity_upper_bound: int = 1000,
                 head_radius: float = 50,
                 n_jobs: int = -1) -> None:

        os.makedirs(output_directory_path, exist_ok=True)

        num_volumes: int = sitk.ReadImage(nifti_image_path).GetSize()[3]
        print(f"Number of Volumes: {num_volumes}")
        if num_volumes <= 0:
            raise ValueError(
                "Input NiFTI Image Must Be 4D." \
                f"Your NiFTI's Dimensions:\n{sitk.ReadImage(nifti_image_path).GetSize()}"
            )

        slice_timing: OrderedDict[float, list[int]] = GetSliceTiming(json_data=json_file_path).return_slice_timing()

        volume_images: list[str] = ExtractNiFTIImage(
             input_nifti_image_path=nifti_image_path,
             output_directory_path=output_directory_path,
             file_prefix="volume_outputs",
             index_to_extract=-1,
             n_jobs=n_jobs
        ).return_images() # pyright: ignore[reportAssignmentType]


        results: list[tuple[int, list[float]]] = Parallel(n_jobs=n_jobs)(
            delayed(self.get_intravolume_motion)(
                volume_num=volume_num,
                volume_image_path=volume_images[volume_num],
                slice_timing=slice_timing,
                new_spacing=new_spacing,
                output_directory_path=output_directory_path,
                voxel_intensity_lower_bound=voxel_intensity_lower_bound,
                voxel_intensity_upper_bound=voxel_intensity_upper_bound,
                run_environment=run_environment,
                smsmireg_executable_path=smsmireg_executable_path,
                singularity_image_file=singularity_image_file,
                head_radius=head_radius
            )
           for volume_num in range(num_volumes)                  
        ) # pyright: ignore[reportAssignmentType]

        max_disp_per_volume: list[tuple[int, float]] = sorted([
            (volume_num, max(displacement_list))
            for volume_num, displacement_list in
            results
        ], key=lambda x: x[1]
        )

        self.reference_volume_index: int = max_disp_per_volume[0][0]
        max_disp_at_reference_volume: float = max_disp_per_volume[0][1]

        print(f"Selected Reference Volume Index: {self.reference_volume_index}")
        print(f"Max Displacement At this Reference Volume: {max_disp_at_reference_volume} mm")
        

        fig = go.Figure()
        fig.update_layout(
            title_text=f"{os.path.basename(nifti_image_path)}: Reference Volume Selection - Intravolume Motion Calculations",
        )

        y_list: list[float] = [
            displacement_value
            for _, displacement_list in results
            for displacement_value in displacement_list
        ]

        x_list: list[float] = [
            aquisition_num / (len(slice_timing) - 1)
            for aquisition_num, _ in enumerate(y_list)
        ]
        fig.add_trace(go.Scatter(
            x=x_list,
            y=y_list,
            mode="lines+markers",
            name="Displacement Values"
        ))

        refvol_x_list: list[float] = [x_val for x_val in x_list if int(x_val) == self.reference_volume_index]
        refvol_y_list: list[float] = list(np.interp(refvol_x_list, x_list, y_list))
        fig.add_trace(go.Scatter(
            x=refvol_x_list,
            y=refvol_y_list,
            mode="lines+markers",
            name="Displacement Values in Reference Volume"
        ))
        fig.update_xaxes(title_text="Displacement Order")
        fig.update_yaxes(title_text="Displacement (in Millimeters)")

        output_file_path: str = os.path.join(output_directory_path, "plotted_intravolume_displacements.html")
        fig.write_html(output_file_path)

        print(f"Plot saved to: {output_file_path}")

    def get_intravolume_motion(self, 
                               volume_num: int , 
                               volume_image_path: str,
                               slice_timing: OrderedDict[float, list[int]],
                               new_spacing: tuple[float, float, float],
                               output_directory_path: str,
                               voxel_intensity_lower_bound: int,
                               voxel_intensity_upper_bound: int,
                               run_environment: str, 
                               smsmireg_executable_path: str | None = None,
                               singularity_image_file: str | None = None,
                               head_radius: int = 50
                               ) -> tuple[int, list[float]]:

        UpsampleReferenceVolume(
            input_nifti_image=volume_image_path,
            output_file_path=os.path.join(output_directory_path, f"upsampled_{os.path.basename(volume_image_path)}"),
            new_spacing=new_spacing
        ).return_resampled_image()

        limited_upsampled_image_path: str = LimitVoxelIntensityRange(
            nifti_image_paths=[os.path.join(output_directory_path, f"upsampled_{os.path.basename(volume_image_path)}")],
            output_directory=output_directory_path,
            lower_bound=voxel_intensity_lower_bound, 
            upper_bound=voxel_intensity_upper_bound
        ).return_output_image_paths()[0]

        initial_transform_path: str = os.path.join(output_directory_path, os.path.basename(limited_upsampled_image_path).replace(".nii", "_identity.tfm"))
        MakeIdentityTransform(
            input_nifti_image=limited_upsampled_image_path,
            output_file_path=initial_transform_path
        )
        print(f"Identity Transform: {initial_transform_path}")

        all_transform_paths: list[str] = []
        for slice_group_num, target_slice_indices in enumerate(slice_timing.values()):
            output_transform_path: str = RunAlignments(
                reference_volume_path=limited_upsampled_image_path,
                target_volume_path=volume_image_path,
                target_slice_indices=target_slice_indices,
                initial_transform_path=initial_transform_path,
                working_directory=output_directory_path,
                output_transform_label=f"transform-{'{:04d}'.format(volume_num)}-{'{:04d}'.format(slice_group_num)}",
                run_environment=run_environment,
                smsmireg_executable_path=smsmireg_executable_path,
                singularity_image_file=singularity_image_file
            ).return_output_transform_path()

            all_transform_paths.append(output_transform_path)
        
        displacements: list[float] = CalculateDisplacements(
            transform_paths=all_transform_paths,
            output_file_path=os.path.join(output_directory_path, f"displacements-{'{:04d}'.format(volume_num)}.txt"),
            file_extension=".tfm",
            head_radius=head_radius,
            verbose=True,
            compare_to_first_transform=True
            ).return_displacements()

        return (volume_num, displacements)

    
    def return_reference_volume_index(self) -> int:
        return self.reference_volume_index


if __name__ == "__main__":
    import argparse
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Identify a suitable reference volume."
    )
    parser.add_argument(
        "--nifti_image_path",
        required=True,
        help="Must be 4D."
    )
    parser.add_argument(
        "--json_file_path",
        required=True
    )
    parser.add_argument(
        "--output_directory_path",
        required=False,
        default="outputs",
        help=f"Default: {os.path.abspath('outputs')}"
    )
    parser.add_argument(
        "--run_environment",
        required=False,
        choices=[
            "docker",
            "singularity",
            "local"
        ],
        default="docker",
        help=\
            "Sms-Mi-Reg can be run in a Docker image, a Singularity image " + \
            "or locally. Please enter one of the following options: " + \
            "'docker', 'singularity', or 'local'. " + \
            "Default: 'docker'"
    )
    parser.add_argument(
        "--smsmireg_executable_path",
        required=False,
        default=None,
        help=\
            "To run locally, please provide the executable " + \
            "Sms-Mi-Reg path on your local machine. This path is outputted " + \
            "by compiling retro-motion-measurement.cxx: " + \
            "https://github.com/ComputationalRadiology/sms-mi-reg/blob/main/retro-motion-measurement.cxx. "
    )
    parser.add_argument(
        "--singularity_image_file",
        required=False,
        default=None,
        help=\
            "To run in a Singularity image, please provide the path " + \
            "to the '.sif' image file."
    )
    parser.add_argument(
        "--n_jobs",
        required=False,
        type=int,
        default=os.cpu_count(),
        help=f"Default: {os.cpu_count()}"
    )
    args: argparse.Namespace = parser.parse_args()
    IdentifyReferenceVolume(
        nifti_image_path=os.path.abspath(args.nifti_image_path),
        json_file_path=os.path.abspath(args.json_file_path),
        output_directory_path=os.path.abspath(args.output_directory_path),
        run_environment=args.run_environment,
        smsmireg_executable_path=\
            os.path.abspath(args.smsmireg_executable_path)
            if args.smsmireg_executable_path else None,
        singularity_image_file=\
            os.path.abspath(args.singularity_image_file)
            if args.singularity_image_file else None,
        n_jobs=args.n_jobs
    )