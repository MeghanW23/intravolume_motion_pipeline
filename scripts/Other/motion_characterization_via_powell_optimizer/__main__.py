from collections import OrderedDict
import os 
import sys
import SimpleITK as sitk

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from get_slice_timing import GetSliceTiming
from identify_reference_volume import IdentifyReferenceVolume
from extract_images import ExtractNiFTIImage
from limit_voxel_intensity_range import LimitVoxelIntensityRange
from make_identity_transform import MakeIdentityTransform
from upsample_volume import UpsampleReferenceVolume
from powell_optimizer import ParameterSearch
from get_motion_threshold import GetMotionThreshold
from graph_transform_directory import GraphTransformDirectory

class PowellMotionCharacterizationWrapper:
    def __init__(self, 
                 nifti_image_path: str, 
                 json_file_path: str, 
                 output_directory_path: str = 'outputs', 
                 working_directory_path: str = 'working',
                 reference_volume_index: int | None = None, # pyright: ignore[reportRedeclaration]
                 n_jobs: int = -1) -> None:

        os.makedirs(output_directory_path, exist_ok=True)
        os.makedirs(working_directory_path, exist_ok=True)

        nifti_image: sitk.Image = sitk.ReadImage(nifti_image_path) # pyright: ignore[reportArgumentType]
        dimensions: tuple[int, int, int, int] = nifti_image.GetSize()
        print(f"NiFTI Image Dimensions: {dimensions}")
        if len(dimensions) != 4:
            raise ValueError("Input data must be 4D.")

        """
        =======================================================
        GET SLICE TIMING 
        =======================================================
        """
        slice_timing_module: GetSliceTiming = GetSliceTiming(
            json_data=json_file_path, # pyright: ignore[reportArgumentType]
            output_json_timing_path=os.path.join(output_directory_path, "slice_timing.json"),
            output_txt_timing_path=os.path.join(output_directory_path, "slice_aquisition_times.txt")
        ) 
        slice_timing_module.print_slice_timing()
        slice_timing: OrderedDict[float, list[int]] = slice_timing_module.return_slice_timing()
        if reference_volume_index is None:
            reference_volume_index: int = IdentifyReferenceVolume(
                nifti_image_path=nifti_image_path,
                json_file_path=json_file_path,
                n_jobs=n_jobs,
                output_directory=os.path.join(output_directory_path, "reference_volume_selection_outputs")
            ).return_selected_reference_volume_index()

        """
        =======================================================
        FIND A REFERENCE_VOLUME
        =======================================================
        """
        if reference_volume_index == None:
            print("\nSelecting a Motion Free Reference Volume...")
            reference_volume_index: int = IdentifyReferenceVolume(
                nifti_image_path=nifti_image_path, # pyright: ignore[reportArgumentType]
                json_file_path=json_file_path, # pyright: ignore[reportArgumentType]
                output_directory=os.path.join(output_directory, "reference_volume_script_outputs"),
                n_jobs=n_jobs # pyright: ignore[reportArgumentType]
            ).return_selected_reference_volume_index()
            print(f"We will use reference volume index: {reference_volume_index}")

    
        """
        =======================================================
        EXTRACT 3D VOLUMES FROM 4D NIFTI IMAGE
        =======================================================
        """
        print(f"\nExtracting 3D Images from the 4D NiFTI Image with {n_jobs} jobs")
        volume_paths: list[str] = ExtractNiFTIImage(
            input_nifti_image_path=nifti_image_path,  # type: ignore
            output_directory_path=working_directory_path,
            file_prefix="volume_outputs",
            n_jobs=n_jobs # type: ignore
        ).return_images()
        print(f"{len(volume_paths)} 3D Volumes Were Extracted.")

        """
        =======================================================
        LIMIT INTENSITY RANGE OF EACH VOLUME
        =======================================================
        """
        print(f"Limiting the Voxel Intensity in {len(volume_paths)} Volumes.")
        volume_paths: list[str] = LimitVoxelIntensityRange(
            nifti_image_paths=volume_paths,
            output_directory=working_directory_path,
            lower_bound=10,
            upper_bound=1000,
            n_jobs=n_jobs
        ).return_output_image_paths()
        print(f"Limited the Intensity in {len(volume_paths)} Volumes.")

        """
        =======================================================
        UPSAMPLE THE REFERENCE VOLUME
        =======================================================
        """
        reference_volume_path: str = os.path.join(
            os.path.dirname(volume_paths[reference_volume_index]),
            f"upsampled_{os.path.basename(volume_paths[reference_volume_index])}"
        )
        UpsampleReferenceVolume(
            input_nifti_image=volume_paths[reference_volume_index],
            output_file_path=reference_volume_path,
            new_spacing=(1.236, 1.236, 1.236) # pyright: ignore[reportArgumentType]
        )
        

        """
        =======================================================
        MAKE IDENTITY TRANSFORM FROM REFERENCE VOLUME
        =======================================================
        """
        print("Creating Identity Transform")
        identity_transform_path: str = os.path.join(working_directory_path, "identity-centered.tfm")
        MakeIdentityTransform(
            input_nifti_image=reference_volume_path,
            output_file_path=identity_transform_path
        )
        print(f"Created Identity Transform at: {identity_transform_path}")
        fixed_parameters: list[float] = list(sitk.ReadTransform(identity_transform_path).GetFixedParameters())
        print(f"Fixed Parameters: {fixed_parameters}")

        """
        =======================================================
        GET ALIGNMENT FOR EACH SLICE GROUP 
        =======================================================
        """

        num_volumes: int = dimensions[-1]
        parameters: list[list[float]] = []
        reference_volume: sitk.Image = sitk.ReadImage(reference_volume_path)

        # Iterate through each volume 
        for volume_num in range(num_volumes):
            volume_path: str = volume_paths[volume_num]

            # Iterate through each slice group
            for slice_group_num, slice_num_list in enumerate(slice_timing.values()):

                print(
                    "\033[1m" + f"\nAligning Reference Volume to " \
                    f"Volume {volume_num + 1} of {num_volumes}: " \
                    f"Slice Group {slice_group_num + 1} of {len(slice_timing)}" + "\033[0m")

                # Load the slice images in the slice group
                print(f"Slices: {', '.join(str(slice_num) for slice_num in slice_num_list)}")
                slice_images: list[sitk.Image] = [
                    ExtractNiFTIImage(input_nifti_image_path=volume_path, 
                                      index_to_extract=slice_num, n_jobs=1
                                      ).return_images()[0][1]
                    for slice_num in slice_num_list
                ] # pyright: ignore[reportAssignmentType]
                
                print("Fixed Parameters:")
                print(', '.join(str(val) for val in fixed_parameters))

                input_parameters: list[float] = parameters[-1] if len(parameters) > 0 else [0, 0, 0, 0, 0, 0]
                print("Input Parameters:")
                print(', '.join(str(val) for val in input_parameters))

                # Run the powell optimizer, return the parameters that were found
                output_parameters: list[float] = ParameterSearch(
                    reference_volume=reference_volume,
                    slices=slice_images,
                    initial_versor_parameters=input_parameters,
                    fixed_parameters=fixed_parameters,
                    # must end in '.tfm' or '.txt':
                    output_transform_filename=f"{'{:04d}'.format(volume_num)}-{'{:04d}'.format(slice_group_num)}.tfm",
                    output_directory=output_directory_path,
                    verbose=True
                ).get_output_parameters()

                # Add the parameters to the parameter list
                parameters.append(output_parameters)

        """"
        =======================================================
        GET MOTION THRESHOLD
        =======================================================
        """
        mm_threshold: float = GetMotionThreshold(
            nifti_image=nifti_image,
            threshold_as_percent=10
        ).return_mm_threshold()

        GraphTransformDirectory(
            transform_directory=output_directory_path,
            json_path=json_file_path,
            output_file_path=os.path.join(output_directory_path, "parameter-plot.html"),
            input_rotation_unit="versor",
            threshold_in_mm=mm_threshold
        )

if __name__ == "__main__":
    import argparse
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=\
            "Characterize intravolume motion (via alignment of a reference volume " \
            "to each slice group in a NiFTI timeseries) using scipy's Powell Optimizer"
    )
    parser.add_argument("--nifti_image_path", required=True, help="Must be 4D.")
    parser.add_argument("--json_file_path", required=True)
    parser.add_argument("--output_directory_path", required=False, default="outputs", 
                        help=f"Default: {os.path.abspath('outputs')}")
    parser.add_argument("--working_directory_path", required=False, default="working", 
                        help=f"Default: {os.path.abspath('working')}")
    parser.add_argument("--reference_volume_index", required=False, type=int, default=None,
                        help=\
                            "Index of reference volume in input 4D NiFTI Image. "
                            "If none, we will select one.")
    parser.add_argument("--n_jobs", required=False, type=int, default=-1, help="Default: -1")
    args: argparse.Namespace = parser.parse_args()
    PowellMotionCharacterizationWrapper(
        nifti_image_path=os.path.abspath(args.nifti_image_path),
        json_file_path=os.path.abspath(args.json_file_path),
        output_directory_path=os.path.abspath(args.output_directory_path),
        working_directory_path=os.path.abspath(args.working_directory_path),
        reference_volume_index=args.reference_volume_index,
        n_jobs=args.n_jobs
    )