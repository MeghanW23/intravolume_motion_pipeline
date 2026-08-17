import os 
import sys
import shutil
import warnings
from typing import Sequence
import SimpleITK as sitk 
from collections import OrderedDict

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# in order of usage:
from decompress_dicoms import DecompressDicoms
from dicom_to_nifti import DicomToNifti
from get_slice_timing import GetSliceTiming
from extract_images import ExtractNiFTIImage
from identify_reference_volume import IdentifyReferenceVolume
from limit_voxel_intensity_range import LimitVoxelIntensityRange
from upsample_volume import UpsampleReferenceVolume
from make_identity_transform import MakeIdentityTransform
from run_alignment import RunAlignments
from calculate_displacements import CalculateDisplacements
from write_parameters_to_textfile import WriteParametersToTextFile
from get_motion_threshold import GetMotionThreshold
from graph_transform_directory import GraphTransformDirectory
from fourier_transform_on_displacement_values import FourierTransform

class CharacterizeIntraVolumeMotion:
    def __init__(self, 
                 dicom_directory: str | None = None, 
                 series_name: str | None = None,
                 nifti_image_path: str | None = None, # pyright: ignore[reportRedeclaration]
                 json_file_path: str | None = None, # pyright: ignore[reportRedeclaration]
                 working_directory: str = os.path.abspath('working'),
                 output_directory: str = os.path.abspath('outputs'),
                 run_environment: str = 'docker',
                 smsmireg_executable_path: str | None = None,
                 singularity_image_path: str | None = None,
                 n_jobs: int | None = os.cpu_count(),
                 motion_threshold: int = 10,
                 reference_volume_index: int | None = None, # pyright: ignore[reportRedeclaration],
                 limit_voxel_intensity: bool = True,
                 voxel_intensity_lower_bound: int | None = 50,
                 voxel_intensity_upper_bound: int | None = 1000,
                 dcmdjpeg_path: str = 'dcmdjpeg',
                 dcm2niix_path: str = 'dcm2niix',
                 upsample_reference_volume: bool = True,
                 reference_volume_spacing: Sequence[float] | None = (1.236, 1.236, 1.236),
                 head_radius: float = 50
                 ) -> None:
        
        print(f"\n========== Starting Intravolume Motion Characterization ========== ")
        print(f"DICOM Directory: {dicom_directory}")
        print(f"Series Name: {series_name}")
        print(f"NiFTI Image Path: {nifti_image_path}")
        print(f"JSON File Path: {json_file_path}")
        print(f"Working Directory: {working_directory}")
        print(f"Output Directory: {output_directory}")
        print(f"Sms-Mi-Reg Run Environment: {run_environment}")
        print(f"Compiled Sms-Mi-Reg File: {smsmireg_executable_path}")
        print(f"Singularity Image Path: {singularity_image_path}")
        print(f"N_Jobs: {n_jobs}")
        print(f"Motion Threshold: {motion_threshold}")
        print("======================================================================\n")

        self.validate_inputted_data(dicom_directory, nifti_image_path, json_file_path)

        os.makedirs(working_directory, exist_ok=True)
        os.makedirs(output_directory, exist_ok=True)
        os.chdir(working_directory)

        if dicom_directory != None: 
            """
            =======================================================
            DECOMPRESS DICOMS 
            =======================================================
            """
            decompression_directory: str = os.path.join(working_directory, "decompressed_dicoms")
            print(f"Decompressing DICOMs into directory: {decompression_directory}")
            DecompressDicoms(
                dicom_directory=dicom_directory,
                output_directory=decompression_directory,
                dcmdjpeg_path=dcmdjpeg_path,
                series_name=series_name
            )

            """
            =======================================================
            DO DICOM TO NIFTI
            =======================================================
            """
            dcm2niix_module: DicomToNifti = DicomToNifti(
                dicom_directory=decompression_directory,
                output_directory=working_directory,
                dcm2niix_path=dcm2niix_path
            )
            nifti_image_path: str = dcm2niix_module.return_nifti_image() # pyright: ignore[reportAssignmentType]
            json_file_path: str = dcm2niix_module.return_json_file() # pyright: ignore[reportAssignmentType]

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
        slice_timing_dir: str = os.path.join(output_directory, "motion-char_slice-timing-info")
        os.makedirs(slice_timing_dir, exist_ok=True)
        slice_timing_module: GetSliceTiming = GetSliceTiming(
            json_data=json_file_path, # pyright: ignore[reportArgumentType]
            output_json_timing_path=os.path.join(slice_timing_dir, "slice_timing.json")
        ) 
        slice_timing_module.print_slice_timing()
        slice_timing: OrderedDict[float, list[int]] = slice_timing_module.return_slice_timing()
        
        """
        =======================================================
        EXTRACT 3D VOLUMES FROM 4D NIFTI IMAGE
        =======================================================
        """
        print(f"\nExtracting 3D Images from the 4D NiFTI Image with {n_jobs} jobs")
        volume_paths: list[str] = ExtractNiFTIImage(
            input_nifti_image_path=nifti_image_path,  # type: ignore
            output_directory_path=working_directory,
            file_prefix="volume_outputs",
            n_jobs=n_jobs # type: ignore
        ).return_images()
        print(f"{len(volume_paths)} 3D Volumes Were Extracted.")

        """
        =======================================================
        GET DISPLACEMENT THRESHOLD IN MM 
        =======================================================
        """
        print(
            "Calculating the motion threshold in mm from " + \
            "the threshold as a percentage of the diagonal " + \
            f"of a single voxel: {motion_threshold}% "
        )
        mm_motion_threshold: float = GetMotionThreshold(
            nifti_image=nifti_image_path, # pyright: ignore[reportArgumentType]
            threshold_as_percent=motion_threshold
        ).return_mm_threshold()


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
                working_directory=os.path.join(working_directory, "reference_volume_script_outputs"),
                threshold_in_mm=mm_motion_threshold,
                run_environment=run_environment,
                smsmireg_executable_path=smsmireg_executable_path,
                singularity_image_file=singularity_image_path,
                voxel_intensity_lower_bound=voxel_intensity_lower_bound, # pyright: ignore[reportArgumentType]
                voxel_intensity_upper_bound=voxel_intensity_upper_bound,  # pyright: ignore[reportArgumentType]
                n_jobs=n_jobs # pyright: ignore[reportArgumentType]
            ).return_reference_volume_index()
            print(f"We will use reference volume index: {reference_volume_index}")

        """
        =======================================================
        LIMIT INTENSITY RANGE OF EACH VOLUME
        =======================================================
        """
        if limit_voxel_intensity:
            print(f"Limiting the Voxel Intensity in {len(volume_paths)} Volumes.")
            volume_paths: list[str] = LimitVoxelIntensityRange(
                nifti_image_paths=volume_paths,
                output_directory=working_directory,
                lower_bound=voxel_intensity_lower_bound, # pyright: ignore[reportArgumentType]
                upper_bound=voxel_intensity_upper_bound, # pyright: ignore[reportArgumentType]
                n_jobs=n_jobs # pyright: ignore[reportArgumentType]
            ).return_output_image_paths()
            print(f"Limited the Intensity in {len(volume_paths)} Volumes.")

        """
        =======================================================
        UPSAMPLE THE REFERENCE VOLUME
        =======================================================
        """
        input_reference_volume_path: str = volume_paths[reference_volume_index]
        self.reference_volume_path: str = ""
        if upsample_reference_volume:
            self.reference_volume_path: str = os.path.join(working_directory, f"upsampled_{os.path.basename(input_reference_volume_path)}") 
            UpsampleReferenceVolume(
                input_nifti_image=input_reference_volume_path,
                output_file_path=self.reference_volume_path,
                new_spacing=reference_volume_spacing # pyright: ignore[reportArgumentType]
            )
        else:
            self.reference_volume_path: str = input_reference_volume_path
        print(f"Reference Volume Path: {self.reference_volume_path}")
        print("Copying Reference Volume to Output Directory...")
        shutil.copy(
            src=self.reference_volume_path,
            dst=os.path.join(output_directory, "refvol_" + os.path.basename(self.reference_volume_path))
        )

        """
        =======================================================
        MAKE IDENTITY TRANSFORM FROM REFERENCE VOLUME
        =======================================================
        """
        print("Creating Identity Transform")
        identity_transform_path: str = os.path.join(working_directory, "identity-centered.tfm")
        MakeIdentityTransform(
            input_nifti_image=self.reference_volume_path,
            output_file_path=identity_transform_path
        )
        print(f"Created Identity Transform at: {identity_transform_path}")

        """
        =======================================================
        RUN ALIGNMENT LOOP: ITERATE THROUGH VOLUMES
        =======================================================
        """        
        transform_paths: list[str] = [identity_transform_path]
        for volume_num, volume_path in enumerate(volume_paths):
            transform_paths: list[str] = self.run_alignments_for_one_volume(
                volume_path=volume_path,
                working_directory=working_directory,
                volume_num=volume_num,
                num_volumes=len(volume_paths),
                slice_timing=slice_timing,
                reference_volume_path=self.reference_volume_path,
                transform_paths=transform_paths,
                run_environment=run_environment,
                executable_smsmireg_file=smsmireg_executable_path,
                singularity_image_path=singularity_image_path,
                n_jobs=n_jobs # pyright: ignore[reportArgumentType]
            )

        """
        =======================================================
        COPY TRANSFORM RESULTS TO OUTPUT DIRECTORY
        =======================================================
        """
        output_transform_directory: str = os.path.join(output_directory, "transforms")
        os.makedirs(output_transform_directory, exist_ok=True)
        print(f"Copying All Transforms into the Output Directory at: {output_transform_directory}")
        
        for file_path in transform_paths[1:]:
            if not os.path.exists(file_path):
                warnings.warn(
                    message=\
                        f"Could not find transform: {file_path}. " \
                        "It will not be copied into the transform directory."
                )
            else:
                shutil.copy(
                    src=file_path,
                    dst=os.path.join(
                        output_transform_directory,
                        os.path.basename(file_path)
                    )
                )
        print("All Transforms Copied.")

        """
        =======================================================
        CALCULATE DISPLACEMENTS FROM THE OUTPUT TRANSFORMS
        =======================================================
        """
        print("Computing Displacements Between Transforms.")
        CalculateDisplacements(
            transform_directory=output_transform_directory,
            output_file_path=os.path.join(output_directory, "displacements.txt"),
            head_radius=head_radius
        )

        """
        =======================================================
        WRITE PARAMETER RESULTS TO TEXT FILES 
        =======================================================
        """
        versor_param_file: str = os.path.join(output_directory, "versor-parameters.txt")
        print(f"Writing Versor Parameters to Text File: {versor_param_file}")
        WriteParametersToTextFile(
            transform_directory=output_transform_directory,
            output_file_path=versor_param_file,
            output_rotation_unit='versor'
        )

        radian_param_file: str = os.path.join(output_directory, "radian-parameters.txt")
        print(f"Writing Radian Parameters to Text File: {radian_param_file}")
        WriteParametersToTextFile(
            transform_directory=output_transform_directory,
            output_file_path=radian_param_file,
            output_rotation_unit='radian'
        )

        degree_param_file: str = os.path.join(output_directory, "degree-parameters.txt")
        print(f"Writing Degree Parameters to Text File: {degree_param_file}")
        WriteParametersToTextFile(
            transform_directory=output_transform_directory,
            output_file_path=degree_param_file,
            output_rotation_unit='degrees'
        )

        graph_directory: str = os.path.join(output_transform_directory, "motion-char_graphs")
        os.makedirs(graph_directory, exist_ok=True)
        """
        =======================================================
        GRAPH DISPLACEMENT AND PARAMETER RESULTS 
        =======================================================
        """
        graph_plot_path: str = os.path.join(graph_directory, "parameter-plot.html")
        print(f"Plotting results to: {graph_plot_path}")
        GraphTransformDirectory(
            transform_directory=output_transform_directory,
            json_path=json_file_path, # pyright: ignore[reportArgumentType]
            output_file_path=graph_plot_path,
            input_rotation_unit="versor",
            threshold_in_mm=mm_motion_threshold
        )
        print(f"See Plot at: {graph_plot_path}")


        """
        =======================================================
        DO FOURIER TRANSFORM ON DISPLACEMENT RESULTS
        =======================================================
        """
        fourier_plot_path: str = os.path.join(graph_directory, "fourier-transform.html")
        print(f"Creating Fourier Transform at: {fourier_plot_path}")
        FourierTransform(
            transform_directory=output_transform_directory,
            nifti_image_path=nifti_image_path,
            json_file_path=json_file_path,
            output_file_path=fourier_plot_path,
            transform_suffix='.tfm',
            input_rotation_unit='versor',
            also_save_png_file=True
        )
        print(f"See Fourier Transform at: {fourier_plot_path}")
        

    def validate_inputted_data(self,
                            dicom_directory: str | None = None,
                            nifti_image_path: str | None = None,
                            json_file_path: str | None = None):

        error_msg: str = \
            "Please enter either a value for 'dicom_directory' OR " + \
            "a value for BOTH: 'nifti_image_path' AND 'json_file_path'."
        

        dicom_mode: bool = dicom_directory is not None
        nifti_mode: bool = nifti_image_path is not None and json_file_path is not None
        nifti_partial: bool = nifti_image_path is not None or json_file_path is not None

        # valid only if exactly one mode is fully satisfied, and no overlap
        if dicom_mode and (nifti_partial):
            raise ValueError(error_msg)
        if not dicom_mode and not nifti_mode:
            raise ValueError(error_msg)


    def run_alignments_for_one_volume(self, 
                                      volume_path: str, 
                                      working_directory: str,
                                      volume_num: int, 
                                      num_volumes: int,
                                      slice_timing: OrderedDict[float, list[int]],
                                      reference_volume_path: str,
                                      transform_paths: list[str],
                                      run_environment: str, 
                                      executable_smsmireg_file: str | None,
                                      singularity_image_path: str | None,
                                      n_jobs: int) -> list[str]:
        print(
            f"\n\n------ Processing Volume: {os.path.basename(volume_path)} " + \
            f"({volume_num + 1} of {num_volumes}) ------"
        )

        """
        =======================================================
        ITERATE THROUGH EVERY SLICE GROUP IN THE VOLUME
        =======================================================
        """
        for slice_group, slice_indices_list in enumerate(slice_timing.values()):
            """
            ============================================================
            RUN THE ALIGNMENT (USING THE PREVIOUS ALIGNMENT'S TRANSFORM)
            ============================================================
            """          
            print(
                f"\nProcessing Slice Group {slice_group + 1} of {len(slice_timing)} " + \
                f"in Volume {volume_num + 1} of {num_volumes}."   
            )
            output_transform_path: str = RunAlignments(
                reference_volume_path=reference_volume_path,
                target_volume_path=volume_path,
                target_slice_indices=slice_indices_list,
                initial_transform_path=transform_paths[-1],
                working_directory=working_directory,
                output_transform_label=f"{'{:04d}'.format(volume_num)}-{'{:04d}'.format(slice_group)}",
                run_environment=run_environment,
                smsmireg_executable_path=executable_smsmireg_file,
                singularity_image_file=singularity_image_path
            ).return_output_transform_path()

            print(f"Alignment Transform Created: {output_transform_path}")

            transform_paths.append(output_transform_path)

        return transform_paths

        
if __name__ == "__main__":
    import argparse
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=\
            "Characterize the intra-volume motion of n an SMS-Accelerated fMRI NiFTI timeseries. " + \
            "The output is a directory of SimpleITK VersorRigid3DTransforms characterizing the " + \
            "motion of each slice group aquisition across 6 rigid body transformation dimensions."
    )
    parser.add_argument(
        "--dicom_directory",
        required=False,
        help=\
            "Please enter either a value for --dicom_directory OR " + \
            " a value for BOTH: --nifti_image_file_path and --json_file_path.",
        default=None
    )
    parser.add_argument(
        "--series_name",
        required=False,
        default=None,
        help=\
            "This pipeline runs on single-subject, single run data. " + \
            "If your DICOM Directory has more than one run, please provide " + \
            "the name of the task/series you are wanting to run the analysis on. " + \
            "Any inputted series name must match the string in the DICOM metadata " + \
            "key 'SeriesDescription' (not case sensitive)."
    )
    parser.add_argument(
        "--nifti_image_file_path",
        required=False,
        help=\
            "4D NiFTI image. " + \
            "Please enter either a value for --dicom_directory OR " + \
            " a value for BOTH: --nifti_image_file_path and --json_file_path.",
        default=None
    )
    parser.add_argument(
        "--json_file_path",
        required=False,
        help=\
            "Please enter either a value for --dicom_directory OR " + \
            " a value for BOTH: --nifti_image_file_path and --json_file_path.",
        default=None
    )
    parser.add_argument(
        "--reference_volume_index",
        type=int,
        required=False,
        default=None,
        help=\
            "The index to the 3D reference volume within the input 4D timeseries. " \
            "Leave the default value of 'None' if you want the script to run an analysis to " \
            "select a reference volume. "
    )
    parser.add_argument(
        "--output_directory_path",
        required=False,
        default='outputs',
        help=f"Default: {os.path.abspath('outputs')}"
    )
    parser.add_argument(
        "--working_directory_path",
        required=False,
        default='working',
        help=f"Default: {os.path.abspath('working')}"
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
        "--singularity_image_path",
        required=False,
        default=None,
        help=\
            "To run in a Singularity image, please provide the path " + \
            "to the '.sif' image file."
    )
    parser.add_argument(
        "--n_jobs",
        required=False,
        default=os.cpu_count(),
        type=int,
        help="Default: the number of CPU cores on your machine."
    )
    parser.add_argument(
        "--motion_threshold",
        required=False,
        default=10,
        type=int,
        help=\
            "The percentage of the diameter of a single voxel. " + \
            "Default: 10 Percent."
    )
    args: argparse.Namespace = parser.parse_args()
    CharacterizeIntraVolumeMotion(
        dicom_directory=\
            os.path.abspath(args.dicom_directory)
            if args.dicom_directory else None,
        series_name=args.series_name,
        nifti_image_path=\
            os.path.abspath(args.nifti_image_file_path)
            if args.nifti_image_file_path else None,
        json_file_path=\
            os.path.abspath(args.json_file_path)
            if args.json_file_path else None,
        working_directory=os.path.abspath(args.working_directory_path),
        output_directory=os.path.abspath(args.output_directory_path),
        run_environment=args.run_environment,
        smsmireg_executable_path=\
            os.path.abspath(args.smsmireg_executable_path)
            if args.smsmireg_executable_path else None,
        singularity_image_path=\
                os.path.abspath(args.singularity_image_path)
                if args.singularity_image_path else None,
        n_jobs=args.n_jobs,
        motion_threshold=args.motion_threshold,
        reference_volume_index=args.reference_volume_index
    )
