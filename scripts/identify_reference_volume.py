import os
import json
import subprocess
import numpy as np
from typing import Any 
import SimpleITK as sitk 
from collections import OrderedDict

from run_alignment import RunAlignments
from limit_voxel_intensity_range import LimitVoxelIntensityRange


class IdentifyReferenceVolume:

    def __init__(self, 
                 nifti_image_path: str, 
                 json_file_path: str, 
                 working_directory: str = 'working',
                 threshold_in_mm: float = 0.6,
                 reference_volume_spacing: tuple[float, float, float] = (1.236, 1.236, 1.236),
                 run_environment: str = "docker",
                 smsmireg_executable_path: str | None = None,
                 singularity_image_file: str | None = None,
                 voxel_intensity_lower_bound: int = 50,
                 voxel_intensity_upper_bound: int = 1000,
                 n_jobs: int = -1):
        
        print("\n-----")
        print(f"Nifti Image Path: {nifti_image_path}")
        print(f"JSON File Path: {json_file_path}")
        print(f"Working Directory: {working_directory}")
        print(f"Threshold in Milimeters: {threshold_in_mm}")
        print(f"Reference Volume Spacing: {reference_volume_spacing}")
        print(f"Run Environment: {run_environment}")
        print(f"Executable Path: {smsmireg_executable_path}")
        print(f"Singularity Path: {singularity_image_file}")
        print("-----\n")

        os.makedirs(working_directory, exist_ok=True)

        # Make Sure the Volume is 4D 
        num_volumes: int = sitk.ReadImage(nifti_image_path).GetSize()[3]
        print(f"Number of Volumes: {num_volumes}")
        if num_volumes <= 0:
            raise ValueError(
                "Input NiFTI Image Must Be 4D." \
                f"Your NiFTI's Dimensions:\n{sitk.ReadImage(nifti_image_path).GetSize()}"
            )

        # Extract Slice Timing from JSON File 
        slice_timing: OrderedDict[float, list[int]] = self.get_slice_timing(json_file_path)
        print(f"Slice Timing: {slice_timing}")
        
        # Calculate Slice Group, Aquisiton Info 
        num_slice_groups_per_volume: int = len(slice_timing)
        print(f"Number of Slice Groups Per Volume: {num_slice_groups_per_volume}")
        num_aquisitions: int = num_slice_groups_per_volume * num_volumes
        print(f"Total Number of Aquisition: {num_aquisitions}")

        # Extract 3D Volumes from 4D Input NiFTI Image
        volume_paths: list[str] = self.extract_volumes(nifti_image_path, working_directory)
        print(f"Extracted {len(volume_paths)} 3D Volumes")
        if len(volume_paths) != num_volumes:
            raise ValueError(
                f"len(volume_paths) ({len(volume_paths)}) != num_volumes ({num_volumes})"
            )

        # Limit the Intensity Range
        volume_paths: list[str] = LimitVoxelIntensityRange(
            nifti_image_paths=volume_paths,
            output_directory=working_directory,
            lower_bound=voxel_intensity_lower_bound,
            upper_bound=voxel_intensity_upper_bound,
            n_jobs=n_jobs
        ).return_output_image_paths()

        # Iterate Through Each 3D Volume 
        for volume_num, volume_path in enumerate(volume_paths):
            
            # Extract 2D Slices from the 3D Volume
            slice_paths: list[str] = self.extract_slices(volume_path, volume_num, working_directory)
            print(f"\nExtracted {len(slice_paths)} Slices from Volume {volume_num + 1} of {len(volume_paths)}")

            # Upsample the 3D Volume 
            upsampled_volume_path: str = self.upsample_reference_volume(
                volume_path,
                reference_volume_spacing,
                working_directory
            )
            print(f"Upsampled Volume {volume_num + 1} of {len(volume_paths)}")

            # Extract Identity Transform, Center of Rotation from the Upsampled 3D Volume
            identity_transform_path: str = self.make_identity_transform(upsampled_volume_path, working_directory)
            print(f"Identity Transform Path: {identity_transform_path}")

            rotation_center: list[float] = self.get_rotation_center(identity_transform_path)
            print(f"Rotation Center: {rotation_center}")

            # Iterate Through Slice Groups 
            transform_paths: list[str] = [identity_transform_path]
            for slice_group_num, slice_nums in enumerate(slice_timing.values()):
                print(
                    "\n" +
                    f"Aligning Slice Group {'{:03d}'.format(slice_group_num + 1)} of {'{:03d}'.format(num_slice_groups_per_volume)} " +
                    f"in Volume {'{:04d}'.format(volume_num + 1)} of {'{:04d}'.format(len(volume_paths))}"
                )
                output_label = '{:04d}'.format(volume_num) + '-' + '{:04d}'.format(slice_group_num)

                # Align Each Slice Group to the First Slice Group of the Volume (First Slice Group Aligns to Identity)
                output_transform_path: str = RunAlignments(
                    reference_volume_path=upsampled_volume_path,
                    target_volume_path=upsampled_volume_path,
                    target_slice_indices=slice_nums,
                    initial_transform_path=identity_transform_path,
                    working_directory=working_directory,
                    output_transform_label=output_label,
                    run_environment=run_environment,
                    smsmireg_executable_path=smsmireg_executable_path,
                    singularity_image_file=singularity_image_file
                ).return_output_transform_path()

                # Get Created Transform Path, Add to List 
                transform_paths.append(output_transform_path)
            
            # Calculate the Displacements Between the Transform of First Slice Group of the Volume with Each Other Transform
            displacements: list[float] = []
            for i, _ in enumerate(transform_paths):
                if i <= 1:
                    continue 
                else: 
                    displacement: float = self.calculate_displacements(
                        transform_path_1=transform_paths[1],
                        transform_path_2=transform_paths[i],
                        rotation_center=rotation_center
                    )
                    displacements.append(displacement)
                    print(f"Displacement: {displacement} mm")

            print(f"\nAll Displacements at Volume {volume_num + 1} of {len(volume_paths)}:")
            print(', '.join([str(value) for value in displacements]))

            # Write Displacement Values At This Volume to a .txt File
            displacements_file: str = os.path.join(working_directory, f"displacements-at-volume-{'{:04d}'.format(volume_num)}.txt")
            self.write_displacements_to_file(
                displacements=displacements,
                file_path=displacements_file
            )
            print(f"Displacement Values Written To: {displacements_file}")
            
            # If Any Displacement Values Exceed/Equal the mm Threshold, Continue To the Next Volume 
            if any(displacement_value >= threshold_in_mm for displacement_value in displacements):
                print(f"At Least One Displacement Value >= {threshold_in_mm}mm")
                

            # If All of the Displacement Values Are Under the mm Threshold, Exit the Script 
            else:
                print(f"All Displacement Values < {threshold_in_mm}mm")
                print(f"Reference Volume Selected: {upsampled_volume_path}")
                self.reference_volume_index = volume_num
                self.reference_volume_path = upsampled_volume_path
                return 

        raise ValueError(
            f"No Good Reference Volume Found for Image: {nifti_image_path} "
            f"at a Threshold of: {threshold_in_mm} mm. "
            f"Please Lower your Threshold and Try Again."
        )
        

    def get_slice_timing(self, json_path: str) -> OrderedDict[float, list[int]]:

        def find_matching_indexes(numbers) -> dict[float, list[int]]:
            
            num_index_map: dict[float, list[int]] = {}
        
            for index, number in enumerate(numbers):
                if number in num_index_map:
                    num_index_map[number].append(index)
                else:
                    num_index_map[number] = [index]
        
            return {number: indexes for number, indexes in num_index_map.items() if len(indexes) > 1}

        with open(json_path) as f:
            json_data: dict[str, Any] = json.load(f)
            if not 'SliceTiming' in json_data:
                print(f"'SliceTiming' Key Not In JSON File.")
                exit(0)
            else:
               return OrderedDict(sorted(find_matching_indexes(json_data['SliceTiming']).items()))
    
    def extract_volumes(self, nifti_image_path: str, working_directory: str) -> list[str]:

        nifti_image: sitk.Image = sitk.ReadImage(nifti_image_path)
        nifti_dimensions: tuple[int, int, int, int] = nifti_image.GetSize()
        
        volume_paths: list[str] = []

        for volume_num in range(0, nifti_dimensions[3]):

            extract: sitk.ExtractImageFilter = sitk.ExtractImageFilter()
            extract.SetSize(
                (nifti_dimensions[0], nifti_dimensions[1], nifti_dimensions[2], 0)
            )
            extract.SetIndex(
                (0, 0, 0, volume_num)
            )

            volume_path: str = os.path.join(working_directory, f"volume_outputs-{'{:04d}'.format(volume_num)}.nii")
            sitk.WriteImage(
                extract.Execute(nifti_image),
                volume_path
            )
            volume_paths.append(volume_path)

        return volume_paths


    def extract_slices(self, volume_path: str, volume_num: int, working_directory: str) -> list[str]:
        
        volume_img: sitk.Image = sitk.ReadImage(volume_path)
        volume_dimensions: tuple[float, float, float] = volume_img.GetSize()

        slice_paths: list[str] = []
        
        for slice_num in range(0, volume_dimensions[2]): # pyright: ignore[reportArgumentType]
            
            slice_img: sitk.Image = sitk.RegionOfInterest(
                volume_img,
                (volume_dimensions[0], volume_dimensions[1], 1),
                (0, 0, slice_num)
            )
            
            slice_path: str = os.path.join(
                working_directory, 
                f"slice_outputs-{'{:04d}'.format(volume_num)}-{'{:03d}'.format(slice_num)}.nii"
            )
            sitk.WriteImage(
                slice_img,
                slice_path
            )
            slice_paths.append(slice_path)
        
        return slice_paths
    

    def make_identity_transform(self, reference_volume_path: str, working_directory: str) -> str:

        reference_volume: sitk.Image = sitk.ReadImage(reference_volume_path)

        image_center: list[float] = reference_volume.TransformContinuousIndexToPhysicalPoint(
            [(index-1)/2.0 for index in reference_volume.GetSize()] 
        )
                
        transform: sitk.AffineTransform = sitk.AffineTransform(3)
        transform.SetIdentity()
        transform.SetCenter(image_center)

        transform_path: str = os.path.join(
            working_directory, 
            f"{os.path.basename(reference_volume_path).replace('.nii.gz', '').replace('.nii', '')}_identity-centered.tfm"
        )

        sitk.WriteTransform(
            transform,
            transform_path

        )
        return transform_path


    def get_rotation_center(self, identity_transform_path: str) -> list[float]:
        with open(identity_transform_path, mode='r') as f:
            for line in f: 
                if 'FixedParameters' in line:
                    return [
                        float(param_str.strip())
                        for param_str in line.split(' ')[1:]
                    ]
            else:
                raise ValueError(
                    f"Fixed Parameters not found in transform_path: {identity_transform_path}"
                )

    def upsample_reference_volume(self, reference_volume_path: str, spacing: tuple[float, float, float], working_directory: str) -> str:
        
        def resample_img(img: sitk.Image, spacing: tuple[float, float, float], sz: tuple[int, int, int], interpolator = sitk.sitkLinear) -> sitk.Image:
            # interpolator could be sitk.sitkLinear
            # interpolator could be sitk.sitkBSpline
            r = sitk.ResampleImageFilter()
            r.SetInterpolator(interpolator)
            r.SetOutputPixelType( img.GetPixelID() )
            r.SetDefaultPixelValue(0)
            r.SetOutputOrigin(img.GetOrigin())
            r.SetOutputSpacing(spacing)
            r.SetOutputDirection(img.GetDirection())
            r.SetSize(sz)
            return r.Execute(img)

        
        def resample_img_new_spacing(img: sitk.Image, new_spacing: tuple[float, float, float]) -> sitk.Image:
            spacing = np.array(img.GetSpacing())
            sz = np.array(img.GetSize())
            new_sz = np.floor(spacing / new_spacing * sz).astype(np.uint32)
            new_sz = 2*np.floor((new_sz+1)/2).astype(np.uint32)
            return resample_img(img, new_spacing, new_sz.tolist())


        upsampled_image: sitk.Image = resample_img_new_spacing(
            sitk.ReadImage(reference_volume_path),
            new_spacing=spacing
        )
        image_path: str = os.path.join(working_directory, f"UPSAMPLED_{os.path.basename(reference_volume_path)}")
        sitk.WriteImage(
            upsampled_image,
            image_path
        )
        return image_path


    def run_command(self, command: list[str], verbose: bool = False):
        if verbose:
            print(f"Running Command: {command}")
        
        result: subprocess.CompletedProcess = subprocess.run(
            command,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Command returned non-zero return code: {result.returncode}")
            print(f"stdout:{result.stdout}")
            print(f"stderr: {result.stderr}")
            exit(0)
        elif verbose:
            print(f"Command ran sucessfully.")
            

    def sms_mi_reg(self, working_directory: str, reference_volume_path: str, input_transform_path: str, output_transform_label: str, input_slice_paths: list[str]):
        """
        Usage: sms-mi-reg [--help] [--version] [--optimizer VAR] [--maxiter VAR] referenceVolume inputTransform outputTransformLabel inputSlices

        Positional arguments:
        referenceVolume       The volume that is moving to be aligned to the slices.
        inputTransform        The transform to initialize the alignment.
        outputTransformLabel  Name phrase used in the construction of the output transform file name.
        inputSlices           The list of file names of the fixed target slices. [nargs: 1 or more]

        Optional arguments:
        -h, --help            shows help message and exits
        -v, --version         prints version information and exits
        --optimizer           Choice of optimizer (LN_COBYLA, LN_BOBYQA, LN_NELDERMEAD, LN_SBPLX). Default is LN_SBPLX. [default: "LN_SBPLX"]
        --maxiter             Maximum number of optimizer iterations. Default is 1000. [default: 1000]
        """

        self.run_command(
            command=[
                "docker", "run", "--rm",
                "-v", f"{working_directory}:/data",
                "crl/sms-mi-reg", "sms-mi-reg",
                os.path.basename(reference_volume_path),
                os.path.basename(input_transform_path),
                output_transform_label
            ] + [os.path.basename(slice_path)
                 for slice_path in input_slice_paths
            ] + [
                "--optimizer", "LN_SBPLX"
            ],
            verbose=True
        )

    
    def calculate_displacements(self, transform_path_1: str, transform_path_2: str, rotation_center: list[float], radius: int = 50) -> float:

        parameters1: list[float] = [0, 0, 0, 0, 0, 0]
        parameters2: list[float] = [0, 0, 0, 0, 0, 0]

        if not 'identity-centered' in os.path.basename(transform_path_1):        
            parameters1: list[float] = self.extract_parameters(transform_path_1)

        if not 'identity-centered' in os.path.basename(transform_path_2):    
            parameters2: list[float] = self.extract_parameters(transform_path_2)

        print(f"Comparing:\n{parameters1}\nVS\n{parameters2}")

        transform1: sitk.Euler3DTransform = self.create_euler_transform(parameters1, rotation_center)
        transform2: sitk.Euler3DTransform = self.create_euler_transform(parameters2, rotation_center)

        A0: np.ndarray = np.asarray(transform2.GetMatrix()).reshape(3, 3)
        c0: np.ndarray = np.asarray(transform2.GetCenter())
        t0: np.ndarray = np.asarray(transform2.GetTranslation())

        A1: np.ndarray = np.asarray(transform1.GetInverse().GetMatrix()).reshape(3, 3)
        c1: np.ndarray = np.asarray(transform1.GetInverse().GetCenter())
        t1: np.ndarray = np.asarray(transform1.GetInverse().GetTranslation())

        combined_mat: np.ndarray = np.dot(A0,A1)
        combined_center: np.ndarray = c1
        combined_translation: np.ndarray = np.dot(A0, t1+c1-c0) + t0+c0-c1

        versorrigid3d: sitk.VersorRigid3DTransform = sitk.VersorRigid3DTransform()
        versorrigid3d.SetCenter(combined_center)
        versorrigid3d.SetTranslation(combined_translation)
        versorrigid3d.SetMatrix(combined_mat.flatten())

        euler3d: sitk.Euler3DTransform = sitk.Euler3DTransform()
        euler3d.SetCenter(combined_center)
        euler3d.SetTranslation(combined_translation)
        euler3d.SetMatrix(combined_mat.flatten())

        # Compute displacement (Tisdall et al. 2012)
        params: np.ndarray = np.asarray( euler3d.GetParameters() )
        theta: float = np.abs(
            np.arccos(0.5 * (-1 + np.cos(params[0]) * np.cos(params[1]) + \
            np.cos(params[0]) * np.cos(params[2]) + \
            np.cos(params[1]) * np.cos(params[2]) + \
            np.sin(params[0]) * np.sin(params[1]) * np.sin(params[2])))
        )
        drot: float = radius * np.sqrt((1 - np.cos(theta)) ** 2 + np.sin(theta) ** 2)
        dtrans: float = np.linalg.norm(params[3:]) # pyright: ignore[reportAssignmentType]
        displacement: float = drot + dtrans

        return displacement
    

    def extract_parameters(self, transform_path: str) -> list[float]:
        with open(transform_path, mode='r') as f:
            for line in f: 
                if 'Parameters' in line and 'Fixed' not in line:
                    return [
                        float(param_str.strip())
                        for param_str in line.split(" ")[1:]
                    ]
            else:
                raise ValueError(
                    f"Could not find parameters in transform path: {transform_path}"
                )


    def create_euler_transform(self, parameters: list[float], rotation_center: list[float]) -> sitk.Euler3DTransform:

        # Create a VersorTransform to interpret the versor
        versor_transform: sitk.VersorRigid3DTransform = sitk.VersorRigid3DTransform()
        versor_transform.SetParameters(parameters)
        versor_transform.SetCenter(rotation_center)

        # Extract Euler angles (in radians) from the VersorTransform
        euler_angles: np.ndarray = versor_transform.GetMatrix()
        euler_angles: np.ndarray = np.array(euler_angles).reshape(3, 3)  # Convert to 3x3 matrix

        # Convert rotation matrix to Euler angles (ZYX convention)
        sy: np.ndarray = np.sqrt(euler_angles[0, 0] ** 2 + euler_angles[1, 0] ** 2)
        singular: np.ndarray = sy < 1e-6

        if not singular:
            x = np.arctan2(euler_angles[2, 1], euler_angles[2, 2])
            y = np.arctan2(-euler_angles[2, 0], sy)
            z = np.arctan2(euler_angles[1, 0], euler_angles[0, 0])
        else:
            x = np.arctan2(-euler_angles[1, 2], euler_angles[1, 1])
            y = np.arctan2(-euler_angles[2, 0], sy)
            z = 0

        # Create the Euler3DTransform
        euler_transform: sitk.Euler3DTransform = sitk.Euler3DTransform()
        euler_transform.SetRotation(x, y, z)  # Angles are in radians
        euler_transform.SetTranslation(parameters[3:])
        euler_transform.SetCenter(rotation_center)
        
        return euler_transform


    def write_displacements_to_file(self, displacements: list[float], file_path: str):
        with open(file_path, mode='w') as f:
            for displacement in displacements:
                f.write(str(displacement) + '\n')

    
    def return_reference_volume_index(self) -> int:
        return self.reference_volume_index 

    def return_reference_volume_path(self) -> str:
        return self.reference_volume_path
    
if __name__ == "__main__":
    """
        
        python get_reference_volume.py \
            --nifti_file_path ../data/sub-006_ses-02_task-prerifg.nii.gz \
            --json_file_path ../data/sub-006_ses-02_task-prerifg.json \
            --threshold_as_percent_of_voxel 10 \
            --reference_volume_spacing 1.236 1.236 1.236
    
    """

    import argparse 
    parser = argparse.ArgumentParser(description="Select a Motion-Free Reference Volume")
    parser.add_argument(
        "--nifti_file_path",
        required=True
    )
    parser.add_argument(
        "--json_file_path",
        required=True
    )
    parser.add_argument(
        "--working_directory_path",
        required=False,
        default='working',
        help='Default: ./working'
    )
    parser.add_argument(
        "--threshold_in_mm",
        required=True,
        type=float
    )
    parser.add_argument(
        "--reference_volume_spacing",
        required=False,
        type=float,
        nargs=3,
        default=(1.236, 1.236, 1.236),
        help="Default: (1.236, 1.236, 1.236)"
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
    args = parser.parse_args()
    IdentifyReferenceVolume(
        nifti_image_path=os.path.abspath(args.nifti_file_path),
        json_file_path=os.path.abspath(args.json_file_path),
        working_directory=os.path.abspath(args.working_directory_path),
        threshold_in_mm=args.threshold_in_mm,
        reference_volume_spacing=args.reference_volume_spacing,
        run_environment=args.run_environment,
        smsmireg_executable_path=\
            os.path.abspath(args.smsmireg_executable_path)
            if args.smsmireg_executable_path else None,
        singularity_image_file=\
            os.path.abspath(args.singularity_image_file)
            if args.singularity_image_file else None
    )
