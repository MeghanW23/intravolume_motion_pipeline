import os
import sys 
import json
import subprocess
import numpy as np
import SimpleITK as sitk 
from collections import OrderedDict

class GetReferenceVolume:

    def __init__(self,
                 sms_mi_reg_executable_path, 
                 nifti_image_path, 
                 json_file_path, 
                 working_directory_path,
                 motion_threshold_percent = 1,
                 reference_volume_spacing = (1.236, 1.236, 1.236)):
        
        print("\n-----")
        print(f"Inputted sms-mi-reg Executable Path: {sms_mi_reg_executable_path}")
        print(f"Nifti Image Path: {nifti_image_path}")
        print(f"JSON File Path: {json_file_path}")
        print(f"Working Directory: {working_directory_path}")
        print(f"Threshold as Percent of Voxel: {motion_threshold_percent}%")
        print(f"Reference Volume Spacing: {reference_volume_spacing}")
        print("-----\n")

        os.makedirs(working_directory_path, exist_ok=True)
        os.chdir(working_directory_path)
        
        # 1. Get Motion Threshold Based on Voxel Spacing
        threshold_in_mm = self.get_threshold_in_mm(json_path=json_file_path, threshold_as_percent_of_voxel=motion_threshold_percent)
        print(f"Threshold in mm: {threshold_in_mm}")

        # 2. Make Sure the Volume is 4D 
        num_volumes = sitk.ReadImage(nifti_image_path).GetSize()[3]
        print(f"Number of Volumes: {num_volumes}")
        if num_volumes <= 0:
            print(f"\nERROR: Input NiFTI Image Must Be 4D.")
            print(f"Your NiFTI's Dimensions:\n{sitk.ReadImage(nifti_image_path).GetSize()}")
            exit(0)

        # 3. Extract Slice Timing from JSON File 
        slice_timing = self.get_slice_timing(json_file_path)
        print(f"Slice Timing: {slice_timing}")
        
        # 4. Calculate Slice Group, Aquisiton Info 
        num_slice_groups_per_volume = len(slice_timing)
        print(f"Number of Slice Groups Per Volume: {num_slice_groups_per_volume}")
        num_aquisitions = num_slice_groups_per_volume * num_volumes
        print(f"Total Number of Aquisition: {num_aquisitions}")

        # 5. Extract 3D Volumes from 4D Input NiFTI Image
        volume_paths = self.extract_volumes(nifti_image_path, working_directory_path)
        print(f"Extracted {len(volume_paths)} 3D Volumes")
        if len(volume_paths) != num_volumes:
            print(f"\nERROR: len(volume_paths) ({len(volume_paths)}) != num_volumes ({num_volumes})")
            exit(0)
        
        # 6. Iterate Through Each 3D Volume 
        for volume_num, volume_path in enumerate(volume_paths):
            
            # 7. Extract 2D Slices from the 3D Volume
            slice_paths = self.extract_slices(volume_path, volume_num, working_directory_path)
            print(f"\nExtracted {len(slice_paths)} Slices from Volume {volume_num + 1} of {len(volume_paths)}")

            # 8. Upsample the 3D Volume 
            upsampled_volume_path = self.upsample_reference_volume(
                volume_path,
                reference_volume_spacing,
                working_directory_path
            )
            print(f"Upsampled Volume {volume_num + 1} of {len(volume_paths)}")

            # 9. Extract Identity Transform, Center of Rotation from the Upsampled 3D Volume
            identity_transform_path = self.make_identity_transform(upsampled_volume_path, working_directory_path)
            print(f"Identity Transform Path: {identity_transform_path}")

            rotation_center = self.get_rotation_center(identity_transform_path)
            print(f"Rotation Center: {rotation_center}")

            # 10. Iterate Through Slice Groups 
            transform_paths = [identity_transform_path]
            for slice_group_num, slice_nums in enumerate(slice_timing.values()):
                print(
                    "\n" +
                    f"Aligning Slice Group {'{:03d}'.format(slice_group_num + 1)} of {'{:03d}'.format(num_slice_groups_per_volume)} " +
                    f"in Volume {'{:04d}'.format(volume_num + 1)} of {'{:04d}'.format(len(volume_paths))}"
                )
                output_label = '{:04d}'.format(volume_num) + '-' + '{:04d}'.format(slice_group_num)

                # 11. Align Each Slice Group to the First Slice Group of the Volume (First Slice Group Aligns to Identity)
                self.sms_mi_reg(
                    smsmireg_executable_path=sms_mi_reg_executable_path,
                    reference_volume_path=upsampled_volume_path,
                    input_transform_path=identity_transform_path if slice_group_num == 0 else transform_paths[1],
                    output_transform_label=output_label,
                    input_slice_paths=[
                        os.path.join(working_directory_path, f"slice_outputs-{'{:04d}'.format(volume_num)}-{'{:03d}'.format(slice_num)}.nii")
                        for slice_num in slice_nums
                    ]
                )

                # 12. Get Created Transform Path, Add to List 
                output_transform_path = os.path.join(working_directory_path, "alignTransform_" + output_label + ".tfm")
                transform_paths.append(output_transform_path)
            
            # 13. Calculate the Displacements Between the Transform of First Slice Group of the Volume with Each Other Transform
            displacements = []
            for i, _ in enumerate(transform_paths):
                if i <= 1:
                    continue 
                else: 
                    displacement = self.calculate_displacements(
                        transform_path_1=transform_paths[1],
                        transform_path_2=transform_paths[i],
                        rotation_center=rotation_center
                    )
                    displacements.append(displacement)
                    print(f"Displacement: {displacement} mm")

            print(f"\nAll Displacements at Volume {volume_num + 1} of {len(volume_paths)}:")
            print(', '.join([str(value) for value in displacements]))

            # 14. Write Displacement Values At This Volume to a .txt File
            displacements_file = os.path.join(working_directory_path, f"displacements-at-volume-{'{:04d}'.format(volume_num)}.txt")
            self.write_displacements_to_file(
                displacements=displacements,
                file_path=displacements_file
            )
            print(f"Displacement Values Written To: {displacements_file}")
            
            # 15. If Any Displacement Values Exceed/Equal the mm Threshold, Continue To the Next Volume 
            if any(displacement_value >= threshold_in_mm for displacement_value in displacements):
                print(f"At Least One Displacement Value >= {threshold_in_mm}mm")

            # 15. If All of the Displacement Values Are Under the mm Threshold, Exit the Script 
            else:
                print(f"All Displacement Values < {threshold_in_mm}mm")
                print(f"Reference Volume Selected: {upsampled_volume_path}")
                self.found_reference_volume = upsampled_volume_path
                exit(0)
            
        print(f"NO GOOD REFERENCE VOLUME FOUND.")


    def get_slice_timing(self, json_path):

        def find_matching_indexes(numbers):
            
            num_index_map = {}
        
            for index, number in enumerate(numbers):
                if number in num_index_map:
                    num_index_map[number].append(index)
                else:
                    num_index_map[number] = [index]
        
            return {number: indexes for number, indexes in num_index_map.items() if len(indexes) > 1}

        with open(json_path) as f:
            json_data = json.load(f)
            if not 'SliceTiming' in json_data:
                print(f"'SliceTiming' Key Not In JSON File.")
                exit(0)
            else:
               return OrderedDict(sorted(find_matching_indexes(json_data['SliceTiming']).items()))


    def get_threshold_in_mm(self, json_path, threshold_as_percent_of_voxel, round_digits = 4):
        with open(json_path) as f:
            json_data = json.load(f)
            if not 'SpacingBetweenSlices' in json_data:
                print(f"'SpacingBetweenSlices' Key Not In JSON File.")
                exit(0) 
            else:
                return round(json_data['SpacingBetweenSlices'] * (threshold_as_percent_of_voxel / 100), round_digits)
        
    
    def extract_volumes(self, nifti_image_path, working_directory_path):

        nifti_image = sitk.ReadImage(nifti_image_path)
        nifti_dimensions = nifti_image.GetSize()
        
        volume_paths = []

        for volume_num in range(0, nifti_dimensions[3]):

            extract = sitk.ExtractImageFilter()
            extract.SetSize(
                (nifti_dimensions[0], nifti_dimensions[1], nifti_dimensions[2], 0)
            )
            extract.SetIndex(
                (0, 0, 0, volume_num)
            )

            volume_path = os.path.join(working_directory_path, f"volume_outputs-{'{:04d}'.format(volume_num)}.nii")
            sitk.WriteImage(
                extract.Execute(nifti_image),
                volume_path
            )
            volume_paths.append(volume_path)

        return volume_paths


    def extract_slices(self, volume_path, volume_num, working_directory_path):
        
        volume_img = sitk.ReadImage(volume_path)
        volume_dimensions = volume_img.GetSize()

        slice_paths = []
        
        for slice_num in range(0, volume_dimensions[2]):
            
            slice_img = sitk.RegionOfInterest(
                volume_img,
                (volume_dimensions[0], volume_dimensions[1], 1),
                (0, 0, slice_num)
            )
            
            slice_path = os.path.join(
                working_directory_path, 
                f"slice_outputs-{'{:04d}'.format(volume_num)}-{'{:03d}'.format(slice_num)}.nii"
            )
            sitk.WriteImage(
                slice_img,
                slice_path
            )
            slice_paths.append(slice_path)
        
        return slice_paths
    

    def make_identity_transform(self, reference_volume_path, working_directory_path):

        reference_volume = sitk.ReadImage(reference_volume_path)

        image_center = reference_volume.TransformContinuousIndexToPhysicalPoint(
            [(index-1)/2.0 for index in reference_volume.GetSize()] 
        )
                
        transform = sitk.AffineTransform(3)
        transform.SetIdentity()
        transform.SetCenter(image_center)

        transform_path = os.path.join(
            working_directory_path, 
            f"{os.path.basename(reference_volume_path).replace('.nii.gz', '').replace('.nii', '')}_identity-centered.tfm"
        )

        sitk.WriteTransform(
            transform,
            transform_path

        )
        return transform_path


    def get_rotation_center(self, identity_transform_path):
        with open(identity_transform_path, mode='r') as f:
            for line in f: 
                if 'FixedParameters' in line:
                    return [
                        float(param_str.strip())
                        for param_str in line.split(' ')[1:]
                    ]
                

    def upsample_reference_volume(self, reference_volume_path, spacing, working_directory_path):
        
        def resample_img(img, spacing, sz, interpolator = sitk.sitkLinear):
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

        
        def resample_img_new_spacing(img, new_spacing):
            spacing = np.array(img.GetSpacing())
            sz = np.array(img.GetSize())
            new_sz = np.floor(spacing / new_spacing * sz).astype(np.uint32)
            new_sz = 2*np.floor((new_sz+1)/2).astype(np.uint32)
            return resample_img(img, new_spacing, new_sz.tolist())


        upsampled_image = resample_img_new_spacing(
            sitk.ReadImage(reference_volume_path),
            new_spacing=spacing
        )
        image_path = os.path.join(working_directory_path, f"UPSAMPLED_{os.path.basename(reference_volume_path)}")
        sitk.WriteImage(
            upsampled_image,
            image_path
        )
        return image_path


    def run_subprocess(self, command, verbose=False):
        if verbose:
            print(f"Running Command:\n{command}")

        result = subprocess.run(
            command,
            text=True,
            capture_output=True)

        if result.returncode != 0:
            print(f"Command: {command} returned non-zero return-code: {result.returncode}")
            print(f"Stdout:\n{result.stdout}") 
            print(f"Stderr:\n{result.stderr}")
            sys.exit(1)
        
        if verbose:
            print(f"Command ran successfully.")
            print(f"Stdout:\n{result.stdout}") 
            if result.stderr:
                print(f"Stderr:\n{result.stderr}")


    def sms_mi_reg(self, smsmireg_executable_path, reference_volume_path, input_transform_path, output_transform_label, input_slice_paths):
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
        self.run_subprocess([
            smsmireg_executable_path,
            reference_volume_path,             # referenceVolume
            input_transform_path,            # inputTransform
            output_transform_label] +          # outputTransformLabel
            input_slice_paths + [                    # inputSlices
            "--optimizer", "LN_SBPLX"          # optimizer
            ],                                                               
        verbose=True) 
    

    def calculate_displacements(self, transform_path_1, transform_path_2, rotation_center, radius = 50):

        parameters1 = [0, 0, 0, 0, 0, 0]
        parameters2 = [0, 0, 0, 0, 0, 0]

        if not 'identity-centered' in os.path.basename(transform_path_1):        
            parameters1 = self.extract_parameters(transform_path_1)

        if not 'identity-centered' in os.path.basename(transform_path_2):    
            parameters2 = self.extract_parameters(transform_path_2)

        print(f"Comparing:\n{parameters1}\nVS\n{parameters2}")

        transform1 = self.create_euler_transform(parameters1, rotation_center)
        transform2 = self.create_euler_transform(parameters2, rotation_center)

        A0 = np.asarray(transform2.GetMatrix()).reshape(3, 3)
        c0 = np.asarray(transform2.GetCenter())
        t0 = np.asarray(transform2.GetTranslation())

        A1 = np.asarray(transform1.GetInverse().GetMatrix()).reshape(3, 3)
        c1 = np.asarray(transform1.GetInverse().GetCenter())
        t1 = np.asarray(transform1.GetInverse().GetTranslation())

        combined_mat = np.dot(A0,A1)
        combined_center = c1
        combined_translation = np.dot(A0, t1+c1-c0) + t0+c0-c1

        versorrigid3d = sitk.VersorRigid3DTransform()
        versorrigid3d.SetCenter(combined_center)
        versorrigid3d.SetTranslation(combined_translation)
        versorrigid3d.SetMatrix(combined_mat.flatten())

        euler3d = sitk.Euler3DTransform()
        euler3d.SetCenter(combined_center)
        euler3d.SetTranslation(combined_translation)
        euler3d.SetMatrix(combined_mat.flatten())

        # Compute displacement (Tisdall et al. 2012)
        params = np.asarray( euler3d.GetParameters() )
        theta = np.abs(np.arccos(0.5 * (-1 + np.cos(params[0]) * np.cos(params[1]) + \
                                        np.cos(params[0]) * np.cos(params[2]) + \
                                        np.cos(params[1]) * np.cos(params[2]) + \
                                        np.sin(params[0]) * np.sin(params[1]) * np.sin(params[2]))))
        drot = radius * np.sqrt((1 - np.cos(theta)) ** 2 + np.sin(theta) ** 2)
        dtrans = np.linalg.norm(params[3:])
        displacement = drot + dtrans

        return displacement
    

    def extract_parameters(self, transform_path):
        with open(transform_path, mode='r') as f:
            for line in f: 
                if 'Parameters' in line and 'Fixed' not in line:
                    return [
                        float(param_str.strip())
                        for param_str in line.split(" ")[1:]
                    ] 


    def create_euler_transform(self, parameters, rotation_center):

        # Create a VersorTransform to interpret the versor
        versor_transform = sitk.VersorRigid3DTransform()
        versor_transform.SetParameters(parameters)
        versor_transform.SetCenter(rotation_center)

        # Extract Euler angles (in radians) from the VersorTransform
        euler_angles = versor_transform.GetMatrix()
        euler_angles = np.array(euler_angles).reshape(3, 3)  # Convert to 3x3 matrix

        # Convert rotation matrix to Euler angles (ZYX convention)
        sy = np.sqrt(euler_angles[0, 0] ** 2 + euler_angles[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(euler_angles[2, 1], euler_angles[2, 2])
            y = np.arctan2(-euler_angles[2, 0], sy)
            z = np.arctan2(euler_angles[1, 0], euler_angles[0, 0])
        else:
            x = np.arctan2(-euler_angles[1, 2], euler_angles[1, 1])
            y = np.arctan2(-euler_angles[2, 0], sy)
            z = 0

        # Create the Euler3DTransform
        euler_transform = sitk.Euler3DTransform()
        euler_transform.SetRotation(x, y, z)  # Angles are in radians
        euler_transform.SetTranslation(parameters[3:])
        euler_transform.SetCenter(rotation_center)
        
        return euler_transform


    def write_displacements_to_file(self, displacements, file_path):
        with open(file_path, mode='w') as f:
            for displacement in displacements:
                f.write(str(displacement) + '\n')

    def return_reference_volume(self):
        return self.found_reference_volume 

if __name__ == "__main__":
    """
        
        python get_reference_volume.py \
            --nifti_file_path ../data/sub-006_ses-02_task-prerifg.nii.gz \
            --json_file_path ../data/sub-006_ses-02_task-prerifg.json \
            --threshold_as_percent_of_voxel 1 \
            --reference_volume_spacing 1.236 1.236 1.236
    
    """

    import argparse 
    parser = argparse.ArgumentParser(description="Select a Motion-Free Reference Volume")
    parser.add_argument(
        "--sms_mi_reg_executable_path",
        required=True
    )
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
        "--threshold_as_percent_of_voxel",
        required=False,
        type=float,
        default=15,
        help="Default = 15 Percent"
    )
    parser.add_argument(
        "--reference_volume_spacing",
        required=False,
        type=float,
        nargs=3,
        default=(1.236, 1.236, 1.236),
        help="Default: (1.236, 1.236, 1.236)"
    )
    
    args = parser.parse_args()
    GetReferenceVolume(
        sms_mi_reg_executable_path=os.path.abspath(args.sms_mi_reg_executable_path),
        nifti_image_path=os.path.abspath(args.nifti_file_path),
        json_file_path=os.path.abspath(args.json_file_path),
        working_directory_path=os.path.abspath(args.working_directory_path),
        motion_threshold_percent=args.threshold_as_percent_of_voxel,
        reference_volume_spacing=args.reference_volume_spacing
    )
