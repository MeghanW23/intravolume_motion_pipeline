import os
import SimpleITK as sitk

import functions 

class GetReferenceVolume:
    def __init__(self, 
                 sms_mi_reg_executable_path, 
                 nifti_image_path, 
                 json_file_path, 
                 working_directory_path,
                 motion_threshold_percent = 1):
        """
            Pipeline starts with an auto-calibration stage that uses a real-time SVR algorithm to find a reference volume. 

            The auto-calibration stage works as follows: 

            - The first fMRI volume is regarded as a provisional reference volume.

            - The slices of the second volume are registered to this volume using SVR as they are acquired. 

            - If the motion measurements on all slices of the second volume are below a predefined threshold,
            the first volume is confirmed as the reference volume as this indicates no motion was detected 
            within any of the slices of the first and second volume.

            - If the motion measurements do not pass the threshold condition, the first volume is discarded, the 
            second volume is regarded as the provisional reference, and all slices of the third volume are registered to the second volume. 

            - The motion measurements between the second and third volumes are then evaluated and compared against the threshold.

            - This process continues until no motion is detected within the slices of two consecutive volumes, 
            which means that the first volume of the two is chosen and used as the motion-free reference for SVR.

        """

        print(f"\nReference Volume Selection Script Inputs:")
        print(f"Inputted sms-mi-reg Executable Path: {sms_mi_reg_executable_path}")
        print(f"Inputted NiFTI Image Path: {nifti_image_path}")
        print(f"Inputted JSON File Path: {json_file_path}")
        print(f"Inputted Working Directory Path: {working_directory_path}")
        print(f"Inputted Motion Threshold (as a % of a voxel size): {motion_threshold_percent} %")
        
        os.makedirs(working_directory_path, exist_ok=True)
        os.chdir(working_directory_path)

        loaded_timeseries = sitk.ReadImage(nifti_image_path)
        num_volumes = loaded_timeseries.GetSize()[3]
        num_slices = loaded_timeseries.GetSize()[2]

        voxel_size = round(loaded_timeseries.GetSpacing()[0], 2)
        print(f"Voxel Size: {voxel_size} mm")

        motion_threshold = round(voxel_size * (float(motion_threshold_percent) / 100), 2)
        print(f"Motion Threshold: {motion_threshold} mm")

        slice_timing = functions.get_slice_timing(json_path=json_file_path)
        print(f"Slice Timing: {slice_timing}")

        volume_paths = functions.extract_volumes(
            loaded_nifti=loaded_timeseries,
            working_directory=working_directory_path,
            num_volumes=num_volumes
        )

        provisional_reference_volume_index = 0
        
        while True:

            # 1. the first fMRI volume is regarded as a provisional reference volume.
            provisional_reference_volume_path = volume_paths[provisional_reference_volume_index]
            print(f"Testing Provisional Reference Volume: {provisional_reference_volume_path}")
            
            identity_transform_path = functions.make_identity_transform(
                volume_num=provisional_reference_volume_index,
                working_directory=working_directory_path,
                reference_volume_path=provisional_reference_volume_path

            )
            print(f"Created Identity Transform: {identity_transform_path}")
            initial_transforms = [identity_transform_path]

            # 2. The slices of the second volume are registered to this volume using SVR as they are acquired. 
            moving_volume_index = provisional_reference_volume_index + 1
            moving_volume_path = volume_paths[moving_volume_index]
            functions.extract_slices(
                volume_num=moving_volume_index,
                volume_image=sitk.ReadImage(moving_volume_path),
                working_directory=working_directory_path,
                num_slices=num_slices
            )
            print(f"Testing Reference Volume Against the Slices of Volume: {moving_volume_path}")

            for slice_group_num, (_, slice_index_list) in enumerate(slice_timing.items()):

                slice_paths = [
                    os.path.join(working_directory_path, f"slice_outputs-{'{:04d}'.format(moving_volume_index)}-{'{:03d}'.format(slice_index)}.nii")
                    for slice_index in slice_index_list
                ]

                print(f"Running Sms-mi-reg for Slice Group {slice_group_num + 1} of {len(slice_timing)}")
                transform_path = functions.run_sms_mi_reg(
                    working_directory=working_directory_path,
                    sms_mi_reg_executable_path=sms_mi_reg_executable_path,
                    reference_volume_path=provisional_reference_volume_path,
                    initial_transform_path=initial_transforms[-1],
                    slice_paths=slice_paths,
                    volume_num=moving_volume_index,
                    slice_group_num=slice_group_num
                )

                initial_transforms.append(transform_path)
            
            displacements = []
            for i in range(len(initial_transforms)):
                if i < 2:
                    # skip identity transform,
                    # thhen compare first slice group transform with the  slice group transform after it
                    # ie. initial_transforms[i - 1] vs initial_transforms[i] where i = 2
                    continue 

                displacements.append(functions.compute_displacement(
                    transform1_path=initial_transforms[i - 1],
                    transform2_path=initial_transforms[i]
                ))
            print(f"Displacements between Volume {provisional_reference_volume_index} and Volume {moving_volume_index}:")
            print(displacements)

            if any(displacement_val >= motion_threshold for displacement_val in displacements):
                # 3a. If the motion measurements do not pass the threshold condition, the first volume is discarded, the 
                # second volume is regarded as the provisional reference, and all slices of the third volume are registered to the second volume.
                # The motion measurements between the second and third volumes are then evaluated and compared against the threshold. 
                # This process continues until no motion is detected within the slices of two consecutive volumes, 
                # which means that the first volume of the two is chosen and used as the motion-free reference for SVR.

                print(f"Provisitional Reference Volume ({os.path.basename(provisional_reference_volume_path)}) has >= {motion_threshold}mm motion")
                provisional_reference_volume_index += 1
            
            elif provisional_reference_volume_index >= len(volume_paths) - 1:
                raise RuntimeError("No motion-free reference volume found in the provided dataset.")

            else:
                            
                # 3b. If the motion measurements on all slices of the second volume are below a predefined threshold,
                # the first volume is confirmed as the reference volume as this indicates no motion was detected 
                # within any of the slices of the first and second volume.
                print(f"Good Reference Volume Found: {provisional_reference_volume_path}")
            
                self.reference_volume = provisional_reference_volume_path
                self.reference_volume_index = provisional_reference_volume_index

                return


    def return_reference_volume(self):
        return self.reference_volume

    def return_reference_volume_index(self):
        return self.reference_volume_index

if __name__ == '__main__':
    
    import argparse

    default_working_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working")

    parser = argparse.ArgumentParser(
        description="Get A Motion-Free Reference Volume"
    )
    parser.add_argument(
        "--sms_mi_reg_executable_path",
        required=True,
        help="The path to the executable sms-mi-reg cpp script."
    )
    parser.add_argument(
        "--nifti_image_path",
        required=True
    )
    parser.add_argument(
        "--json_file_path",
        required=True
    )
    parser.add_argument(
        "--working_directory_path",
        required=False,
        default=default_working_dir_path,
        help=f"Default: {default_working_dir_path}."
    )
    parser.add_argument("--motion_threshold_fraction", 
                        type=float, 
                        default=25,
                        help='Displacement threshold as percentage of voxel size.')
    args = parser.parse_args()


    """
    EXAMPLE USAGE:
        python get_reference_volume.py \
            --sms_mi_reg_executable_path /lab-share/Neuro-Cohen-e2/Groups/IRB-P00049401/mw_motion_pipeline/sms-mi-reg/build/sms-mi-reg \
            --nifti_image_path /lab-share/Neuro-Cohen-e2/Groups/IRB-P00049401/mw_motion_pipeline/runs_and_data/participant_data/p004-ses-04_func-bold_task-NFB2_20250827181314_24.nii.gz \
            --json_file_path /lab-share/Neuro-Cohen-e2/Groups/IRB-P00049401/mw_motion_pipeline/runs_and_data/participant_data/p004-ses-04_func-bold_task-NFB2_20250827181314_24.json \
            --working_directory_path ./test_reference_script/ \
            --motion_threshold_fraction 50
    """

    GetReferenceVolume(
        sms_mi_reg_executable_path=os.path.abspath(args.sms_mi_reg_executable_path),
        nifti_image_path=os.path.abspath(args.nifti_image_path),
        json_file_path=os.path.abspath(args.json_file_path),
        working_directory_path=os.path.abspath(args.working_directory_path),
        motion_threshold_percent=float(args.motion_threshold_fraction)
    )
