import os
import sys
import shutil
import SimpleITK as sitk
from joblib import Parallel, delayed

import functions 

class MotionCharacterization():
    """
    
    Sms-mi-reg: Registration by maximization of mutual information for simultaneous multi-slice MRI
    
    See: 
        https://github.com/ComputationalRadiology/sms-mi-reg
        https://github.com/ComputationalRadiology/fmri-rest-pipe
    
    """

    def __init__(self, 
                 working_directory, 
                 output_directory, 
                 target_sequence_number, 
                 sms_mi_reg_executable_path,
                 dcm2niix_path,
                 nifti_path = None, 
                 json_path = None, 
                 dicom_directory = None, 
                 reference_volume_path = None,
                 reference_volume_index = None,
                 reference_volume_script = None,
                 motion_threshold_percent = 25,
                 upsample_reference_volume = True,
                 reference_volume_spacing = (1.236, 1.236, 1.236)):
        
        
        self.validate_data_inputs(dicom_directory, nifti_path, json_path)

        self.sms_mi_reg_executable_path = sms_mi_reg_executable_path

        self.working_directory = working_directory
        self.output_directory = output_directory
        self.nifti_path = nifti_path
        self.json_path = json_path
        
        print(f"\nMotion Characterization Script Inputs:")
        if dicom_directory:
            print(f"DICOM Directory: {dicom_directory}")
        else:
            print(f"NiFTI Image Path: {self.nifti_path}")
            print(f"JSON File Path: {self.json_path}")
        print(f"Sms-mi-reg Executable Path: {self.sms_mi_reg_executable_path}")
        print(f"dcm2niix Path: {dcm2niix_path}")
        print(f"Working Directory: {self.working_directory}")
        print(f"Output Directory: {self.output_directory}")
        print(f"Motion Threshold: {motion_threshold_percent}% of a voxel")
        print(f"Sequence Number: {target_sequence_number}")
        print(f"Reference Volume Path: {reference_volume_path}")
        print(f"Reference Volume Index: {reference_volume_index}")
        print(f"Reference Volume Script: {reference_volume_script}\n")
        

        os.makedirs(self.working_directory, exist_ok=True)
        os.makedirs(self.output_directory, exist_ok=True)
        
        if dicom_directory:
            print(f"Running dcm2niix")
            self.nifti_path, self.json_path = functions.dicom_to_nifti(
                dcm2niix_path=dcm2niix_path,
                working_directory=working_directory,
                dicom_directory=dicom_directory,
                target_sequence_number=target_sequence_number
            )

        # copy to output directory
        for file_path in (self.nifti_path, self.json_path):
            try:
                shutil.copy(
                    src=file_path,
                    dst=os.path.join(self.output_directory, os.path.basename(file_path))
                )
            except shutil.SameFileError:
                print(f"File Already Exists in the Output Directory.")

        print(f"NiFTI Image Path: {self.nifti_path}")
        print(f"Json File Path: {self.json_path}") 
        
        nifti_4d = sitk.ReadImage(self.nifti_path)
        image_dimensions = nifti_4d.GetSize()
        self.num_volumes = image_dimensions[3]
        self.num_slices = image_dimensions[2]
        print(f"Image Dimensions: {image_dimensions}")
        print(f"Number of Volumes: {self.num_volumes}")
        print(f"Number of Slices: {self.num_slices}")

        self.slice_timing = functions.get_slice_timing(self.json_path)
        print(f"Slice Timing: {self.slice_timing}")

        print(f"Extracting 3D Volumes from 4D NiFTI File")
        functions.extract_volumes(
            loaded_nifti=nifti_4d,
            working_directory=self.working_directory,
            num_volumes=self.num_volumes
        )

        self.reference_volume_path = self.get_reference_volume_path(
            inputted_reference_volume_path=reference_volume_path,
            reference_volume_index=reference_volume_index,
            reference_volume_script=reference_volume_script,
            motion_threshold_percent=motion_threshold_percent,
        )
        if upsample_reference_volume:
            print(f"Upsampling Reference Volume {os.path.basename(self.reference_volume_path)} To Spacing: {reference_volume_spacing}")

            upsampled_reference_volume_path = os.path.join(self.working_directory, f"UPSAMPLED_{os.path.basename(self.reference_volume_path)}")
            self.reference_volume_path = functions.upsample_reference_volume(
                self.reference_volume_path, 
                spacing=reference_volume_spacing,
                output_volume_path=upsampled_reference_volume_path
            )
            self.reference_volume_path = upsampled_reference_volume_path

            print(f"Upsampled Reference Volume at: {self.reference_volume_path}")


        identity_transform_path = functions.make_identity_transform(
            working_directory=self.working_directory,
            reference_volume_path=self.reference_volume_path
        )
        print(f"Identity Transform Path: {identity_transform_path}")
        
        os.chdir(self.working_directory)

        self.initial_transforms = [identity_transform_path]
        
        Parallel(n_jobs=1)(
            delayed(self.iterate_through_volumes)(
                volume_num=volume_num
            )
            for volume_num in range(self.num_volumes)
        )


        # collect all transforms
        transforms = []
        expected_num_transforms = self.num_volumes * len(self.slice_timing) + 1 # + 1 for identity transform
        if not os.path.exists(identity_transform_path):
            print(f"\n\nWARNING: Expected Transform: {identity_transform_path} does not exist\n\n")
        else:
            transforms.append(identity_transform_path)
        for volume_num in range(self.num_volumes):
            for slice_group_num in range(len(self.slice_timing)):
                transform_path = os.path.join(self.working_directory, f"alignTransform_align-{'{:04d}'.format(volume_num)}-{'{:04d}'.format(slice_group_num)}.tfm")
                if not os.path.exists(transform_path):
                    print(f"\n\nWARNING: Expected Transform: {transform_path} does not exist\n\n")
                else:
                    transforms.append(transform_path)
        print(f"I found {len(transforms)} transform paths") 
        print(f"I am missing {expected_num_transforms - len(transforms)} transform(s) of {expected_num_transforms} expected transforms")   

        print(f"Getting Parameters")
        param_txt_path = functions.get_motion_params(
            transform_paths=transforms,
            output_directory=self.output_directory
        )
        print(f"Output Parameters at: {param_txt_path}")
        
        print(f"Calculating Displacements")
        displacement_txt_path = self.calculate_timeseries_displacements(
            transform_paths=transforms
        )
        print(f"Output Displacements at: {displacement_txt_path}")


    def validate_data_inputs(self, dicom_directory, nifti_path, json_path):

        def exit_message():
            print("Please input either '--dicom_directory'")
            print("OR")
            print("'--nifti_file_path' and'--json_file_path'")
            sys.exit(1)

        if not dicom_directory and not nifti_path and not json_path:
            exit_message()
            
        if dicom_directory and nifti_path:
            exit_message()
        
        if dicom_directory and json_path:
            exit_message()
        
        if nifti_path and not json_path:
            exit_message()
        
        if json_path and not nifti_path:
            exit_message()


    def get_reference_volume_path(self, inputted_reference_volume_path, reference_volume_index, reference_volume_script, motion_threshold_percent):
        
        def validate_and_return(reference_volume_path):

            if not os.path.exists(reference_volume_path):
                print(f"Reference Volume: {reference_volume_path} does not exist.")
                sys.exit(1)

            print(f"Reference Volume Path: {reference_volume_path}")
            return reference_volume_path

        
        if inputted_reference_volume_path:
           return validate_and_return(inputted_reference_volume_path) 
        
        elif reference_volume_index:
            return validate_and_return(os.path.join(
                self.working_directory, 
                f"volume_outputs-{'{:04d}'.format(int(reference_volume_index))}.nii"
            )) 

        elif not reference_volume_index and reference_volume_script:
            sys.path.append(reference_volume_script)
            from get_reference_volume import GetReferenceVolume
            print(f"Running Script {os.path.basename(reference_volume_script)} To Get a Good Reference Volume.")
            
            return validate_and_return(GetReferenceVolume(
                sms_mi_reg_executable_path=self.sms_mi_reg_executable_path,
                nifti_image_path=self.nifti_path,
                json_file_path=self.json_path,
                working_directory_path=os.path.join(self.working_directory, "get_reference_volume_working"),
                motion_threshold_percent=motion_threshold_percent
            ).return_reference_volume())

        else:
            print(f"None of the following variables were given: reference_volume_path, reference_volume_index, reference_volume_script.")
            print(f"Thus, we will default to using the first volume as the reference volume.")
            reference_volume_index = 0
            return validate_and_return(os.path.join(self.working_directory, f"volume_outputs-{'{:04d}'.format(reference_volume_index)}.nii"))
        

    def iterate_through_volumes(self, volume_num):
        
        print(f"Starting Volume {volume_num + 1} of {self.num_volumes} ({round(((volume_num + 1) / self.num_volumes) * 100, 2)}% Done)")
        
        volume_path = os.path.join(self.working_directory, f"volume_outputs-{'{:04d}'.format(volume_num)}.nii")

        print(f"Extracting Slices from {os.path.basename(volume_path)}")
        functions.extract_slices(
            volume_num=volume_num,
            volume_image=sitk.ReadImage(volume_path),
            working_directory=self.working_directory,
            num_slices=self.num_slices
        ) 
        

        job_count = 1
        Parallel(n_jobs=job_count)(
            delayed(self.iterate_through_slice_groups)(
                volume_num=volume_num,
                slice_group_num=slice_group_num,
                slice_index_list=slice_index_list

            )
            for slice_group_num, (_, slice_index_list) in enumerate(self.slice_timing.items())
        )


    def iterate_through_slice_groups(self, volume_num, slice_group_num, slice_index_list):
        
        print(f"Registering Volume {volume_num + 1} Slice Group {slice_group_num + 1}")
            
        slice_paths = [
            os.path.join(self.working_directory, f"slice_outputs-{'{:04d}'.format(volume_num)}-{'{:03d}'.format(slice_index)}.nii")
            for slice_index in slice_index_list
        ]
        print("Slice Paths in this Slice Group:")
        print('\n'.join(slice_paths))

        transform_path = functions.run_sms_mi_reg(
            working_directory=self.working_directory,
            sms_mi_reg_executable_path=self.sms_mi_reg_executable_path,
            reference_volume_path=self.reference_volume_path,
            initial_transform_path=self.initial_transforms[-1],
            slice_paths=slice_paths,
            volume_num=volume_num,
            slice_group_num=slice_group_num
        )
        self.initial_transforms.append(transform_path)
        
    
    def calculate_timeseries_displacements(self, transform_paths):
        
        output_path = os.path.join(self.output_directory, "displacements.txt")
        
        displacements = []
        
        if 'identity-centered.tfm' in transform_paths[0]:
            transform_paths = transform_paths[1:]

        for i in range(len(transform_paths)):

            if i == 0:
                continue 
                
            transform1_path = transform_paths[i - 1]
            transform2_path = transform_paths[i]
            
            displacement = functions.compute_displacement(
                transform1_path=transform1_path,
                transform2_path=transform2_path
                
            )
            print(f"Displacement Between {os.path.basename(transform1_path)} and {os.path.basename(transform2_path)}: {displacement}mm")

            displacements.append(displacement)

        with open(output_path, mode='w') as file:
            file.write('\n'.join([str(val) for val in displacements]))
        
        return output_path


    def get_nifti_path(self):
        return self.nifti_path
    

    def get_json_path(self):
        return self.json_path
        
        

if __name__ == '__main__':
    """
    EXAMPLE USAGE:
        python /lab-share/Neuro-Cohen-e2/Groups/IRB-P00049401/mw_motion_pipeline/run_pipeline/pipeline_scripts/motion_characterization.py \
            --sms_mi_reg_executable_path /lab-share/Neuro-Cohen-e2/Groups/IRB-P00049401/mw_motion_pipeline/sms-mi-reg/build/sms-mi-reg \
            --dcm2niix_path /lab-share/Neuro-Cohen-e2/Public/software/fsl/bin/dcm2niix \
            --nifti_file_path /lab-share/Neuro-Cohen-e2/Groups/IRB-P00049401/mw_motion_pipeline/runs_and_data/participant_data/p004-ses-04_func-bold_task-NFB2_20250827181314_24.nii.gz \
            --json_file_path /lab-share/Neuro-Cohen-e2/Groups/IRB-P00049401/mw_motion_pipeline/runs_and_data/participant_data/p004-ses-04_func-bold_task-NFB2_20250827181314_24.json \
            --working_directory /lab-share/Neuro-Cohen-e2/Groups/IRB-P00049401/mw_motion_pipeline/runs_and_data/participant_runs/TEST_working \
            --output_directory /lab-share/Neuro-Cohen-e2/Groups/IRB-P00049401/mw_motion_pipeline/runs_and_data/participant_runs/TEST_outputs
    """
    
    import argparse

    default_working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working")

    default_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

    default_dcm2niix_path = "/lab-share/Neuro-Cohen-e2/Public/software/fsl/bin/dcm2niix"

    parser = argparse.ArgumentParser(
        description="To Characterize Intra-Volume Motion from an Inputted NiFTI or DICOM Timeseries."
    )
    parser.add_argument(
        "--sms_mi_reg_executable_path", 
        required=True, 
        help="The path to the sms-mi-reg executable for running sms-mi-reg outside a docker container."
    )
    parser.add_argument(
        "--dcm2niix_path", 
        required=True, 
        help=f"The path to your dcm2niix installation. Default: {default_dcm2niix_path}",
        default=default_dcm2niix_path
    )
    parser.add_argument(
        "--dicom_directory", 
        required=False,
        help="Give the path to a DICOM Directory or the paths to BOTH a NiFTI file and JSON file."
    )
    parser.add_argument(
        "--nifti_file_path", 
        required=False, 
        help="NiFTI File containing the data from a single run/series. Must be accompanied by a JSON file."
    )
    parser.add_argument(
        "--json_file_path", 
        required=False, 
        help="JSON File containing the JSON data from a single run/series. Must be accompanied by a NiFTI file."
    )
    parser.add_argument(
        "--working_directory", 
        required=False, 
        help=f"Working directory for storing working/temporary files. \
            Default: {default_working_dir}",
        default=default_working_dir
    )
    parser.add_argument(
        "--output_directory", 
        required=False, 
        help=f"Output directory for storing results. \
            Default: {default_output_dir}",
        default=default_output_dir
    )
    parser.add_argument(
        "--sequence_number",
        required=False,
        type=int,
        default=None,
        help="The sequence of the desired task to analyze. Only needed if inputting a multi-run dicom directory. Ex: '15' for func-bold_task-preRIFG_20250827181314_15.nii.gz and func-bold_task-preRIFG_20250827181314_15.json"
    )
    parser.add_argument(
        "--reference_volume_path",
        required=False,
        default=None,
        help="The path to the 3d reference volume (if not using a 3d volume from the input timeseries)."
    )
    parser.add_argument(
        "--reference_volume_index",
        required=False,
        type=int,
        default=None,
        help="The (zero-indexed) index of the desired 3D reference volume.\nDefault is the first\
            volume in the timeseries (reference_volume_index = 0)"
    )
    parser.add_argument(
        "--motion_threshold",
        required=False,
        type=int,
        default=50,
        help="Motion threshold as a percent of the size of a voxel.\nDefault = 50"
    )
    
    args = parser.parse_args()

    MotionCharacterization(
        dicom_directory=os.path.abspath(args.dicom_directory) if args.dicom_directory else None,
        nifti_path=os.path.abspath(args.nifti_file_path) if args.nifti_file_path else None,
        json_path=os.path.abspath(args.json_file_path) if args.json_file_path else None,
        working_directory=os.path.abspath(args.working_directory),
        output_directory=os.path.abspath(args.output_directory),
        target_sequence_number=args.sequence_number,
        sms_mi_reg_executable_path=os.path.abspath(args.sms_mi_reg_executable_path),
        dcm2niix_path=os.path.abspath(args.dcm2niix_path),
        reference_volume_path=os.path.abspath(args.reference_volume_path) if args.reference_volume_path else None,
        reference_volume_index=int(args.reference_volume_index) if args.reference_volume_index else None,
        motion_threshold_percent=int(args.motion_threshold)
    )