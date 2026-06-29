import os
import json
import subprocess 

class StartMotionCorrection:
    def __init__(self, 
                 motion_correction_main_script_dir, 
                 motion_correction_main_function_name,
                 output_dir,
                 input_nifti_image_path,
                 input_json_path,
                 bgremoved_input_nifti_image_path,
                 parameters_textfile_path,
                 recon_abruptmotion_output_filepath,
                 directliftandunliftcodes_dir,
                 operators_dir,
                 scrubbing_threshold,
                 matlab_installation_path):

        # set threads based on Slurm env variable, default to 1 if not set
        slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK', '1')  # 
        print(f"Slurm CPUs per Task: {slurm_cpus}")

        for env_var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'MATLAB_NUM_THREADS']:
            print(f"{env_var}: {slurm_cpus}")
            os.environ[env_var] = slurm_cpus
        
        slice_timing = self.get_slice_timing(input_json_path)
        print(f"Slice Timing from JSON File: {slice_timing}")

        sms_factor = self.get_sms_factor(slice_timing)
        print(f"SMS Factor: {sms_factor}")

        # scrubbing_threshold_in_mm = self.get_scrub_threshold_in_mm(input_json_path, scrubbing_threshold)
        scrubbing_threshold_in_mm = 0.4157
        print(f"Scrubbing Threshold in mm: {scrubbing_threshold_in_mm}mm")

        matlab_command = [
            matlab_installation_path, "-nosplash", "-nodesktop", "-r",
            f"addpath('{motion_correction_main_script_dir}'); \
            clear;close all;clc; \
            {motion_correction_main_function_name}( \
                '{output_dir}', \
                '{input_nifti_image_path}', \
                '{bgremoved_input_nifti_image_path}', \
                '{parameters_textfile_path}', \
                '{recon_abruptmotion_output_filepath}', \
                '{motion_correction_main_script_dir}', \
                '{directliftandunliftcodes_dir}', \
                '{operators_dir}', \
                '{str(sms_factor)}', \
                '{str(scrubbing_threshold_in_mm)}', \
                '{', '.join([str(val) for val in slice_timing])}'); \
            exit;" 
        ]
        print(f"Running Command: {matlab_command}")
        subprocess.run(matlab_command)


    def get_slice_timing(self, json_path):
        with open(json_path, mode='r') as f: 
            return [float(val) for val in list(json.load(f)['SliceTiming'])] 
    

    def get_sms_factor(self, slice_timing_list):
        num_slices = len(slice_timing_list)
        print(f"Number of Slices in Each 3D Volume: {num_slices}")

        num_slice_groups = len(set(slice_timing_list))
        print(f"Number of Slice Groups in Each 3D Volume: {num_slice_groups}")

        sms_factor = num_slices / num_slice_groups
        if not sms_factor.is_integer():
            print(f"\n\nWARNING: Your SMS Factor ({sms_factor}) is not an integer. \
                    \nThe SMS Factor is determined by the equation: \
                    \nsms_factor = num_slices / num_slice_groups\n\n")
        return sms_factor


    def get_scrub_threshold_in_mm(self, json_path, scrubbing_threshold):
        slice_thickness = 2.4
        with open(json_path, mode='r') as f: 
            slice_thickness = json.load(f)['SliceThickness']
        print(f"Slice Thickness: {slice_thickness}mm")

        return float(slice_thickness) * (float(scrubbing_threshold) / 100)
        


if __name__ == '__main__':
    """
    python start_motion_correction.py \
        --motion_correction_main_script_dir /lab-share/Neuro-Cohen-e2/Groups/IRB-P00049401/mw_motion_pipeline/rsfMRI_SMC_mc/main_scripts \
        --motion_correction_main_function_name main_cameraparams \
        --output_dir /lab-share/Neuro-Cohen-e2/Groups/IRB-P00049401/mw_motion_pipeline/testing_data/p004-ses-04_outputs \
        --input_nifti_image_path /lab-share/Neuro-Cohen-e2/Groups/IRB-P00049401/mw_motion_pipeline/testing_data/participant_data/p004-ses-04_func-bold_task-NFB2_20250827181314_24.nii.gz \
        --input_json_path /lab-share/Neuro-Cohen-e2/Groups/IRB-P00049401/mw_motion_pipeline/testing_data/participant_data/p004-ses-04_func-bold_task-NFB2_20250827181314_24.json \
        --bgremoved_input_nifti_image_path /lab-share/Neuro-Cohen-e2/Groups/IRB-P00049401/mw_motion_pipeline/testing_data/p004-ses-04_outputs/p004-ses-04_func-bold_task-NFB2_20250827181314_24_bgremoved.nii.gz \
        --parameters_textfile_path /lab-share/Neuro-Cohen-e2/Groups/IRB-P00049401/mw_motion_pipeline/testing_data/p004-ses-04_outputs/parameters.txt \
        --recon_abruptmotion_output_filepath /lab-share/Neuro-Cohen-e2/Groups/IRB-P00049401/mw_motion_pipeline/testing_data/p004-ses-04_outputs/recon_abruptmotion.nii.gz \
        --directliftandunliftcodes_dir /lab-share/Neuro-Cohen-e2/Groups/IRB-P00049401/mw_motion_pipeline/rsfMRI_SMC_mc/direct-liftandunlift-codes \
        --operators_dir /lab-share/Neuro-Cohen-e2/Groups/IRB-P00049401/mw_motion_pipeline/rsfMRI_SMC_mc/operators \
        --scrubbing_threshold 25
    """
    

    import argparse 

    parser = argparse.ArgumentParser(description="Script for calling the motion correction path")

    parser.add_argument(
        "--motion_correction_main_script_dir", 
        required=True)
    parser.add_argument(
        "--motion_correction_main_function_name", 
        required=True)
    parser.add_argument(
        "--output_dir", 
        required=True)
    parser.add_argument(
        "--input_nifti_image_path", 
        required=True)
    parser.add_argument(
        "--input_json_path",
        required=True)
    parser.add_argument(
        "--bgremoved_input_nifti_image_path", 
        required=True)
    parser.add_argument(
        "--parameters_textfile_path", 
        required=True)
    parser.add_argument(
        "--recon_abruptmotion_output_filepath", 
        required=True)
    parser.add_argument(
        "--directliftandunliftcodes_dir", 
        required=True)
    parser.add_argument(
        "--operators_dir",
        required=True)
    parser.add_argument(
        "--scrubbing_threshold",
        required=False,
        type=float,
        default=25,
        help="The FD threshold for scrubbing the data - input the desired percentage of the width of a voxel. \
            \nEx. 25 if threshold=25% the width of a voxel")
    parser.add_argument(
        "--matlab_installation_path",
        required=False,
        default="/programs/local/MATLAB/R2022b/bin/matlab")
    args=parser.parse_args()
    
    StartMotionCorrection(
        motion_correction_main_script_dir=os.path.abspath(args.motion_correction_main_script_dir),
        motion_correction_main_function_name=os.path.basename(args.motion_correction_main_function_name),
        output_dir=os.path.abspath(args.motion_correction_main_script_dir),
        input_nifti_image_path=os.path.abspath(args.input_nifti_image_path),
        input_json_path=os.path.abspath(args.input_json_path),
        bgremoved_input_nifti_image_path=os.path.abspath(args.bgremoved_input_nifti_image_path),
        parameters_textfile_path=os.path.abspath(args.parameters_textfile_path),
        recon_abruptmotion_output_filepath=os.path.abspath(args.recon_abruptmotion_output_filepath),
        directliftandunliftcodes_dir=os.path.abspath(args.directliftandunliftcodes_dir),
        operators_dir=os.path.abspath(args.operators_dir),
        scrubbing_threshold=args.scrubbing_threshold,
        matlab_installation_path=os.path.abspath(args.matlab_installation_path)

    )   

    