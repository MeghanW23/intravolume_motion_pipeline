import os
import sys
import json 
import subprocess
import numpy as np 
from glob import glob
import SimpleITK as sitk
from collections import OrderedDict
from joblib import Parallel, delayed


def get_slice_timing(json_path):

    def find_matching_indexes(numbers):
        
        num_index_map = {}
    
        for index, number in enumerate(numbers):
            if number in num_index_map:
                num_index_map[number].append(index)
            else:
                num_index_map[number] = [index]
    
        return {number: indexes for number, indexes in num_index_map.items() if len(indexes) > 1}

    slice_timing = []
    with open(json_path) as f:
        slice_timing = json.load(f)['SliceTiming']
    
    return OrderedDict(sorted(find_matching_indexes(slice_timing).items()))


def extract_volumes(loaded_nifti, working_directory, num_volumes, output_file_prefix = "volume_outputs"):
    
    volume_size = list(loaded_nifti.GetSize())[:3] + [0]

    for i in range(num_volumes):

        output_file = os.path.join(working_directory, f"{output_file_prefix}-{'{:04d}'.format(i)}.nii")

        extract = sitk.ExtractImageFilter()
        extract.SetSize(volume_size)
        extract.SetIndex((0,0,0,i))
        volume_image = extract.Execute(loaded_nifti)

        writer = sitk.ImageFileWriter()
        writer.SetFileName(output_file)
        writer.Execute(volume_image)
    
    return sorted(glob(os.path.join(working_directory, f"{output_file_prefix}-*.nii")))


def extract_slices(volume_num, volume_image, working_directory, num_slices, output_file_prefix = "slice_outputs"):
        
    slice_size = list(volume_image.GetSize())[:2] + [1]
    
    for i in range(num_slices):

        output_file = os.path.join(working_directory, f"{output_file_prefix}-{'{:04d}'.format(volume_num)}-{'{:03d}'.format(i)}.nii")

        slice_volume = sitk.RegionOfInterest(
            volume_image, 
            slice_size, 
            (0, 0, i) 
        )
        
        writer = sitk.ImageFileWriter()
        writer.SetFileName(output_file)
        writer.Execute(slice_volume)
    
    return sorted(glob(os.path.join(working_directory, f"{output_file_prefix}-*.nii")))


def make_identity_transform(working_directory, reference_volume_path, volume_num=None): 

    output_path = os.path.join(working_directory, f"identity-centered.tfm")
    if volume_num:
        output_path = os.path.join(working_directory, f"identity-centered-{'{:04d}'.format(volume_num)}.tfm")
    
    reference_image = sitk.ReadImage(reference_volume_path)
    reference_size = reference_image.GetSize()

    image_center = reference_image.TransformContinuousIndexToPhysicalPoint([ 
        (index - 1) / 2.0 
        for index in reference_size
    ])

    transform = sitk.AffineTransform(3)
    transform.SetIdentity()
    transform.SetCenter(image_center)


    # Reading and writing SimpleITK transforms on E3 with the project development conda env (rsfMRI_SMC_mc_env) 
    # has been causing Segmentation faults. Writing directly to the file is a work-around. 
    # To read and write SimpleITK transforms using the SimpleITK python module, 
    # run using the motion characterization conda env: 'intravolume_motion_pipeline_env' or 'pyspark_env'. 
    with open(output_path, mode='w') as file:
        file.write("#Insight Transform File V1.0\n")
        file.write("#Transform 0\n")
        file.write("Transform: AffineTransform_double_3_3\n")
        file.write("Parameters: " + ' '.join([str(float(param)) for param in transform.GetParameters()]) + '\n')
        file.write("FixedParameters: " + ' '.join([str(param) for param in transform.GetFixedParameters()]))
        file.write('\n')  

    return output_path  


def run_subprocess(command, verbose=False):
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


def dicom_to_nifti(working_directory, dcm2niix_path, dicom_directory, return_one_result = True, target_sequence_number = None):

    nifti_pattern: str = os.path.join(working_directory, f"*_{target_sequence_number}.nii.gz") if target_sequence_number else os.path.join(working_directory, f"*.nii.gz")
    json_pattern: str = os.path.join(working_directory, f"*_{target_sequence_number}.json") if target_sequence_number else os.path.join(working_directory, f"*.json")

    run_subprocess([
        dcm2niix_path, 
        "-o", working_directory, 
        "-z", "y", 
        "-b","y",
        "-ba","n", 
        "-w", "1", 
        dicom_directory
    ])

    found_nifti_images = sorted(glob(nifti_pattern))
    found_json_images = sorted(glob(json_pattern))

    if len(found_nifti_images) == 0:
        print("Could not find any nifti files matching:")
        print(nifti_pattern)
        sys.exit(1)
    elif len(found_json_images) == 0:
        print("Could not find any json files matching:")
        print(json_pattern)
        sys.exit(1)

    if return_one_result:
        if len(found_nifti_images) > 1:
            print(f"\n\nWARNING: More than one nifti file matching {nifti_pattern}:")
            print(found_nifti_images)
            print(f"Using file: {found_nifti_images[-1]}")

        
        elif len(found_json_images) > 1:
            print(f"\n\nWARNING: More than one json file matching {json_pattern}:")
            print(found_json_images)
            print(f"Using file: {found_json_images[-1]}\n\n")
    
        return found_nifti_images[-1], found_json_images[-1]    
    
    else:
        return found_nifti_images, found_json_images

def upsample_reference_volume(reference_volume_path, output_volume_path, spacing = (1.236, 1.236, 1.236)):
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
        print(f"Reference Volume New Size: {new_sz}")
        return resample_img(img, new_spacing, new_sz.tolist())

    # check if already upsampled
    input_spacing = [
        round(val, 5)
        for val in list(sitk.ReadImage(reference_volume_path).GetSpacing())
    ]
    target_spacing = [
        round(val, 5)
        for val in list(spacing)
    ]
    if input_spacing == target_spacing:
        print(f"Reference Volume Already Has Dimensions: {input_spacing}")
        return reference_volume_path
    
    upsampled_volume = resample_img_new_spacing(
        sitk.ReadImage(reference_volume_path),
        new_spacing=spacing
    )

    sitk.WriteImage(
        upsampled_volume, 
        output_volume_path
    )
    return output_volume_path
    
def get_parameters_from_tfm_file(transform_path):
        
        with open(transform_path, mode='r') as file:
            for line in file:
                if 'Parameters' in line and not 'FixedParameters' in line:
                    return [float(param) for param in line.split()[1:]]
    

def get_fixed_parameters_from_tfm_file(transform_path):
    
    with open(transform_path, mode='r') as file:
        for line in file:
            if 'FixedParameters' in line:
                return [float(fparam) for fparam in line.split()[1:]]


def get_motion_params(transform_paths, output_directory):
    
    timeseries_params = [get_parameters_from_tfm_file(transform_path) for transform_path in transform_paths if not 'identity-centered' in transform_path]
    
    output_path = os.path.join(output_directory, "parameters.txt")
    with open(output_path, mode='w') as file:
        for param_list in timeseries_params:
            file.write(' '.join([str(param) for param in param_list]) + '\n')
        file.write('\n')
    
    return output_path


def compute_displacement(transform1_path, transform2_path, radius=50):
    
    print(f"Calculating Displacement Between: {os.path.basename(transform1_path)} and {os.path.basename(transform2_path)}.")

    transform1 = sitk.VersorRigid3DTransform()
    transform1.SetParameters(get_parameters_from_tfm_file(transform1_path))
    transform1.SetFixedParameters(get_fixed_parameters_from_tfm_file(transform1_path))

    transform2 = sitk.VersorRigid3DTransform()
    transform2.SetParameters(get_parameters_from_tfm_file(transform2_path))
    transform2.SetFixedParameters(get_fixed_parameters_from_tfm_file(transform2_path))

    A0 = np.asarray(transform2.GetMatrix()).reshape(3, 3)
    c0 = np.asarray(transform2.GetCenter())
    t0 = np.asarray(transform2.GetTranslation())

    A1 = np.asarray(transform1.GetInverse().GetMatrix()).reshape(3, 3)
    c1 = np.asarray(transform1.GetInverse().GetCenter())
    t1 = np.asarray(transform1.GetInverse().GetTranslation())

    combined_mat = np.dot(A0,A1)
    combined_center = c1
    combined_translation = np.dot(A0, t1+c1-c0) + t0+c0-c1

    euler3d = sitk.Euler3DTransform()
    euler3d.SetCenter(combined_center)
    euler3d.SetTranslation(combined_translation)
    euler3d.SetMatrix(combined_mat.flatten())

    parms = np.asarray(euler3d.GetParameters())

    return abs(parms[0]*radius) + abs(parms[1]*radius) + \
        abs(parms[2]*radius) + abs(parms[3]) + abs(parms[4]) + abs(parms[5])


def run_sms_mi_reg(working_directory, sms_mi_reg_executable_path,reference_volume_path, initial_transform_path, slice_paths, volume_num, slice_group_num):
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
    --optimizer           Choice of optimizer (LN_COBYLA, LN_BOBYQA, LN_NELDERMEAD, LN_SBPLX). Default is LN_BOBYQA. [default: "LN_BOBYQA"]
    --maxiter             Maximum number of optimizer iterations. Default is 1000. [default: 1000]
    """
    
    output_transform_label = f"align-{'{:04d}'.format(volume_num)}-{'{:04d}'.format(slice_group_num)}"
    
    run_subprocess([
        sms_mi_reg_executable_path,
        reference_volume_path,             # referenceVolume
        initial_transform_path,            # inputTransform
        output_transform_label] +          # outputTransformLabel
        slice_paths + [                    # inputSlices
        "--optimizer", "LN_SBPLX"          # optimizer
        ],                                                               
    verbose=True) 

    return os.path.join(working_directory, f"alignTransform_{output_transform_label}.tfm") 


def get_input_variables(config_file, verbose=False):

    variables = {}

    with open(config_file, mode='r') as f:
        for line in f:
            # skip lines that arent variable declarations
            if line[0] == "#":
                continue
            elif not line.strip():
                continue
            if not "=" in line:
                continue 
            variable = line.strip().split("=")[0].strip()
            value = line.strip().split("=")[1].split("#")[0].strip()
            variables[variable] = value
    
    if verbose:
        max_width = max([len(key) for key in variables.keys()])
        print(f"\nConfigurations from {config_file}:")
        print("------------------------------------------------------")
        for key, value in variables.items():
            print(f"{key:<{max_width}} = {value}")
        print("------------------------------------------------------")

    return variables


def decompress_dicoms(dcmdjpeg_path, dicom_directory, output_directory, serialize=False, verbose=False):
        
    def decompress_single_dicom(dicom_path, output_path, dcmdjpeg_path, verbose):
        command = [dcmdjpeg_path, dicom_path, output_path]
        if verbose:
            print(f'Running Command: {command}')
        subprocess.run(command)

    os.makedirs(output_directory, exist_ok=True)
    
    dicom_path_list = sorted(glob(os.path.join(dicom_directory, '*.dcm')))
    
    n_jobs = 1
    if not serialize:
        n_jobs = min(os.cpu_count(), len(dicom_path_list))

    print(f"Decompressing {len(dicom_path_list)} DICOMS with {n_jobs} Workers")
    Parallel(n_jobs=n_jobs)(
        delayed(decompress_single_dicom)(
            dicom_path=dicom_path,
            output_path=os.path.join(output_directory, f"{os.path.basename(dicom_path)}"),
            dcmdjpeg_path=dcmdjpeg_path,
            verbose=verbose
        )
            for dicom_path in dicom_path_list
    )

    if len(glob(os.path.join(output_directory, "*.dcm"))) == 0:
        raise RuntimeError("Decompression failed: no DICOM files found.")


def search_for_file(parent_directory, 
                    filename_pattern, 
                    return_one_result = False,
                    not_exist_ok=False):
    
    fullpath_pattern = os.path.join(parent_directory, filename_pattern)
    fullpath_pattern_results = sorted(glob(fullpath_pattern))
    
    if len(fullpath_pattern_results) == 0:
        if not_exist_ok:
            return None
        else:
            print(f"\nERROR: No files found with pattern: {fullpath_pattern}")
            exit(0)

    elif return_one_result:
        if len(fullpath_pattern_results) > 1:
            print(f"\n\nWARNING: More than one file found with pattern '{fullpath_pattern}':")
            print('\n'.join(fullpath_pattern_results))
            print(f"Using last result found: {fullpath_pattern_results[-1]}\n\n")

            return fullpath_pattern_results[-1]
        else:
            return fullpath_pattern_results[0]
    
    else:
        return fullpath_pattern_results
