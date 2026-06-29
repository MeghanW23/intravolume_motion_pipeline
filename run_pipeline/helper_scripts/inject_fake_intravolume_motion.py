
import os
import json
import random
import numpy as np
from glob import glob
import nibabel as nib 
import SimpleITK as sitk
from collections import OrderedDict
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy.interpolate import NearestNDInterpolator

class MakeFakeMotion:
    
    def __init__(self, nifti_image_path, json_file_path, working_directory, output_directory, input_parameter_file=None):

        print(f"Inputted NiFTI Image Path: {nifti_image_path}")
        print(f"Inputted JSON File Path: {json_file_path}")
        print(f"Inputted Working Directory: {working_directory}")
        print(f"Inputted Output Directory: {output_directory}")
        os.makedirs(working_directory, exist_ok=True)
        os.makedirs(output_directory, exist_ok=True)
        
        img = sitk.ReadImage(nifti_image_path)
        dimensions = img.GetSize()
        num_volumes = dimensions[3]
        num_slices = dimensions[2]
        slice_timing = self.get_slice_timing(json_path=json_file_path)
        num_slice_groups = len(slice_timing)
        num_aquisitions = num_volumes * num_slice_groups
        
        parameters = {
            "X-Rotation": np.zeros(num_aquisitions),
            "Y-Rotation": np.zeros(num_aquisitions),
            "Z-Rotation": np.zeros(num_aquisitions),
            "X-Translation": np.zeros(num_aquisitions),
            "Y-Translation": np.zeros(num_aquisitions),
            "Z-Translation": np.zeros(num_aquisitions),
        }
        cumulative_parameters = {}
        print(f"Input NiFTI Image Dimensions:")
        print(', '.join([str(dim) for dim in dimensions]))
        print(f"Number of Volumes: {num_volumes}")
        print(f"Number of Slices Per Volume: {num_slices}")
        print(f"Number of Slice Groups Per Volume: {num_slice_groups}")
        print(f"Number of Aquisitions in the Timeseries: {num_aquisitions}")
        print("Slice Groups:")
        print(', '.join([str(slice_list) for slice_list in list(slice_timing.values())]))
        
        print("Motion Dimensions:")
        print(', '.join(parameters.keys()))

        print(f"\nStep One: Get / Create the Parameters")

        if not input_parameter_file:

            min_degree_rotation = 7.0 # degrees
            max_degree_rotation = 10.0 # degrees
            min_mm_translation = 2 # milimeters 
            max_mm_translation = 7.0 # milimeters 
            max_num_motion_events_per_dimension = 3
            min_motion_event_length = 5 # aquisitions 
            max_motion_event_length = 15 # aquisitions

            print(f"Minimum Rotation: {min_degree_rotation} deg")
            print(f"Maximum Rotation: {max_degree_rotation} deg")
            print(f"Minimum Translation: {min_mm_translation} mm")
            print(f"Maximum Translation: {max_mm_translation} mm")
            print(f"Maximum Number of Motion Events per Dimension: {max_num_motion_events_per_dimension}")
            print(f"Minimum Motion Event Length: {min_motion_event_length} Aquisitions")
            print(f"Maximium Motion Event Length: {max_motion_event_length} Aquisitions")

            for dimension_name in parameters.keys(): 

                num_motion_events = random.randint(1, max_num_motion_events_per_dimension)
                
                length_of_motion_events = [
                    random.randint(min_motion_event_length, max_motion_event_length)
                    for _ in range(num_motion_events)
                ]

                for motion_event_num in range(num_motion_events):
                    print(f"\nMaking Motion Event {motion_event_num + 1} of {num_motion_events} for {dimension_name}")
                    
                    frequency = round(random.uniform(0.25, 1), 2)
                    print(f"Frequency of the Motion Function: {frequency}")

                    motion_event_length = length_of_motion_events[motion_event_num]

                    period = self.get_random_period_in_timeseries(
                        start_range = num_aquisitions - motion_event_length,
                        event_length=motion_event_length,
                        num_slice_groups=num_slice_groups,
                        dimension_parameters=parameters[dimension_name]
                    )
                    print(f"Adding Motion to Aquisitions:")
                    print(', '.join([str(val) for val in period]))

                    motion_value = self.get_motion_value(
                        dimension_name=dimension_name,
                        min_degree_rotation=min_degree_rotation,
                        max_degree_rotation=max_degree_rotation,
                        min_mm_translation=min_mm_translation,
                        max_mm_translation=max_mm_translation
                    )
                    print(f"Motion Value: {motion_value}")

                    motion_function = self.make_motion_function(
                        period=period,
                        amplitude=motion_value/4,
                        frequency=frequency,
                        phase_shift=0
                    )
                    print(f"Motion Function:")
                    print(', '.join([str(val) for val in motion_function]))

                    print(f"Cumulative Motion Function:")
                    print(', '.join([str(val) for val in np.cumsum(motion_function)]))

                    flip_direction = random.choice([True, False])
                    if flip_direction:
                        motion_function = -motion_function

                    self.plot_motion_event(
                        title=f"{dimension_name} Motion Event {motion_event_num + 1} of {num_motion_events}",
                        dimension_name=dimension_name,
                        period=period,
                        motion_function=motion_function,
                        output_file=os.path.join(output_directory, f"{dimension_name}_motion-event-{'{:03d}'.format(motion_event_num + 1)}_plotted.png")
                    )

                    parameters[dimension_name][period[0]:period[-1] + 1] = motion_function
            
                cumulative_parameters[dimension_name] = np.cumsum(parameters[dimension_name])
        else:
            parameters, cumulative_parameters = self.read_parameters_from_file(input_parameter_file)   

           
        print(f"\nMotion Parameters Loaded/Created.")
        
        self.plot_motion_parameters(
            title="Motion Parameters",
            parameters=parameters,
            output_file=os.path.join(output_directory, "all_parameters_plotted.png")
        )

        self.plot_motion_parameters(
            title="Cumulative Motion Parameters",
            parameters=cumulative_parameters,
            output_file=os.path.join(output_directory, "all_cumulative_parameters_plotted.png")
        )

        self.write_parameters_to_textfile(
            parameters=cumulative_parameters,
            num_aquisitions=num_aquisitions,
            output_file_path=os.path.join(output_directory, "parameters.txt")
        )

        print(f"\nStep Two: Apply the Parameters")
        job_count = min(os.cpu_count(), num_volumes)

        self.cut_timeseries_into_slices(
            nifti_image_path=nifti_image_path,
            num_volumes=num_volumes,
            num_slices=num_slices,
            working_directory=working_directory,
            job_count=job_count
        )
        
        reference_volume_path = os.path.join(working_directory, "volume_outputs-0000.nii")

        identity_transform_path = self.make_identity_transform(
            reference_volume_path=reference_volume_path,
            working_directory=working_directory
        )
        print(f"Identity Transform at: {identity_transform_path}")

        fixed_parameters = self.get_fixed_parameters_from_tfm_file(identity_transform_path)
        print("Fixed Parameters:")
        print(', '.join([str(val) for val in fixed_parameters]))
    
        print(f"\nApplying Parameters to the Slices..")
        transform_directory = os.path.join(output_directory, "transforms")
        os.makedirs(transform_directory, exist_ok=True)
        
        Parallel(n_jobs=job_count)(
            delayed(self.apply_to_timeseries)(
                slice_timing=slice_timing,
                volume_num=volume_num,
                num_slice_groups=num_slice_groups,
                num_aquisitions=num_aquisitions,
                cumulative_parameters=cumulative_parameters, 
                fixed_parameters=fixed_parameters,
                transform_directory=transform_directory, 
                working_directory=working_directory,
                reference_volume_path=reference_volume_path
            )
            for volume_num in range(num_volumes)
        )
        print("\nAll Slices Resampled.")
        transforms = sorted(glob(os.path.join(transform_directory, "*tfm")))

        print(f"Calculating Displacements.")
        displacements = []
        for i in range(len(transforms)):
            if i == 0:
                continue 
            displacements.append(self.compute_displacement(
                transform1_path=transforms[i - 1],
                transform2_path=transforms[i],

            ))
        
        print(f"\nDisplacements Calculated.")

        self.write_displacement_to_textfile(
            displacements=displacements,
            output_file_path=os.path.join(output_directory, "displacements.txt")
        )
        self.plot_displacements(
            title="Displacements",
            displacements=displacements,
            output_file=os.path.join(output_directory, "displacements_plotted.png")
        )
        self.plot_displacements(
            title="Cumulative Displacements",
            displacements=np.cumsum(displacements),
            output_file=os.path.join(output_directory, "cumulative_displacements_plotted.png")
        )

        print(f"Merging Resampled Slices.")
        Parallel(n_jobs=job_count)(
            delayed(self.merge_slices)(
                slices_at_this_volume = sorted(glob(os.path.join(working_directory, f"resampled_slice_outputs-{'{:04d}'.format(volume_num)}-*"))),
                volume_path=os.path.join(working_directory, f"volume_outputs-{'{:04d}'.format(volume_num)}.nii"),
                working_directory=working_directory
            )
            for volume_num in range(num_volumes)
        )
        print(f"\nAll Slices Merged Into Volumes.")
        
        motion_added_nifti_path = os.path.join(output_directory, "MotionAddedTimeseries.nii")
        self.merge_volumes_to_timeseries(
            volume_paths=sorted(glob(os.path.join(working_directory, f"resampled_volume_outputs-*"))),
            output_file_path=motion_added_nifti_path
        )
        print("Done.")
        print(f"Output Motion-Added NiFTI Image Path Here: {motion_added_nifti_path}")
        

    def get_random_period_in_timeseries(self, start_range, event_length, num_slice_groups, dimension_parameters, max_attempts = 10):
    
        for _ in range(max_attempts):

            start_timepoint = random.randint(num_slice_groups * 2, start_range) # dont add motion to the first two volumes aquisitions
            end_timepoint = start_timepoint + event_length

            if all(param_value == 0 for param_value in dimension_parameters[start_timepoint:end_timepoint]):
                return np.arange(start_timepoint, end_timepoint)
        
        else:
            print(f"Could not place non-overlapping motion event")
            exit(0)


    def get_motion_value(self, dimension_name, min_degree_rotation, max_degree_rotation, min_mm_translation, max_mm_translation): 
    
        if "Rotation" in dimension_name:
            return round(np.radians(random.uniform(min_degree_rotation, max_degree_rotation)), 10)
        
        elif "Translation" in dimension_name:
            return round(random.uniform(min_mm_translation, max_mm_translation), 2)
    

    def make_motion_function(self, period, amplitude, frequency, phase_shift):

        duration = len(period)
        local_time = np.arange(duration)

        values = amplitude * np.cos(
            2 * np.pi * frequency * local_time / duration + phase_shift
        )

        return values

    def read_parameters_from_file(self, parameter_file):
        
        cumulative_parameters = {
            "X-Rotation": [],
            "Y-Rotation": [],
            "Z-Rotation": [],
            "X-Translation": [],
            "Y-Translation": [],
            "Z-Translation": [],
        }
        
        with open(parameter_file, mode='r') as file:
            
            for i, line in enumerate(file):
                
                aqusition_params = [
                    float(val.strip()) 
                    for val in line.split(' ') 
                    if val.strip()
                ]

                cumulative_parameters['X-Rotation'].append(aqusition_params[0])
                cumulative_parameters['Y-Rotation'].append(aqusition_params[1])
                cumulative_parameters['Z-Rotation'].append(aqusition_params[2])
                cumulative_parameters['X-Translation'].append(aqusition_params[3])
                cumulative_parameters['Y-Translation'].append(aqusition_params[4])
                cumulative_parameters['Z-Translation'].append(aqusition_params[5])

        
        parameters = {}
        for dim, vals in cumulative_parameters.items():
            parameters[dim] = np.diff(vals)
        
        return parameters, cumulative_parameters


    def plot_motion_event(self, title, dimension_name, period, motion_function, output_file):
        
        plt.figure(figsize=(14, 8.5))
        plt.title(title)
        plt.plot(
            period,
            motion_function
        )
        plt.grid()
        plt.xlabel("Aquisition")
        plt.ylabel('Radians' if 'Rotation' in dimension_name else 'Milimeters')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

        print(f"Motion Event Plot at: {output_file}") 


    def plot_motion_parameters(self, title, parameters, output_file):

        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 8.5))
        fig.suptitle(title, fontweight="bold")

        for i, (dimension_name, parameter_list) in enumerate(parameters.items()):
            if 'rotation' in dimension_name.lower():
                parameter_list = [np.rad2deg(val) for val in parameter_list]
            
            plot = None
            if i < 3:
                plot = axes[i, 0]
                plot.set_ylabel("Degrees")
            else:
                plot = axes[i - 3, 1]
                plot.set_ylabel("Millimeters")

            plot.set_xlabel("Aquisition")
            plot.set_title(dimension_name)
            plot.grid(True)

            plot.plot(
                range(1, len(parameter_list) + 1),
                parameter_list
            )

        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Plotted {title} at: {output_file}")
        plt.close()
    

    def write_parameters_to_textfile(self, parameters, num_aquisitions, output_file_path):
        
        with open(output_file_path, mode='w') as f:

            for i in range(num_aquisitions):
                f.write(
                    ' '.join(str(val) for val in [
                        parameters["X-Rotation"][i],
                        parameters["Y-Rotation"][i],
                        parameters["Z-Rotation"][i],
                        parameters["X-Translation"][i],
                        parameters["Y-Translation"][i],
                        parameters["Z-Translation"][i],
                    ]) + '\n'
                )
            print(f"Parameter Textfile at: {output_file_path}")
    

    def write_displacement_to_textfile(self, displacements, output_file_path):
        
        with open(output_file_path, mode='w') as f:

            for displacement in displacements:
                f.write(str(displacement) + '\n')
            
            print(f"Displacement Textfile at: {output_file_path}")
        

    def plot_displacements(self, title, displacements, output_file):
        
        plt.figure(figsize=(14, 8.5))
        plt.title(title, fontweight="bold")
        plt.plot(
            range(1, len(displacements) + 1),
            displacements
        )
        plt.xlabel("Aquisition Number")
        plt.ylabel("Displacement (mm)")
        plt.grid()
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Plotted {title} at: {output_file}")
        plt.close()

        
    def cut_timeseries_into_slices(self, nifti_image_path, num_volumes, num_slices, working_directory, job_count):
        
        print(f"Extracting All Volumes With {job_count} Workers.")
        Parallel(n_jobs=job_count)(
            delayed(self.extract_volumes)(
                nifti_image_path=nifti_image_path,
                volume_num=volume_num,
                num_volumes=num_volumes,
                working_directory=working_directory
            )
            for volume_num in range(num_volumes)
        )
        print("\nAll Volumes Extracted.")

        print(f"Extracting All Slices With {job_count} Workers.")
        Parallel(n_jobs=job_count)(
            delayed(self.extract_slices)(
                volume_image_path=volume_path,
                volume_num=volume_num,
                num_slices=num_slices,
                working_directory=working_directory
            )
            for volume_num, volume_path in enumerate(sorted(glob(os.path.join(working_directory, "volume_outputs-*.nii"))))
        ) 
        print("\nAll Slices Extracted.")


    def extract_volumes(self, nifti_image_path, volume_num, num_volumes, working_directory, output_file_prefix = "volume_outputs"):

        print(f"Extracting Volume {volume_num + 1} of {num_volumes} from the Timeseries.", end='\r')

        loaded_nifti_image = sitk.ReadImage(nifti_image_path)

        volume_size = list(loaded_nifti_image.GetSize())[:3] + [0]    

        output_file = os.path.join(working_directory, f"{output_file_prefix}-{'{:04d}'.format(volume_num)}.nii")

        extract = sitk.ExtractImageFilter()
        extract.SetSize(volume_size)
        extract.SetIndex((0,0,0,volume_num))
        volume_image = extract.Execute(loaded_nifti_image)

        writer = sitk.ImageFileWriter()
        writer.SetFileName(output_file)
        writer.Execute(volume_image)
    
    
    def extract_slices(self, volume_image_path, volume_num, num_slices, working_directory, output_file_prefix = "slice_outputs"):
        
        print(f"Extracting Slices From {os.path.basename(volume_image_path)}", end='\r')

        volume_image = sitk.ReadImage(volume_image_path)

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


    def make_identity_transform(self, reference_volume_path, working_directory):
        
        identity_transform_path = os.path.join(working_directory, "identity-centered.tfm")

        reference_image = sitk.ReadImage(reference_volume_path)
        reference_size = reference_image.GetSize()

        image_center = reference_image.TransformContinuousIndexToPhysicalPoint([
            (index - 1) / 2.0
            for index in reference_size
        ])

        transform = sitk.AffineTransform(3)
        transform.SetIdentity()
        transform.SetCenter(image_center)

        # Allegedly, sitk.WriteTransform() has a long-standing bug
        # when writing an AffineTransform that has a non-zero center,
        # especially to a .tfm text file
        # When I run using sitk.WriteTransform(), I get a Segmentation fault
        # To avoid this, I chose to write to the file directly:
        with open(identity_transform_path, mode='w') as file:
            file.write("#Insight Transform File V1.0\n")
            file.write("#Transform 0\n")
            file.write("Transform: AffineTransform_double_3_3\n")
            file.write("Parameters: " + ' '.join([str(float(param)) for param in transform.GetParameters()]) + '\n')
            file.write("FixedParameters: " + ' '.join([str(param) for param in transform.GetFixedParameters()]))
            file.write('\n')
        
        return identity_transform_path
            

    def get_fixed_parameters_from_tfm_file(self, transform_path):
        with open(transform_path, mode='r') as file:
            for line in file:
                if 'FixedParameters' in line:
                    return [float(fparam) for fparam in line.split()[1:]]
    
    def get_parameters_from_tfm_file(self, transform_path):
        with open(transform_path, mode='r') as file:
            for line in file:
                if 'Parameters' in line and not 'FixedParameters' in line:
                    return [float(param) for param in line.split()[1:]]


    def create_transform(self, parameters, fixed_parameters, output_path):
        
        transform = sitk.Euler3DTransform()
        transform.SetRotation(
            parameters[0],
            parameters[1],
            parameters[2]
        )
        transform.SetTranslation(
            [parameters[3],
            parameters[4],
            parameters[5]]
        )
        transform.SetCenter(fixed_parameters)

        with open(output_path, mode='w') as file:
            file.write("#Insight Transform File V1.0\n")
            file.write("#Transform 0\n")
            file.write("Transform: Euler3DTransform_double_3_3\n")
            file.write("Parameters: " + ' '.join([str(float(param)) for param in transform.GetParameters()]) + '\n')
            file.write("FixedParameters: " + ' '.join([str(param) for param in transform.GetFixedParameters()]))
            file.write('\n')

        return transform


    def get_slice_timing(self, json_path):

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

         
    def apply_to_timeseries(self, slice_timing, volume_num, num_slice_groups, 
                        num_aquisitions, cumulative_parameters, fixed_parameters,
                        transform_directory, working_directory, reference_volume_path):
        
        for slice_group_num, (_, slice_index_list) in enumerate(slice_timing.items()):
                
            aquisition_num = (volume_num * num_slice_groups) + slice_group_num
            print(f"Applying Parameters at Aquisition {aquisition_num + 1} of {num_aquisitions}" + " " * 50, end='\r')

            loaded_transform = self.create_transform(
                parameters=[
                    cumulative_parameters["X-Rotation"][aquisition_num], 
                    cumulative_parameters["Y-Rotation"][aquisition_num],
                    cumulative_parameters["Z-Rotation"][aquisition_num],
                    cumulative_parameters["X-Translation"][aquisition_num],
                    cumulative_parameters["Y-Translation"][aquisition_num],
                    cumulative_parameters["Z-Translation"][aquisition_num]
                ],
                fixed_parameters=fixed_parameters,
                output_path=os.path.join(
                    transform_directory, 
                    f"{'{:04d}'.format(volume_num)}-{'{:04d}'.format(slice_group_num)}.tfm")
            )

            slice_paths = [
                os.path.join(working_directory, f"slice_outputs-{'{:04d}'.format(volume_num)}-{'{:03d}'.format(slice_index)}.nii")
                for slice_index in slice_index_list
            ]

            for slice_path in slice_paths:
                self.apply_transform(
                    loaded_transform=loaded_transform,
                    reference_volume_path=reference_volume_path,
                    slice_path=slice_path,
                    output_filepath=os.path.join(working_directory, f"resampled_{os.path.basename(slice_path)}")
                )


    def apply_transform(self, loaded_transform, reference_volume_path, slice_path, output_filepath):
        
        reader: sitk.ImageFileReader = sitk.ImageFileReader()
        reader.SetFileName(reference_volume_path)
        refImage: sitk.Image = reader.Execute();

        sliceReader: sitk.ImageFileReader = sitk.ImageFileReader()
        sliceReader.SetFileName(slice_path)
        sliceImage: sitk.Image = sliceReader.Execute();

        output_pixel_type: int = sliceReader.GetPixelID()
        resampleFilter: sitk.ResampleImageFilter = sitk.ResampleImageFilter()
        resampleFilter.SetInterpolator( sitk.sitkLinear )
        resampleFilter.SetTransform(loaded_transform)
        resampleFilter.SetOutputPixelType( output_pixel_type )
        resampleFilter.SetDefaultPixelValue( 0.0 )
        resampleFilter.SetReferenceImage( refImage )

        newImage: sitk.Image = resampleFilter.Execute( sliceImage )
        writer = sitk.ImageFileWriter()
        writer.SetFileName( output_filepath )
        writer.Execute( newImage )

        return output_filepath


    def merge_slices(self, slices_at_this_volume, volume_path, working_directory):

        output_file_path = os.path.join(working_directory, f"resampled_{os.path.basename(volume_path)}.gz")

        ref_img = nib.load(volume_path)
        ref_data = nib.load(volume_path).get_fdata()

        accumulator = np.zeros_like(ref_data, dtype=np.float32)
        weight_map = np.zeros_like(ref_data, dtype=np.float32)

        for _, slice_path in enumerate(slices_at_this_volume):
            print(f"Adding data from {os.path.basename(slice_path)} to {os.path.basename(output_file_path)}", end='\r')

            slice_data = nib.load(slice_path).get_fdata()
            slice_data[slice_data == 0] = np.nan

            mask = ~np.isnan(slice_data)

            accumulator[mask] += slice_data[mask]
            weight_map[mask] += 1.0

        valid_voxels = weight_map > 0

        reconstructed_data = np.full_like(
            a=ref_data,
            fill_value=np.nan,
            dtype=np.float32
        )
        reconstructed_data[valid_voxels] = (accumulator[valid_voxels] / weight_map[valid_voxels])

        interp_reconstructed_data = self.nn_interp(reconstructed_data, brain_mask=(ref_data != 0))

        reconstructed_img = nib.Nifti1Image(interp_reconstructed_data, ref_img.affine, ref_img.header)

        nib.save(reconstructed_img, output_file_path)
    

    def nn_interp(self, volume_data, brain_mask):

        volume_data_copy = volume_data.copy()
        nan_mask = np.isnan(volume_data_copy)
        
        # Only fill NaNs inside brain mask
        fill_mask = nan_mask & brain_mask
        
        if not fill_mask.any():
            return volume_data_copy

        # coordinates of valid data
        coords =  np.array(np.nonzero(~nan_mask)).T.astype(np.float64)   # shape (N, 3)
        """if coords.shape[0] == 0:
            return np.zeros_like(volume_data_copy, dtype=np.float64)"""
        
        values = volume_data_copy[~nan_mask].astype(np.float64)
        nan_coords = np.array(np.nonzero(fill_mask)).T.astype(np.float64)

        
        """
        # Spline Interpolation 
        interpolator = RBFInterpolator(
            coords,
            values,
            kernel="thin_plate_spline", 
            degree=1,                   
            neighbors=32                 
        )"""

        # Nearest Neighbor Interpolation
        interpolator = NearestNDInterpolator(coords, values)

        # Coordinates to fill
        volume_data_copy[fill_mask] = interpolator(nan_coords)

        return volume_data_copy


    def merge_volumes_to_timeseries(self, volume_paths, output_file_path):
        
        print(f"Joining {len(volume_paths)} Volumes.")

        volumes = [sitk.ReadImage(volume_path) for volume_path in volume_paths]

        join_array = sitk.JoinSeries(volumes, 0.0, 1.0)

        writer = sitk.ImageFileWriter()
        writer.SetFileName(output_file_path)
        writer.Execute(join_array)
    

    def compute_displacement(self, transform1_path, transform2_path, radius=50):
        
        print(f"Computing Displacement Between Transform {os.path.basename(transform1_path)} and Transform {os.path.basename(transform2_path)}" + " " * 20, end='\r')
        
        transform1 = sitk.Euler3DTransform()
        parameters1 = self.get_parameters_from_tfm_file(transform1_path)
        transform1.SetRotation(
            parameters1[0],
            parameters1[1],
            parameters1[2]
        )
        transform1.SetTranslation([
            parameters1[3],
            parameters1[4],
            parameters1[5]
        ])
        transform1.SetCenter(self.get_fixed_parameters_from_tfm_file(transform1_path))

        transform2 = sitk.Euler3DTransform()
        parameters2 = self.get_parameters_from_tfm_file(transform2_path)
        transform2.SetRotation(
            parameters2[0],
            parameters2[1],
            parameters2[2]
        )
        transform2.SetTranslation([
            parameters2[3],
            parameters2[4],
            parameters2[5]
        ])
        transform2.SetCenter(self.get_fixed_parameters_from_tfm_file(transform2_path))

        combined_transform = sitk.CompositeTransform([
            transform2, 
            transform1.GetInverse()
        ])

        # Sample a point at radius distance from center
        center = np.array(transform1.GetCenter())

        # Point on sphere (x-direction)
        p = center + np.array([radius, 0, 0])

        # Apply relative transform
        p_trans = np.array(combined_transform.TransformPoint(tuple(p)))

        # Displacement magnitude
        displacement = np.linalg.norm(p_trans - p)

        return displacement


if __name__ == '__main__':
    
    import argparse

    default_working_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working")
    default_output_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

    parser = argparse.ArgumentParser(
        description='Script for create fake intra-volume motion in a NiFTi timeseries. NOTE: Script will output transforms as SITK Euler3DTransforms.'
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
        default=default_working_directory,
        help=f"Default: {default_working_directory}"
    )
    parser.add_argument(
        "--output_directory_path",
        required=False,
        default=default_output_directory,
        help=f"Default: {default_output_directory}"
    )
    parser.add_argument(
        "--input_parameter_file",
        required=False,
        help='A text file containing the desired 6 rigid body movements at each slice group. Rotation parameters must be in RADIANS.'
    )
    args = parser.parse_args()

    MakeFakeMotion(
        nifti_image_path=os.path.abspath(args.nifti_image_path),
        json_file_path=os.path.abspath(args.json_file_path),
        working_directory=os.path.abspath(args.working_directory_path),
        output_directory=os.path.abspath(args.output_directory_path),
        input_parameter_file=os.path.abspath(args.input_parameter_file) if args.input_parameter_file else None
    )
