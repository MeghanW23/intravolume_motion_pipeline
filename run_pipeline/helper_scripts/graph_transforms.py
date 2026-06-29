import os 
import json
import math
import numpy as np 
from glob import glob
import SimpleITK as sitk
from matplotlib import patches
from matplotlib import pyplot as plt 

class GraphTransformDirectory:
    def __init__(self, transform_directory, json_path, output_directory, input_rotation_unit, 
                 plot_tile = "Motion Characterization Plots", threshold_as_percent_of_voxel = None,
                 transform_suffix = ".tfm", framewise_displacements = False, do_not_plot = False):
        
        os.makedirs(output_directory, exist_ok=True)
        
        # Get motion values from transform directory 
        transforms = self.find_transform_paths(transform_directory, transform_suffix)
        print(f"{len(transforms)} Total Transforms Found")
        
        self.parameters = self.extract_parameters(transforms, input_rotation_unit)
        print(f"{len(self.parameters)} Total 6 Dimension Parameter Lists Found")
        
        self.displacements = [0] + [self.compute_displacement(transform1=transforms[i - 1], transform2=transforms[i]) 
            for i in range(len(transforms)) 
            if i != 0
        ]
        print(f"{len(self.displacements)} Total Displacements (Min: {round(min(self.displacements), 4)}mm, Max: {round(max(self.displacements), 3)}mm)")
        
        self.num_slice_groups = 1
        if not framewise_displacements:
            self.num_slice_groups = self.get_num_slice_groups(json_path)
        num_volumes = len(transforms) / self.num_slice_groups 
        if not num_volumes.is_integer():
            print(f"\nWARNING: Number of volumes is not an integer (num_transforms / num_slice_groups  = {num_volumes}). Casting to int.\n")
        num_volumes = int(num_volumes)

        # Write extracted info to files
        self.write_parameters_to_text_file(
            self.parameters,
            output_text_file_path=os.path.join(output_directory, "parameters.txt"),
            num_slice_groups=self.num_slice_groups
        )
        self.write_displacements_to_text_file(
            self.displacements,
            output_text_file_path=os.path.join(output_directory, "displacements.txt"),
            num_slice_groups=self.num_slice_groups
        )

        # Get motion threshold from JSON file's 'SpacingBetweenSlices' key, get motion flags based on threshold
        self.mm_displacement_threshold = None
        self.motion_flagged_volumes = []
        if threshold_as_percent_of_voxel:
            
            self.mm_displacement_threshold = self.get_threshold_in_mm(json_path, threshold_as_percent_of_voxel)
            
            self.motion_flagged_volumes = self.get_motion_flagged_volumes(
                num_volumes=num_volumes,
                num_slice_groups=self.num_slice_groups,
                displacement_values=self.displacements,
                mm_displacement_threshold=self.mm_displacement_threshold
            )
            print(f"Motion Flagged Volumes:\n{self.motion_flagged_volumes}")

            self.write_flags_to_text_file(
                self.motion_flagged_volumes,
                output_text_file_path=os.path.join(output_directory, f"motion_flags_{threshold_as_percent_of_voxel}-percent-of-voxelsize-threshold.txt")
            )        
        # Get xaxis labels/ticks based on JSON file's 'SliceTiming' key
        self.xaxis_ticks, self.xaxis_tick_labels = self.make_xaxis_ticks(self.num_slice_groups, num_volumes)

        # Set up plot 
        if do_not_plot:
            # if plotting multiple graphs, we can end here 
            return 
        
        fig = plt.figure(figsize=(14, 8.5))
        fig.suptitle(plot_tile, fontweight='bold')

        # Create the subplots (one for each of the parameters, one for the displacements)
        gs = fig.add_gridspec(
            4, 2,
            height_ratios=[1, 1, 1, 2]  # displacement row is 2x taller
        )

        parameter_axes = [
            fig.add_subplot(gs[0,0]),
            fig.add_subplot(gs[1,0]),
            fig.add_subplot(gs[2,0]),
            fig.add_subplot(gs[0,1]),
            fig.add_subplot(gs[1,1]),
            fig.add_subplot(gs[2,1])
        ]
        displacement_axes = fig.add_subplot(gs[3,:])

        # Add data to the subplots 
        self.plot_parameters(
            parameter_axes,
            self.parameters,
            self.xaxis_ticks,
            self.xaxis_tick_labels
        )

        self.plot_displacements(
            displacement_axes, 
            self.displacements,
            self.xaxis_ticks,
            self.xaxis_tick_labels,
            threshold_as_percent_of_voxel,
            self.mm_displacement_threshold,
            self.motion_flagged_volumes,
            self.num_slice_groups
        )
        
        output_plot_path = os.path.join(output_directory, "plot.png") if not self.mm_displacement_threshold else os.path.join(output_directory, f"plot_{threshold_as_percent_of_voxel}-percent-of-voxelsize-threshold.png")

        # Save and close 
        plt.tight_layout()
        plt.savefig(output_plot_path)
        plt.close()
        
        
    def find_transform_paths(self, transform_directory, transform_suffix):

        transforms = [
            sitk.ReadTransform(transform_path) 
            for transform_path in sorted(glob(os.path.join(transform_directory, f"*{transform_suffix}")))
            if not "identity" in os.path.basename(transform_path)
        ]
        if not transforms:
            print(f"\nERROR: No Transforms Found Matching: {os.path.join(transform_directory, f'*{transform_suffix}')}")
            exit(0)
        
        return transforms


    def extract_parameters(self, transforms, input_rotation_unit):
        parameters = [list(transform.GetParameters()) for transform in transforms]
        if input_rotation_unit == 'versor':
            return [
                self.versor_to_degrees(rx, ry, rz) + [tx, ty, tz]
                for rx, ry, rz, tx, ty, tz in parameters
            ]

        elif input_rotation_unit == 'radians':
            return [
                self.radians_to_degrees(rx, ry, rz) + [tx, ty, tz]
                for rx, ry, rz, tx, ty, tz in parameters
            ]
        
        elif input_rotation_unit == 'degrees':
            return parameters

        else:
            print(f"\nERROR: input_rotation_unit must be one of the following options: 'versor', 'radians', 'degrees'.")
            exit(0)


    def versor_to_degrees(self, x, y, z):
        
        # reconstruct quaternion
        w = math.sqrt(max(0.0, 1 - x*x - y*y - z*z))

        # rotation matrix
        r00 = 1 - 2*(y*y + z*z)
        r01 = 2*(x*y - z*w)

        r10 = 2*(x*y + z*w)
        r11 = 1 - 2*(x*x + z*z)

        r20 = 2*(x*z - y*w)
        r21 = 2*(y*z + x*w)
        r22 = 1 - 2*(x*x + y*y)

        # euler angles
        ry = math.asin(-r20)

        if abs(r20) != 1:
            rx = math.atan2(r21, r22)
            rz = math.atan2(r10, r00)
        else:
            rx = 0
            rz = math.atan2(-r01, r11)

        return [
            math.degrees(rx),
            math.degrees(ry),
            math.degrees(rz)
        ]


    def radians_to_degrees(self, x, y, z):
        return [
            math.degrees(x),
            math.degrees(y),
            math.degrees(z)
        ]


    def compute_displacement(self, transform1, transform2, radius = 50):

        A0 = np.asarray(transform2.GetMatrix()).reshape(3, 3)
        c0 = np.asarray(transform2.GetCenter())
        t0 = np.asarray(transform2.GetTranslation())

        A1 = np.asarray(transform1.GetInverse().GetMatrix()).reshape(3, 3)
        c1 = np.asarray(transform1.GetInverse().GetCenter())
        t1 = np.asarray(transform1.GetInverse().GetTranslation())

        combined_mat = np.dot(A0,A1)
        combined_translation = np.dot(A0, t1+c1-c0) + t0+c0-c1

        versorrigid3d = sitk.VersorRigid3DTransform()
        versorrigid3d.SetCenter(c1)
        versorrigid3d.SetTranslation(combined_translation)
        versorrigid3d.SetMatrix(combined_mat.flatten())

        euler3d = sitk.Euler3DTransform()
        euler3d.SetCenter(c1)
        euler3d.SetTranslation(combined_translation)
        euler3d.SetMatrix(combined_mat.flatten())

        parms = np.asarray(euler3d.GetParameters())
        
        return np.sqrt(
            (parms[0]*radius)**2 +
            (parms[1]*radius)**2 +
            (parms[2]*radius)**2 +
            parms[3]**2 +
            parms[4]**2 +
            parms[5]**2
        )


    def make_xaxis_ticks(self, num_slice_groups, num_volumes, tick_divisor = 30):


        xaxis_ticks = [
            num_slice_groups * volume_num 
            for volume_num in range(num_volumes) 
            if volume_num % tick_divisor == 0
        ]
        xaxis_tick_labels = [
            volume_num
            for volume_num in range(num_volumes)
            if volume_num % tick_divisor == 0
        ]

        return xaxis_ticks, xaxis_tick_labels
    
    
    def get_num_slice_groups(self, json_path):
                
        with open(json_path, mode='r') as f:
            return len(set(json.load(f)['SliceTiming']))
        

    def get_threshold_in_mm(self, json_path, threshold_as_percent_of_voxel):
        with open(json_path, mode='r') as f:
            mm_displacement_threshold = json.load(f)['SpacingBetweenSlices'] * (threshold_as_percent_of_voxel / 100)
            print(f"Threshold Displacement ({threshold_as_percent_of_voxel}% of Voxel Width): {round(mm_displacement_threshold, 4)} mm")
            return mm_displacement_threshold
 

    def plot_displacements(self, displacement_axes, displacements, xaxis_ticks, xaxis_tick_labels, threshold_as_percent_of_voxel, mm_displacement_threshold, motion_flagged_volumes, num_slice_groups):
        
        title = "Displacements"
        displacement_axes.set_xlabel("Volume Number")
        displacement_axes.set_ylabel("Milimeters")
        displacement_axes.plot(
            range(len(displacements)),
            displacements
        )
        displacement_axes.set_xticks(xaxis_ticks)
        displacement_axes.set_xticklabels(xaxis_tick_labels)

        
        if mm_displacement_threshold:

            # plot threshold as red dashed line 
            displacement_axes.axhline(y=mm_displacement_threshold, color='red', linestyle='--', label=f"Motion Threshold:\n{mm_displacement_threshold} mm / {threshold_as_percent_of_voxel} Percent of Voxel Width")
            
            # plot motion flags at each above-threshold volume 
            max_displacement_val = max(displacements)
            for i, motion_flagged_volume_num in enumerate(motion_flagged_volumes):
                if i == 0:
                    title = title + f" ({len(motion_flagged_volumes)} Flagged Volumes)"
                    rect = patches.Rectangle(
                            xy=(motion_flagged_volume_num * num_slice_groups, 0),
                            width=num_slice_groups,
                            height=max_displacement_val,
                            facecolor='gray',
                            edgecolor='black',
                            alpha=0.25,
                            label="Flagged Volume"
                        )
                else:
                    rect = patches.Rectangle(
                            xy=(motion_flagged_volume_num * num_slice_groups, 0),
                            width=num_slice_groups,
                            height=max_displacement_val,
                            facecolor='gray',
                            edgecolor='black',
                            alpha=0.25
                        )
                displacement_axes.add_patch(rect)
            
            displacement_axes.legend()
        
        displacement_axes.set_title(title)
        displacement_axes.grid()


    def plot_parameters(self, parameter_axes, parameters, xaxis_ticks, xaxis_tick_labels):
        
        dimensions = ["X Rotation", "Y Rotation", "Z Rotation", "X Translation", "Y Translation", "Z Translation"]
        for dimension_num, dimension_name in enumerate(dimensions):
            parameter_axes[dimension_num].set_title(dimension_name)
            parameter_axes[dimension_num].set_xlabel("Volume Number")
            parameter_axes[dimension_num].set_ylabel("Milimeters" if dimension_num > 2 else "Degrees")
            parameter_axes[dimension_num].plot(
                range(len(parameters)),
                [parameter_list[dimension_num] for parameter_list in parameters]
            )
            parameter_axes[dimension_num].grid()
            parameter_axes[dimension_num].set_xticks(xaxis_ticks)
            parameter_axes[dimension_num].set_xticklabels(xaxis_tick_labels)


    def get_motion_flagged_volumes(self, num_volumes, num_slice_groups, displacement_values, mm_displacement_threshold):
        
        motion_flagged_volumes = []
        for volume_num in range(num_volumes):
            displacements_at_this_volume = displacement_values[volume_num*num_slice_groups:(volume_num*num_slice_groups) + num_slice_groups]
            if any([displacement > mm_displacement_threshold for displacement in displacements_at_this_volume]):
                motion_flagged_volumes.append(volume_num)
        
        return motion_flagged_volumes 

    
    def write_flags_to_text_file(self, motion_flagged_volumes, output_text_file_path):
        with open(output_text_file_path, mode='w') as f:
            for motion_flagged_volume in motion_flagged_volumes:
                f.write(f"{motion_flagged_volume}\n") 

    def write_parameters_to_text_file(self, parameters, output_text_file_path, num_slice_groups):
        with open(output_text_file_path, mode='w') as f:
            for parameter_list in parameters:
                f.write(' '.join([str(parameter) for parameter in parameter_list]) + '\n')  

    def write_displacements_to_text_file(self, displacements, output_text_file_path, num_slice_groups):
        with open(output_text_file_path, mode='w') as f:
            for displacement in displacements:
                f.write(f"{displacement}\n") 

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Graph the Parameters and Displacements of a Directory of Transforms")
    parser.add_argument(
        "--transform_directory",
        required=True,
        help="Must be a directory of Transforms ending in '.tfm' or '.txt.'. Must be sortable via sort()."
    )
    parser.add_argument(
        "--json_path",
        required=True
    )
    parser.add_argument(
        "--output_directory",
        required=True
    )
    parser.add_argument(
        "--plot_title",
        required=False,
        default="Slice-by-Slice Motion Characterization Plots",
        help="Default Tile: 'Slice-by-Slice Motion Characterization Plots'."
    )
    parser.add_argument(
        "--input_rotation_unit",
        required=True,
        choices=['versor', 'radians', 'degrees']
    )
    parser.add_argument(
        "--transform_suffix",
        required=False,
        choices=['.tfm', '.txt'],
        default='.tfm',
        help="The file extension of all the transforms in the directory. Default: '.tfm'"
    )
    parser.add_argument(
        "--threshold",
        required=False,
        type=int,
        default=25,
        help="Threshold as a percent of the width of a single voxel. Default is 25 Percent"
    )
    parser.add_argument(
        "--framewise_displacements",
        action='store_true',
        help="Use this flag if the inputted transforms characterize framewise motion, not intravolume motion."
    )
    
    args = parser.parse_args()

    GraphTransformDirectory(
        transform_directory=os.path.abspath(args.transform_directory),
        json_path=os.path.abspath(args.json_path),
        output_directory=os.path.abspath(args.output_directory),
        plot_tile=args.plot_title,
        input_rotation_unit=args.input_rotation_unit,
        transform_suffix=args.transform_suffix,
        threshold_as_percent_of_voxel=args.threshold,
        framewise_displacements=args.framewise_displacements
    )