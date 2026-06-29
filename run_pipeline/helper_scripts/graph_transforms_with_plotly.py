import os 
import json
import math
import numpy as np 
from glob import glob
import SimpleITK as sitk
import plotly.graph_objects as go 
from plotly.subplots import make_subplots

class ProcessTransformDirectory:
    def __init__(self, transform_directory, json_path, output_directory, input_rotation_unit, 
                 plot_tile = "Motion Characterization Plots", threshold_as_percent_of_voxel = None,
                 transform_suffix = ".tfm", framewise_displacements = False):

        os.makedirs(output_directory, exist_ok=True)

        # Get and load transforms from transform directory 
        transforms = self.find_transform_paths(transform_directory, transform_suffix)
        self.num_aquisitions = len(transforms)
        print(f"{self.num_aquisitions} Total Transforms Found")
        
        # Get parameters from files and convert rotation unit to degrees
        self.parameters = self.extract_parameters(transforms, input_rotation_unit)
        print(f"{len(self.parameters)} Total 6 Dimension Parameter Lists Found")
        self.write_parameters_to_text_file(
            self.parameters,
            output_text_file_path=os.path.join(output_directory, "parameters.txt")
        )
        
        # Calculate Displacements
        self.displacements = [0] + [self.compute_displacement(transform1=transforms[i - 1], transform2=transforms[i]) 
            for i in range(self.num_aquisitions) 
            if i != 0
        ]
        self.write_displacements_to_text_file(
            self.displacements,
            output_text_file_path=os.path.join(output_directory, "displacements.txt")
        )
        print(f"{len(self.displacements)} Total Displacements (Min: {round(min(self.displacements), 4)}mm, Max: {round(max(self.displacements), 3)}mm)")
        
        # Extract number of slice groups per volume from the JSON file's 'SliceTiming' Key
        self.num_slice_groups = 1
        if not framewise_displacements:
            self.num_slice_groups = self.get_num_slice_groups(json_path)
        
        # Determine total number of volumes
        self.num_volumes = self.num_aquisitions / self.num_slice_groups 
        if not self.num_volumes.is_integer():
            print(f"\nWARNING: Number of volumes is not an integer (num_transforms / num_slice_groups  = {self.num_volumes}). Casting to int.\n")
        self.num_volumes = int(self.num_volumes)

        # Get motion threshold from JSON file's 'SpacingBetweenSlices' key and get motion flags based on this threshold
        self.mm_displacement_threshold = None
        self.motion_flagged_volumes = []
        if threshold_as_percent_of_voxel:
            
            self.mm_displacement_threshold = self.get_threshold_in_mm(json_path, threshold_as_percent_of_voxel)
            
            self.motion_flagged_volumes = self.get_motion_flagged_volumes(
                num_volumes=self.num_volumes,
                num_slice_groups=self.num_slice_groups,
                displacement_values=self.displacements,
                mm_displacement_threshold=self.mm_displacement_threshold
            )

            self.write_flags_to_text_file(
                self.motion_flagged_volumes,
                output_text_file_path=os.path.join(output_directory, f"motion-flagged-volumes_thr-{threshold_as_percent_of_voxel}-percent.txt")
            )
            print(f"Motion Flagged Volumes:\n{self.motion_flagged_volumes}")

        # Set up plot layout
        fig = make_subplots(rows=3, cols=3,
                            specs=[
                                [{}, {}, {}], 
                                [{}, {}, {}], 
                                [{"colspan": 3}, None, None]
                            ],
                            subplot_titles=[
                                "<b>X Rotation", "<b>Y Rotation", "<b>Z Rotation",
                                "<b>X Translation", "<b>Y Translation</b>", "<b>Z Translation</b>",
                                "<b>Displacements</b>",
                            ],
                            shared_yaxes=True,
                            shared_xaxes=True,
                            vertical_spacing=0.1,
                            horizontal_spacing=0.025,  # default is 0.2,
                            row_heights=[0.3, 0.3, 0.4]
                            )
        fig.update_layout(
            margin=dict(l=15, r=15, t=60, b=20),
            hoverlabel=dict(bgcolor="white"),
            title_text="<b>" + plot_tile + "</b>",
            title_x=0.5,
            legend=dict(
                x=0.005,  # 0=left, 1=right
                y=0.235,  # 0=bottom, 1=top
                bgcolor ='rgba(255,255,255,0.8)', # Background color of the legend box
                bordercolor="black",               # Border color of the legend box
                borderwidth=1                      # Border thickness
            ),
            legend_tracegroupgap=2

        )           

        # Format these values so they align with the num of aquisitions (to be used in the hover text)       
        formatted_volume_nums = [
            (aquisition_num // self.num_slice_groups) + 1
            for aquisition_num in range(self.num_aquisitions)
        ]
        formatted_slice_group_nums = [
            (aquisition_num - ((aquisition_num // self.num_slice_groups) * self.num_slice_groups)) + 1
            for aquisition_num in range(self.num_aquisitions)
        ]  
        formatted_rotations = [
            ', '.join(str(round(rot_val, 2)) + " deg" for rot_val in parameter_list[:3])
            for parameter_list in self.parameters
        ] 
        formatted_translations = [
            ', '.join(str(round(trans_val, 2)) + " mm" for trans_val in parameter_list[3:])
            for parameter_list in self.parameters
        ] 
        formatted_displacements = [
            str(round(displacement_val, 2)) 
            for displacement_val in self.displacements
        ]

        formatted_motion_flag_values = [] # wether or not a each aquisition is in a motion flagged volume or not
        for aquisition_num in range(self.num_aquisitions):
            if (aquisition_num // self.num_slice_groups) in self.motion_flagged_volumes:
                formatted_motion_flag_values.append("True")
            else:
                formatted_motion_flag_values.append("False")


        # Plot parameters (6 plots total / 3x2 plots)
        for dimension_num in range(6):
            
            # Extract all parameters in a given dimension
            line = [parameter_list[dimension_num] for parameter_list in self.parameters]
            x_vals = [(i / self.num_slice_groups) + 1 for i in range(len(line))]

            # Get row and column number
            row_num = None
            col_num = None
            unit_name = None
            if dimension_num < 3:
                row_num = 1
                col_num = dimension_num + 1
                unit_name = "deg"
            else:
                row_num = 2
                col_num = dimension_num - 2
                unit_name = "mm"
            
            # Plot parameters
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=line,
                    line=dict(color="blue"),
                    mode="lines+markers",
                    marker=dict(size=3, color="darkblue"),
                    customdata=np.column_stack([
                        formatted_volume_nums, 
                        formatted_slice_group_nums, 
                        formatted_rotations, 
                        formatted_translations, 
                        formatted_displacements,
                        formatted_motion_flag_values]),
                    hovertemplate=(
                        "<b>Volume:</b> %{customdata[0]} of " + f"{self.num_volumes} total volumes<br>" +
                        "<b>Slice Group:</b> %{customdata[1]} of " + f"{self.num_slice_groups} groups per volume<br>" +
                        "<b>Parameter</b>: %{y} " + unit_name + "<br>" +
                        "<b>Rotation (X, Y, Z):</b> %{customdata[2]}<br>" +
                        "<b>Translation (X, Y, Z):</b> %{customdata[3]}<br>" +
                        "<b>Total Displacement:</b> %{customdata[4]} mm<br>" +
                        "<b>In a Motion-Flagged Volume:</b> %{customdata[5]}<br>" +
                        "<extra></extra>"
                    ),
                    showlegend=False
                ),
                row=row_num,
                col=col_num
            )

        # Plot displacements
        displacement_x_vals = [(i / self.num_slice_groups) + 1 for i in range(len(self.displacements))]
        fig.add_trace(
            go.Scatter(
                x=displacement_x_vals,
                y=self.displacements,
                line=dict(color="blue"),
                mode="lines+markers",
                marker=dict(size=3, color="darkblue"),
                name="Displacements",
                customdata=np.column_stack([
                    formatted_volume_nums, 
                    formatted_slice_group_nums, 
                    formatted_rotations, 
                    formatted_translations,
                    formatted_motion_flag_values]),
                hovertemplate=(
                        "<b>Volume:</b> %{customdata[0]} of " + f"{self.num_volumes} total volumes<br>" +
                        "<b>Slice Group:</b> %{customdata[1]} of " + f"{self.num_slice_groups} groups per volume<br>" +
                        "<b>Displacement:</b> %{y} mm <br>" +
                        "<b>Rotation (X, Y, Z): </b> %{customdata[2]}<br>" +
                        "<b>Translation (X, Y, Z):</b> %{customdata[3]}<br>" +
                        "<b>In a Motion-Flagged Volume:</b> %{customdata[4]}<br>" +
                        "<extra></extra>"
                    ),
                showlegend=False
            ),
            row=3,
            col=1
            
        )  

        # Plot motion flags
        for i, flagged_volume in enumerate(self.motion_flagged_volumes):
            fig.add_trace(
                go.Scatter(
                    x=[
                        flagged_volume + 1, 
                        flagged_volume + 1, 
                        flagged_volume + 2, 
                        flagged_volume + 2
                    ],
                    y=[
                        min(self.displacements) - 0.1, 
                        max(self.displacements) * 1.1, 
                        max(self.displacements) * 1.1, 
                        min(self.displacements) - 0.1
                    ], 
                    fill="toself",
                    fillcolor="rgba(0,0,0,0.25)",
                    line=dict(width=0),
                    legendgroup="motion_flags",
                    mode="none",
                    name=f"Motion Flags: {len(self.motion_flagged_volumes)} of {self.num_volumes} Volumes Flagged",
                    showlegend=True if i == 0 else False,
                    hoverinfo="skip",
                ),
            row=3, col=1
        )

        # Plot displacement threshold line
        if self.mm_displacement_threshold:
            fig.add_hline(
                y=self.mm_displacement_threshold,
                line=dict(color="black", width=2, dash="dash"),
                row=3, col=1,
                name=f"Threshold: {threshold_as_percent_of_voxel}% of a Voxel's Width or {self.mm_displacement_threshold} mm",
                showlegend=True
            )
        
        # Format y axis
        fig.update_yaxes(title_text="Degrees", title_font=dict(size=12), title_standoff=5, showticklabels=True, row=1, col=1)
        fig.update_yaxes(title_text="", title_font=dict(size=12), title_standoff=5, showticklabels=True, row=1, col=2)
        fig.update_yaxes(title_text="", title_font=dict(size=12), title_standoff=5, showticklabels=True, row=1, col=3)

        fig.update_yaxes(title_text="Millimeters", title_font=dict(size=12), title_standoff=5, showticklabels=True, row=2, col=1)
        fig.update_yaxes(title_text="", title_font=dict(size=12), title_standoff=5, showticklabels=True, row=2, col=2)
        fig.update_yaxes(title_text="", title_font=dict(size=12), title_standoff=5, showticklabels=True, row=2, col=3)
        
        fig.update_yaxes(
            title_text="Millimeters", 
            title_font=dict(size=12), 
            title_standoff=5, 
            showticklabels=True, 
            range=[
                min(self.displacements) - 0.1, # min
                max(self.displacements) + (max(self.displacements) * 0.1) # max
            ],
            row=3
        )   
    
        # Format x axis        
        fig.update_xaxes(title_text="Volume Number", title_font=dict(size=12), title_standoff=5, showticklabels=True, row=1)
        fig.update_xaxes(title_text="Volume Number", title_font=dict(size=12), title_standoff=5, showticklabels=True, row=2)
        fig.update_xaxes(title_text="Volume Number", title_font=dict(size=12), title_standoff=5, showticklabels=True, row=3)
        fig.update_xaxes(matches="x")

        # Save plot 
        plot_path = os.path.join(output_directory, f"plot_thr-{threshold_as_percent_of_voxel}-percent.html")
        fig.write_html(plot_path)
        fig.show()
        print(f"Plot at: {plot_path}")        

        
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


    def write_parameters_to_text_file(self, parameters, output_text_file_path):
        with open(output_text_file_path, mode='w') as f:
            for parameter_list in parameters:
                f.write(' '.join([str(parameter) for parameter in parameter_list]) + '\n')  


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

       
    def write_displacements_to_text_file(self, displacements, output_text_file_path):
        with open(output_text_file_path, mode='w') as f:
            for displacement in displacements:
                f.write(str(displacement)) 


    def get_num_slice_groups(self, json_path):
                
        with open(json_path, mode='r') as f:
            return len(set(json.load(f)['SliceTiming']))
        

    def get_threshold_in_mm(self, json_path, threshold_as_percent_of_voxel):
        with open(json_path, mode='r') as f:
            mm_displacement_threshold = json.load(f)['SpacingBetweenSlices'] * (threshold_as_percent_of_voxel / 100)
            print(f"Threshold Displacement ({threshold_as_percent_of_voxel}% of Voxel Width): {round(mm_displacement_threshold, 4)} mm")
            return mm_displacement_threshold


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
    
    
 
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract and Graph the Parameters and Displacements of a Directory of Transforms")
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
        "--output_directory_path",
        required=True,
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
        help="Threshold as a percent of the width of a single voxel. Default: 25 Percent"
    )
    parser.add_argument(
        "--framewise_displacements",
        action='store_true',
        help="Use this flag if the inputted transforms characterize framewise motion, not intravolume motion."
    )
    
    args = parser.parse_args()

    ProcessTransformDirectory(
        transform_directory=os.path.abspath(args.transform_directory),
        json_path=os.path.abspath(args.json_path),
        output_directory=os.path.abspath(args.output_directory_path),
        plot_tile=args.plot_title,
        input_rotation_unit=args.input_rotation_unit,
        transform_suffix=args.transform_suffix,
        threshold_as_percent_of_voxel=args.threshold,
        framewise_displacements=args.framewise_displacements
    )
