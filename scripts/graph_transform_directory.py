import os 
import json
import math
import numpy as np 
from glob import glob
import SimpleITK as sitk
import plotly.graph_objects as go 
from plotly.subplots import make_subplots

class GraphTransformDirectory:
    def __init__(self, 
                 transform_directory: str, 
                 json_path: str, 
                 output_file_path: str, 
                 input_rotation_unit: str, 
                 plot_tile: str = "Motion Characterization Plots", 
                 threshold_in_mm: float | None = None,
                 transform_suffix: str = ".tfm", framewise_displacements = False):

        # Get and load transforms from transform directory 
        transforms: list[sitk.Euler3DTransform | sitk.VersorRigid3DTransform] = \
            self.find_transform_paths(transform_directory, transform_suffix)
        self.num_aquisitions: int = len(transforms)
        print(f"{self.num_aquisitions} Total Transforms Found")
        

        # Get parameters from files and convert rotation unit to degrees
        self.parameters: list[list[float]] = self.extract_parameters(transforms, input_rotation_unit)
        print(f"{len(self.parameters)} Total 6 Dimension Parameter Lists Found")


        # Calculate Displacements
        self.displacements: list[float] = [0] + [
            self.compute_displacement(
                transform1=transforms[i - 1], 
                transform2=transforms[i]) 
            for i in range(self.num_aquisitions) 
            if i != 0
        ]

        # Extract number of slice groups per volume from the JSON file's 'SliceTiming' Key
        self.num_slice_groups: int = 1
        if not framewise_displacements:
            self.num_slice_groups: int = self.get_num_slice_groups(json_path)
        
        # Determine total number of volumes
        self.num_volumes: float = self.num_aquisitions / self.num_slice_groups  # type: ignore
        if not self.num_volumes.is_integer(): # type: ignore
            print(f"\nWARNING: Number of volumes is not an integer (num_transforms / num_slice_groups  = {self.num_volumes}). Casting to int.\n")
        self.num_volumes: int = int(self.num_volumes)

        # Get motion threshold from JSON file's 'SpacingBetweenSlices' key and get motion flags based on this threshold
        self.motion_flagged_volumes: list[int] = []
        if threshold_in_mm:
                        
            self.motion_flagged_volumes: list[int] = self.get_motion_flagged_volumes(
                num_volumes=self.num_volumes,
                num_slice_groups=self.num_slice_groups,
                displacement_values=self.displacements,
                mm_displacement_threshold=threshold_in_mm
            )
            print(f"Motion Flagged Volumes:\n{self.motion_flagged_volumes}")
        
        print("Initializing Plot")
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
        print("Formatting Data")
        # Format these values so they align with the num of aquisitions (to be used in the hover text)       
        formatted_volume_nums: list[int] = [
            (aquisition_num // self.num_slice_groups) + 1
            for aquisition_num in range(self.num_aquisitions)
        ]
        formatted_slice_group_nums: list[int] = [
            (aquisition_num - ((aquisition_num // self.num_slice_groups) * self.num_slice_groups)) + 1
            for aquisition_num in range(self.num_aquisitions)
        ]  
        formatted_rotations: list[str] = [
            ', '.join(str(round(rot_val, 2)) + " deg" for rot_val in parameter_list[:3])
            for parameter_list in self.parameters
        ] 
        formatted_translations : list[str] = [
            ', '.join(str(round(trans_val, 2)) + " mm" for trans_val in parameter_list[3:])
            for parameter_list in self.parameters
        ] 
        formatted_displacements: list[str] = [
            str(round(displacement_val, 2)) 
            for displacement_val in self.displacements
        ]

        formatted_motion_flag_values: list[str] = [] # wether or not a each aquisition is in a motion flagged volume or not
        for aquisition_num in range(self.num_aquisitions):
            if (aquisition_num // self.num_slice_groups) in self.motion_flagged_volumes:
                formatted_motion_flag_values.append("True")
            else:
                formatted_motion_flag_values.append("False")


        # Plot parameters (6 plots total / 3x2 plots)
        print("Plotting Parameters")
        for dimension_num in range(6):
            
            # Extract all parameters in a given dimension
            line: list[float] = [parameter_list[dimension_num] for parameter_list in self.parameters]
            x_vals: list[float] = [(i / self.num_slice_groups) + 1 for i in range(len(line))]

            # Get row and column number
            row_num: int | None = None # pyright: ignore[reportAssignmentType, reportRedeclaration]
            col_num: int | None = None # pyright: ignore[reportAssignmentType, reportRedeclaration]
            unit_name: str | None = None # pyright: ignore[reportAssignmentType, reportRedeclaration]
            if dimension_num < 3:
                row_num: int = 1
                col_num : int = dimension_num + 1
                unit_name: str = "deg"
            else:
                row_num: int = 2
                col_num: int = dimension_num - 2
                unit_name: str = "mm"
            
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
        print("Plotting Displacements")
        displacement_x_vals: list[float] = [(i / self.num_slice_groups) + 1 for i in range(len(self.displacements))]
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
        print("Plotting Motion Flags")
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
        print("Plotting Displacement Threshold")
        if threshold_in_mm:
            fig.add_hline(
                y=threshold_in_mm,
                line=dict(color="black", width=2, dash="dash"),
                row=3, col=1, # pyright: ignore[reportArgumentType]
                name=f"Threshold: {round(threshold_in_mm, 4)} mm",
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
        print("Formatting Axes")    
        fig.update_xaxes(title_text="Volume Number", title_font=dict(size=12), title_standoff=5, showticklabels=True, row=1)
        fig.update_xaxes(title_text="Volume Number", title_font=dict(size=12), title_standoff=5, showticklabels=True, row=2)
        fig.update_xaxes(title_text="Volume Number", title_font=dict(size=12), title_standoff=5, showticklabels=True, row=3)
        fig.update_xaxes(matches="x")

        # Save plot 
        print("Saving Plot")
        fig.write_html(output_file_path)
        # fig.show()  

        
    def find_transform_paths(self, 
                             transform_directory: str,
                             transform_suffix: str
                             ) -> list[sitk.VersorRigid3DTransform | sitk.Euler3DTransform]:

        transforms: list[sitk.VersorRigid3DTransform | sitk.Euler3DTransform] = [
            sitk.ReadTransform(transform_path) 
            for transform_path in sorted(glob(os.path.join(transform_directory, f"*{transform_suffix}")))
            if not "identity" in os.path.basename(transform_path)
        ]
        if not transforms:
            raise FileNotFoundError(
                f" No Transforms Found Matching: {os.path.join(transform_directory, f'*{transform_suffix}')}"
            )
        return transforms


    def extract_parameters(self, 
                           transforms: list[sitk.VersorRigid3DTransform | sitk.Euler3DTransform], 
                           input_rotation_unit: str
                           ) -> list[list[float]]:
        
        parameters: list[list[float]] = [list(transform.GetParameters()) for transform in transforms]
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
            raise ValueError(
                "input_rotation_unit must be one of the following options: 'versor', 'radians', 'degrees'."
            )


    def versor_to_degrees(self, x: float, y: float, z: float) -> list[float]:
        
        # reconstruct quaternion
        w: float = math.sqrt(max(0.0, 1 - x*x - y*y - z*z))

        # rotation matrix
        r00: float = 1 - 2*(y*y + z*z)
        r01: float = 2*(x*y - z*w)

        r10: float = 2*(x*y + z*w)
        r11: float = 1 - 2*(x*x + z*z)

        r20: float = 2*(x*z - y*w)
        r21: float = 2*(y*z + x*w)
        r22: float = 1 - 2*(x*x + y*y)

        # euler angles
        ry: float = math.asin(-r20)

        if abs(r20) != 1:
            rx: float = math.atan2(r21, r22)
            rz: float = math.atan2(r10, r00)
        else:
            rx: float = 0
            rz: float = math.atan2(-r01, r11)

        return [
            math.degrees(rx),
            math.degrees(ry),
            math.degrees(rz)
        ]


    def radians_to_degrees(self, x: float, y: float, z: float) -> list[float]:
        return [
            math.degrees(x),
            math.degrees(y),
            math.degrees(z)
        ]

    def compute_displacement(self, 
                             transform1: sitk.VersorRigid3DTransform | sitk.Euler3DTransform, 
                             transform2: sitk.VersorRigid3DTransform | sitk.Euler3DTransform, 
                             radius: float = 50) -> float:

        A0: np.ndarray = np.asarray(transform2.GetMatrix()).reshape(3, 3)
        c0: np.ndarray = np.asarray(transform2.GetCenter())
        t0: np.ndarray = np.asarray(transform2.GetTranslation())

        A1: np.ndarray = np.asarray(transform1.GetInverse().GetMatrix()).reshape(3, 3)
        c1: np.ndarray = np.asarray(transform1.GetInverse().GetCenter())
        t1: np.ndarray = np.asarray(transform1.GetInverse().GetTranslation())

        combined_mat: np.ndarray = np.dot(A0,A1)
        combined_center: np.ndarray = c1
        combined_translation: np.ndarray = np.dot(A0, t1+c1-c0) + t0+c0-c1

        euler3d: sitk.Euler3DTransform = sitk.Euler3DTransform()
        euler3d.SetCenter(combined_center)
        euler3d.SetTranslation(combined_translation)
        euler3d.SetMatrix(combined_mat.flatten())

        parms: np.ndarray = np.asarray(euler3d.GetParameters())

        return \
            abs(parms[0]*radius) + \
            abs(parms[1]*radius) + \
            abs(parms[2]*radius) + \
            abs(parms[3]) + \
            abs(parms[4]) + \
            abs(parms[5])

    def get_num_slice_groups(self, json_path: str) -> int:
                
        with open(json_path, mode='r') as f:
            return len(set(json.load(f)['SliceTiming']))
        


    def get_motion_flagged_volumes(self,
                                   num_volumes: int, 
                                   num_slice_groups: int, 
                                   displacement_values: list[float], 
                                   mm_displacement_threshold: float
                                   ) -> list[int]:
        
        motion_flagged_volumes: list[int] = []
        for volume_num in range(num_volumes):
            displacements_at_this_volume:list[float] = displacement_values[volume_num*num_slice_groups:(volume_num*num_slice_groups) + num_slice_groups]
            if any([displacement > mm_displacement_threshold for displacement in displacements_at_this_volume]):
                motion_flagged_volumes.append(volume_num)
        
        return motion_flagged_volumes 


    
 
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
        "--output_file_path",
        required=True,
        help="Should be a .html file"
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
        "--threshold_in_mm",
        required=False,
        type=float,
        default=0.6,
        help="Threshold in mm. Default: 0.6 mm"
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
        output_file_path=os.path.abspath(args.output_file_path),
        plot_tile=args.plot_title,
        input_rotation_unit=args.input_rotation_unit,
        transform_suffix=args.transform_suffix,
        threshold_in_mm=args.threshold_in_mm,
        framewise_displacements=args.framewise_displacements
    )