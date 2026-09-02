import os 
import sys
import math
import json
from glob import glob 
from typing import Any
from collections import OrderedDict
from plotly import graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from calculate_displacements import CalculateDisplacements
from get_slice_timing import GetSliceTiming

class CompareFDvsSD:
    """
    Plot Frame-wise Displacement (calculated by fMRIPrep) vs Slice-wise Displacement (calculated by our software)
    """
    def __init__(self, 
                 intravolume_transform_directory: str, 
                 framewise_transform_directory: str,
                 json_file_path: str,
                 mm_motion_threshold: float,
                 also_save_png: bool,
                 output_directory: str = 'outputs',  
                 head_radius: int = 50,
                 input_rotation_unit: str = "versor",
                 plot_tile: str = "Framewise vs Intravolume Motion Characterization") -> None:

        os.makedirs(output_directory, exist_ok=True)

        # Load parameters in degrees from input transforms 
        print(f"Loading Parameters from Transform Directory: {intravolume_transform_directory}")
        sd_parameters: dict[str, list[float]] = self.load_parameters_from_transform_directory(
            intravolume_transform_directory,
            input_rotation_unit=input_rotation_unit
        )
        print(f"Loading Parameters from Transform Directory: {framewise_transform_directory}")
        fd_parameters: dict[str, list[float]] = self.load_parameters_from_transform_directory(
            framewise_transform_directory,
            input_rotation_unit=input_rotation_unit)

        # Calculate displacements from input transforms 
        print(f"Loading Displacements from Transform Directory: {intravolume_transform_directory}")    
        sd_displacements: list[float]= CalculateDisplacements(
            transform_directory=intravolume_transform_directory,
            head_radius=head_radius,
            file_extension='.tfm',
            verbose=False
        ).return_displacements()
        print(f"{len(sd_displacements)} Values Extracted")
        print(f"Loading Displacements from Transform Directory: {framewise_transform_directory}")    
        fd_displacements: list[float] = CalculateDisplacements(
            transform_directory=framewise_transform_directory,
            head_radius=head_radius,
            file_extension='.tfm',
            verbose=False
        ).return_displacements()
        print(f"{len(fd_displacements)} Values Extracted")

        # Get the timing of each aquisition
        num_slice_groups: int = int((len(sd_displacements) + 1) / (len(fd_displacements) + 1))
        print(f"Number of Slice Groups: {num_slice_groups}")

        slice_timing: OrderedDict[float, list[int]] = GetSliceTiming(json_file_path).return_slice_timing()
        slice_timings: list[float] = sorted(list(slice_timing.keys()))

        t_r: float = 1
        with open(json_file_path, mode='r') as file:
            json_data: dict[str, Any] = json.load(file)
            if not 'RepetitionTime' in json_data:
                raise KeyError(f"Key: 'RepetitionTime' not in JSON file: {json_file_path}")
            t_r: float = json_data['RepetitionTime']
        print(f"Repetition Time: {t_r}")

        slice_level_x_axis: list[float] = []
        for volume_num in range(len(fd_displacements)):
            for slice_time in slice_timings:
                slice_level_x_axis.append((volume_num * t_r) + slice_time)

        
        # Plot data
        fig = make_subplots(
            rows=3, cols=3,
            specs=[[{}, {}, {}], [{}, {}, {}], [{"colspan": 3}, None, None]],
            subplot_titles=[
                "<b>X Rotation", "<b>Y Rotation", "<b>Z Rotation",
                "<b>X Translation", "<b>Y Translation</b>", "<b>Z Translation</b>",
                "<b>Displacements</b>",
            ],
            row_heights=[0.3, 0.3, 0.4],
            vertical_spacing=0.1,
            horizontal_spacing=0.025,  # default is 0.2,
        )
        fig.update_layout(
            title_text="<b>" + plot_tile + "</b>",
            title_x=0.5,
            hovermode='x unified'
        )

        # Plot each dimension
        for dimension_num, (dimension_name, parameter_values) in enumerate(sd_parameters.items()):
            if dimension_num < 3:
                row_num: int = 1
                col_num : int = dimension_num + 1
            else:
                row_num: int = 2
                col_num: int = dimension_num - 2
                
            fig.add_trace(
                go.Scatter(
                    x=slice_level_x_axis,
                    y=parameter_values,
                    mode="lines+markers",
                    marker=dict(size=5),
                    line=dict(color="red"),
                    name="Intravolume Motion",
                    legendgroup="Intravolume Motion",
                    showlegend=False
                ),
                row=row_num,
                col=col_num
            )

        for dimension_num, (dimension_name, parameter_values) in enumerate(fd_parameters.items()):
            if dimension_num < 3:
                row_num: int = 1
                col_num : int = dimension_num + 1
            else:
                row_num: int = 2
                col_num: int = dimension_num - 2
            fig.add_trace(
                go.Scatter(
                    x=[volume_num * t_r for volume_num in range(len(fd_displacements))],
                    y=parameter_values,
                    mode="lines+markers",
                    marker=dict(size=5),
                    line=dict(color="blue"),
                    name="Framewise Motion",
                    legendgroup="Framewise Motion",
                    showlegend=False
                ),
                row=row_num,
                col=col_num
            )


        # Plot displacements
        fig.add_trace(
            go.Scatter(
                x=slice_level_x_axis,
                y=sd_displacements,
                mode="lines+markers",
                marker=dict(size=5),
                line=dict(color="red"),
                legendgroup="Intravolume Motion",
                name="Intravolume Motion",
                showlegend=True
            ),
            row=3,
            col=1
        )
        fig.add_trace(
            go.Scatter(
                x=[volume_num * t_r for volume_num in range(len(fd_displacements))],
                y=fd_displacements,
                mode="lines+markers",
                marker=dict(size=5),
                line=dict(color="blue"),
                name="Framewise Motion",
                legendgroup="Framewise Motion",
                showlegend=True
            ),
            row=3,
            col=1
        )

        # Plot threshold
        fig.add_hline(
            y=mm_motion_threshold,
            line=dict(color="black", width=2, dash="dash"),
            row=3, col=1, # pyright: ignore[reportArgumentType]
            name=f"Threshold: {round(mm_motion_threshold, 4)} mm",
            showlegend=True
        )

        # Plot motion flags 
        fd_flagged_volumes: list[int] = [
            volume_num
            for volume_num, framewise_displacement in enumerate(fd_displacements)
            if framewise_displacement > mm_motion_threshold
        ]
        print(f"Volumes with above-threshold frame-wise displacement: {fd_flagged_volumes}")
        for i, flagged_volume in enumerate(fd_flagged_volumes):
            fig.add_trace(
                go.Scatter(
                    x=[
                        flagged_volume * t_r - (t_r / 2), 
                        flagged_volume * t_r - (t_r / 2), 
                        flagged_volume * t_r - (t_r / 2) + t_r, 
                        flagged_volume * t_r - (t_r / 2) + t_r
                    ],
                    y=[
                        min(sd_displacements + fd_displacements) - 0.1, 
                        max(sd_displacements + fd_displacements) * 1.1, 
                        max(sd_displacements + fd_displacements) * 1.1, 
                        min(sd_displacements + fd_displacements) - 0.1
                    ], 
                    fill="toself",
                    fillcolor="rgba(0,0,255,0.25)",
                    line=dict(width=0),
                    legendgroup="Framewise Motion Flags",
                    mode="none",
                    name=f"Framewise Motion Flags:<br>{len(fd_flagged_volumes)} of {len(fd_displacements) + 1} Volumes Flagged",
                    showlegend=True if i == 0 else False,
                    hoverinfo="skip",
                ),
            row=3, col=1
        )

        intravolume_flagged_volumes: list[int] = self.get_intravolume_motion_flagged_volumes(
            num_volumes=len(fd_displacements) + 1, 
            num_slice_groups=num_slice_groups,
            displacement_values=sd_displacements,
            mm_displacement_threshold=mm_motion_threshold
        )
        print(f"Volumes with above-threshold intravolume displacement: {intravolume_flagged_volumes}")
        for i, flagged_volume in enumerate(intravolume_flagged_volumes):
            fig.add_trace(
                go.Scatter(
                    x=[
                        flagged_volume * t_r - (t_r / 2), 
                        flagged_volume * t_r - (t_r / 2), 
                        flagged_volume * t_r - (t_r / 2) + t_r, 
                        flagged_volume * t_r - (t_r / 2) + t_r
                    ],
                    y=[
                        min(sd_displacements + fd_displacements) - 0.1, 
                        max(sd_displacements + fd_displacements) * 1.1, 
                        max(sd_displacements + fd_displacements) * 1.1, 
                        min(sd_displacements + fd_displacements) - 0.1
                    ], 
                    fill="toself",
                    fillcolor="rgba(255,0,0,0.25)",
                    line=dict(width=0),
                    legendgroup="Intravolume Motion Flags",
                    mode="none",
                    name=f"Intravolume Motion Flags:<br>{len(intravolume_flagged_volumes)} of {len(fd_displacements) + 1} Volumes Flagged",
                    showlegend=True if i == 0 else False,
                    hoverinfo="skip",
                ),
            row=3, col=1
        )
        
            
        # Update Axes
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
            row=3
        )   
        
        fig.update_xaxes(title_text="Time (s)", title_font=dict(size=12), title_standoff=5, showticklabels=True, row=1)
        fig.update_xaxes(title_text="Time (s)", title_font=dict(size=12), title_standoff=5, showticklabels=True, row=2)
        fig.update_xaxes(title_text="Time (s)", title_font=dict(size=12), title_standoff=5, showticklabels=True, row=3)
        fig.update_xaxes(matches="x")

        # 4. Save plot
        fig.write_html(os.path.join(output_directory, "plotted-parameters.html"))
        print(f"Figure saved to: {os.path.join(output_directory, 'plotted-parameters.html')}")
        if also_save_png:
            try:
                fig.write_image(os.path.join(output_directory, "plotted-parameters.png"), width=1200, height=800)
                print(f"Figure saved to: {os.path.join(output_directory, 'plotted-parameters.png')}")
            except Exception as e:
                if "BrowserFailedError" in type(e).__name__ or "browser seemed to close" in str(e):
                    raise RuntimeError(
                        f"Chrome failed to launch for image export: {e}. "
                        f"\n\nBCH RESEARCHERS: If you are on a login node on E3, try re-running on a compute node."
                    )
                else:
                    raise

        

    def load_parameters_from_transform_directory(self, transform_directory: str, input_rotation_unit: str) -> dict[str, list[float]]:
        
        transforms: list[str] = glob(os.path.join(transform_directory, "*.tfm"))
        parameters: dict[str, list[float]] = {
            'r_x': [],
            'r_y': [],
            'r_z': [],
            't_x': [],
            't_y': [],
            't_z': []
        }
        for transform_path in transforms:
            with open(transform_path, mode='r') as file:
                for line in file: 
                    if "Parameters" in line and not "FixedParameters" in line:
                        acq_parameters: list[float] = [
                            float(param) for param in 
                            line.split(" ")[1:]
                        ]
                        r_x_deg, r_y_deg, r_z_deg = self.convert_rotations_to_degrees(
                            x=acq_parameters[0],
                            y=acq_parameters[1],
                            z=acq_parameters[2],
                            input_rotation_unit=input_rotation_unit
                        )
                        
                        parameters["r_x"].append(r_x_deg)
                        parameters["r_y"].append(r_y_deg)
                        parameters["r_z"].append(r_z_deg)
                        parameters["t_x"].append(acq_parameters[3])
                        parameters["t_y"].append(acq_parameters[4])
                        parameters["t_z"].append(acq_parameters[5])
                        
        return parameters

    def convert_rotations_to_degrees(self , x: float, y: float, z: float, input_rotation_unit: str)  -> list[float]:
        if input_rotation_unit == 'versor':
            return self.versor_to_degrees(x, y, z)

        elif input_rotation_unit == 'radians':
            return self.radians_to_degrees(x, y, z)

        elif input_rotation_unit == 'degrees':
            return [x, y, z]

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


    def get_intravolume_motion_flagged_volumes(self, 
                                               num_volumes: int, 
                                               num_slice_groups: int,
                                               displacement_values: list[float], 
                                               mm_displacement_threshold: float) -> list[int]:
            
            motion_flagged_volumes: list[int] = []
            for volume_num in range(num_volumes):
                displacements_at_this_volume:list[float] = displacement_values[volume_num*num_slice_groups:(volume_num*num_slice_groups) + num_slice_groups]
                if any([displacement > mm_displacement_threshold for displacement in displacements_at_this_volume]):
                    motion_flagged_volumes.append(volume_num)
            
            return motion_flagged_volumes

if __name__ == "__main__":
    import argparse
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Plot Frame-wise Displacement (calculated by fMRIPrep) vs Slice-wise Displacement (calculated by our software)"
    )
    parser.add_argument(
        "--intravolume_transform_directory",
        required=True,
        help="The directory of transforms outputted by the Intravolume Motion characterization software."
    )
    parser.add_argument(
        "--framewise_transform_directory",
        required=True,
        help=\
            "The directory of transforms outputted by running the confounds fmriprep file through " \
            "script: fmriprep-motion-est_to_transform_directory.py."
    )
    parser.add_argument(
        "--json_file_path",
        required=True
    )
    parser.add_argument(
        "--mm_motion_threshold",
        required=True,
        type=float,
        help="The threshold, in mm, for what is considered 'too much' motion."
    )
    parser.add_argument(
        "--output_directory",
        required=False,
        default="outputs",
        help=f"Default: {os.path.abspath('outputs')}."
    )
    parser.add_argument(
        "--also_save_to_png",
        required=False,
        action="store_true",
        help="Flag for saving both an .html file and a .png file."
    )
    args: argparse.Namespace = parser.parse_args()
    CompareFDvsSD(
        intravolume_transform_directory=os.path.abspath(args.intravolume_transform_directory),
        framewise_transform_directory=os.path.abspath(args.framewise_transform_directory),
        json_file_path=os.path.abspath(args.json_file_path),
        mm_motion_threshold=args.mm_motion_threshold,
        output_directory=os.path.abspath(args.output_directory),
        also_save_png=args.also_save_to_png
    )