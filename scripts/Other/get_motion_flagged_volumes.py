import os 
import json
import numpy as np
import pandas as pd 
from typing import Any
from plotly import graph_objects as go 

class GetMotionFlaggedVolumes:
    def __init__(self, 
                 json_file_path: str,
                 displacements: list[float] | None = None,  # pyright: ignore[reportRedeclaration]
                 displacement_text_file: str | None = None,
                 mm_threshold: float = 0.4157,
                 output_text_file: str | None = None,
                 output_graph_file: str | None = None,
                 also_save_graph_to_png: bool = True,
                 lambda_val: float = 0.8) -> None:

        if not displacements:
            if not displacement_text_file:
                raise ValueError("Please enter value(s) for arg: 'displacements' or 'displacment_text_file'.")
            print(f"Loading Displacements from: {displacement_text_file}")
            displacements: list[float] = self.load_displacements(displacement_text_file) # pyright: ignore[reportAssignmentType, reportRedeclaration]
        displacements: pd.Series = pd.Series(displacements)
        print(f"Received {len(displacements)} Displacement Values")

        mean_displacement: float = displacements.mean()
        if mean_displacement < 0.15:
            level_of_movement: str = 'low'
        elif 0.15 <= mean_displacement <= 0.3:
            level_of_movement: str = 'medium'
        else:
            level_of_movement: str = 'high'
        print(f"Participant is a {level_of_movement.capitalize()} Mover.")

        num_slice_groups_per_volume: int = self.get_num_slice_groups_per_volume(json_file_path)
        print(f"There are {num_slice_groups_per_volume} Aquisitions per Volume")
        
        exponential_averages: pd.Series = displacements.ewm(
            alpha=1 - lambda_val,
            adjust=False
        ).mean()
        print(f"Outputted Exponential Averages:\n{exponential_averages}")

        motion_flagged_volumes: list[int] = []
        for displacement_num, (displacement_value, exponential_average_value) in enumerate(zip(displacements, list(exponential_averages))):
            aquisition_num: int = displacement_num + 1
            in_volume_num: int = aquisition_num // num_slice_groups_per_volume

            print(f"\nProcessing Displacement Number: {displacement_num + 1} of {len(displacements)}")
            print(f"Aquisition Number: {aquisition_num}")
            print(f"This Aquisition is in Volume Number: {in_volume_num}")
            print(f"Displacement Value: {displacement_value} mm")
            print(f"Exponential Average: {exponential_average_value} mm")
            print(f"Above Motion Threshold: {exponential_average_value > mm_threshold}")

            if exponential_average_value > mm_threshold:
                if not in_volume_num in motion_flagged_volumes:
                    motion_flagged_volumes.append(in_volume_num)
        
        print(f"\nMotion Flagged Volumes: {motion_flagged_volumes}")
        print(f"We Flagged {len(motion_flagged_volumes)} Volumes of {(len(displacements) + 1) // num_slice_groups_per_volume} Volumes")

        if output_text_file:
            self.flagged_list_to_text_file(
                motion_flagged_volumes,
                output_file_path=output_text_file
            )
            print(f"Motion Flagged Volume List Written to: {output_text_file}")

        if output_graph_file:
            print("Plotting ...")
            fig = go.Figure()
            fig.update_layout(
                title_text=f"Motion Flagging Script Outputs: {level_of_movement.capitalize()} Mover",
                hovermode="x unified"
            )
            fig.add_trace(go.Scatter(
                x=list(range(len(displacements))),
                y=displacements,
                mode="lines+markers",
                name="Displacement Values"
            ))
            fig.add_trace(go.Scatter(
                x=list(range(len(exponential_averages))),
                y=exponential_averages,
                mode="lines+markers",
                name=f"Exponential Averages (λ = {round(lambda_val, 2)})"
            ))
            # plot motion flags 
            for i, volume_num in enumerate(motion_flagged_volumes):
                first_aquisition_num = volume_num * num_slice_groups_per_volume 
                last_aquisition_num = first_aquisition_num + (num_slice_groups_per_volume - 1)
                fig.add_trace(go.Scatter(
                    x=[first_aquisition_num, 
                       first_aquisition_num, 
                       last_aquisition_num, 
                       last_aquisition_num],
                    y=[min(displacements) - 0.1, 
                       max(displacements) * 1.1, 
                       max(displacements) * 1.1, 
                       min(displacements) - 0.1],
                    fill="toself",
                    fillcolor="rgba(0,0,0,0.25)",
                    line=dict(width=0),
                    legendgroup="motion_flags",
                    mode="none",
                    name=f"Motion Flags: {len(motion_flagged_volumes)} of {(len(displacements) + 1) // num_slice_groups_per_volume} Volumes Flagged",
                    showlegend=True if i == 0 else False,
                    hoverinfo="skip",
                ))

            # plot threshold
            fig.add_hline(
                y=mm_threshold  ,
                line=dict(color="black", width=2, dash="dash"),
                name=f"Exponential Average Threshold: {round(mm_threshold, 4)} mm",
                showlegend=True
            )

            fig.update_xaxes(title_text="Aquisition")
            fig.update_yaxes(title_text="Millimeters")
            
            fig.write_html(output_graph_file)
            print(f".html Plot at: {output_graph_file}")

            if also_save_graph_to_png:
                print("Saving Plot to .png File...")
                try:
                    fig.write_image(
                        output_graph_file.replace(".html", ".png"), 
                        width=1400,
                        height=900,
                        scale=2
                    )
                    print(f".png Plot at: {output_graph_file.replace('.html', '.png')}")
                
                except Exception as e:
                    if "BrowserFailedError" in type(e).__name__ or "browser seemed to close" in str(e):
                        raise RuntimeError(
                            f"Chrome failed to launch for image export: {e}. "
                            f"\n\nBCH RESEARCHERS: If you are on a login node on E3, try re-running on a compute node."
                        )
                    else:
                        raise


    def load_displacements(self, displacement_text_file: str) -> list[float]:
        displacements: list[float] = []
        with open(displacement_text_file, mode='r') as file:
            for line in file:
                if not line.strip():
                    continue 
                displacements.append(float(line.strip()))
        return displacements

    def get_num_slice_groups_per_volume(self, json_file_path: str) -> int:
        with open(json_file_path, mode='r') as file:
            data: dict[str, Any] = json.load(file)
            if not 'SliceTiming' in data:
                raise KeyError(f"Key: 'SliceTiming' not found in JSON File: {json_file_path}")

            return len(set(data['SliceTiming']))

    def flagged_list_to_text_file(self, motion_flagged_volumes: list[int], output_file_path: str):
        with open(output_file_path, mode='w') as file:
            for volume_num in motion_flagged_volumes:
                file.write(str(volume_num) + "\n")

class GetMotionFlaggedVolumes_ScanParams:
    def __init__(self,
                 json_file_path: str,
                 displacements: list[float] | None = None,  # pyright: ignore[reportRedeclaration]
                 displacement_text_file: str | None = None,
                 output_text_file: str | None = None, # pyright: ignore[reportRedeclaration]
                 output_graph_file: str | None = None, # pyright: ignore[reportRedeclaration]
                 also_save_graph_to_png: bool = True,
                 lambda_vals: list[float] = [round(val, 2) for val in np.arange(stop=0.95, start=0.7, step=0.05)] # pyright: ignore[reportArgumentType]
                 ) -> None:
        
        print(f"Testing Lambda Values: {[str(val) for val in lambda_vals]}")

        if not displacements:
            if not displacement_text_file:
                raise ValueError("Please enter value(s) for arg: 'displacements' or 'displacment_text_file'.")
            print(f"Loading Displacements from: {displacement_text_file}")
            displacements: list[float] = self.load_displacements(displacement_text_file) # pyright: ignore[reportAssignmentType, reportRedeclaration]
        displacements: pd.Series = pd.Series(displacements)
        print(f"Received {len(displacements)} Displacement Values")

        displacement_percentiles: list[float] = list(np.arange(start=0.5, stop=0.9, step=0.05)) # pyright: ignore[reportAssignmentType]
        mm_thresholds: list[float] = [
            displacements.quantile(displacement_percentile)
            for displacement_percentile in displacement_percentiles
        ] # pyright: ignore[reportAssignmentType]
        print(f"Testing Displacement Thresholds (in mm): {[str(round(val, 4)) for val in mm_thresholds]}")

        original_output_text_file: str | None = output_text_file
        original_output_graph_file: str | None = output_graph_file
        for lambda_val in lambda_vals:
            for perc_threshold, mm_threshold in zip(displacement_percentiles, mm_thresholds):
                output_text_file: str | None = \
                    original_output_text_file.replace(".txt", f"_thr-{round(perc_threshold, 4)}-percentile_lambda-{round(lambda_val, 2)}.txt") \
                    if original_output_text_file else None

                output_graph_file: str | None = \
                    original_output_graph_file.replace(".html", f"_thr-{round(perc_threshold, 4)}-percentile_lambda-{round(lambda_val, 2)}.html") \
                    if original_output_graph_file else None
                
                GetMotionFlaggedVolumes(
                    json_file_path=json_file_path,
                    displacements=list(displacements),
                    mm_threshold=mm_threshold,
                    output_text_file=output_text_file,
                    output_graph_file=output_graph_file,
                    also_save_graph_to_png=False,
                    lambda_val=lambda_val
                )

    def load_displacements(self, displacement_text_file: str) -> list[float]:
            displacements: list[float] = []
            with open(displacement_text_file, mode='r') as file:
                for line in file:
                    if not line.strip():
                        continue 
                    displacements.append(float(line.strip()))
            return displacements
    
if __name__ == "__main__":
    import argparse
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=\
            "Calculate with 3D Volumes to Flag as Having Intravolume Motion " + \
            "from the Displacement Values."
    )
    parser.add_argument(
            "--json_file_path",
            required=True
        )
    parser.add_argument(
        "--displacements",
        type=float,
        nargs="+",
        required=False,
        default=None,
        help=\
            "A list of the displacement values. " + \
            "Please enter value(s) for arg: '--displacements' or '--displacment_text_file'"
    )
    parser.add_argument(
        "--displacement_text_file",
        required=False,
        default=None,
        help=\
            "The text file containing a list of the displacement values. " + \
            "Please enter value(s) for arg: '--displacements' or '--displacment_text_file'"
    )
    parser.add_argument(
        "--mm_threshold",
        type=float,
        required=False,
        default="0.4157",
        help="The mm threshold where a volume has 'too much' intravolume motion. Default: 0.4157mm"
    )
    parser.add_argument(
        "--output_text_file",
        required=False,
        default=None,
        help="If you want to export the list of flagged volumes to a text file, add a .txt output path here."
    )
    parser.add_argument(
        "--output_graph_file",
        required=False,
        default=None,
        help="For graphing displacements, exponential averages, and motion flags, add a .html output path here."
    )
    parser.add_argument(
        "--scan_mutliple_params",
        action='store_true',
        help="Flag for trying out a series of lamba values and thresholds."
    )
    args: argparse.Namespace = parser.parse_args()

    if not args.scan_mutliple_params:
        GetMotionFlaggedVolumes(
            json_file_path=os.path.abspath(args.json_file_path),
            displacements=args.displacements,
            displacement_text_file=\
                os.path.abspath(args.displacement_text_file) 
                if args.displacement_text_file else None,
            mm_threshold=args.mm_threshold,
            output_text_file=\
                os.path.abspath(args.output_text_file)
                if args.output_text_file else None,
            output_graph_file=\
                os.path.abspath(args.output_graph_file)
                if args.output_graph_file else None
        )
    else:
        GetMotionFlaggedVolumes_ScanParams(
            json_file_path=os.path.abspath(args.json_file_path),
            displacements=args.displacements,
            displacement_text_file=\
                os.path.abspath(args.displacement_text_file) 
                if args.displacement_text_file else None,
            output_text_file=\
                os.path.abspath(args.output_text_file)
                if args.output_text_file else None,
            output_graph_file=\
                os.path.abspath(args.output_graph_file)
                if args.output_graph_file else None
        )