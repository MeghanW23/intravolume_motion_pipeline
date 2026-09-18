import os 
import json 
import warnings
import numpy as np
import pandas as pd
from typing import Any
from matplotlib import pyplot as plt 

class CompareFDvsSD:
    def __init__(self, 
                 intravolume_displacements_text_file: str, 
                 fmriprep_confounds_tsv_file: str,
                 json_file_path: str,
                 mm_threshold: float,
                 output_graph_path: str = "fd-vs-sd.png") -> None:

        # Get input data 
        slice_timing: list[float] = self.load_slice_timing(json_file_path)
        print(f"Slice Timing: {slice_timing}")

        t_r: float = self.get_repetition_time(json_file_path)
        print(f"Repetition Time: {t_r}s")

        intravolume_displacements: list[float] =  [0] + self.load_intravolume_displacements(intravolume_displacements_text_file)
        print(f"{len(intravolume_displacements)} Intravolume Displacement Values Loaded.")
        intravolume_displacements_min = np.nanmin(intravolume_displacements)
        intravolume_displacements_max = np.nanmax(intravolume_displacements)
        print(f"Intra-Frame-Wise Displacement Range: {round(intravolume_displacements_min, 4)} mm to {round(intravolume_displacements_max, 4)} mm")
        
        framewise_displacements: list[float] = self.load_fmriprep_displacements(fmriprep_confounds_tsv_file)
        print(f"{len(framewise_displacements)} Framewise Displacement Values Loaded.")
        framewise_displacements_min = np.nanmin(framewise_displacements)
        framewise_displacements_max = np.nanmax(framewise_displacements)
        print(f"Framewise Displacement Range: {round(framewise_displacements_min, 4)} mm to {round(framewise_displacements_max, 4)} mm")

        num_volumes: int = len(framewise_displacements)
        print(f"Number of Volumes: {num_volumes}")
        num_slice_groups: float | int = len(intravolume_displacements) / len(framewise_displacements) # pyright: ignore[reportRedeclaration, reportAssignmentType]
        if not num_slice_groups.is_integer(): # pyright: ignore[reportAttributeAccessIssue]
            warnings.warn(
                message=\
                    f"Your number of slice groups is not an integer: {num_slice_groups}. " + \
                    f"We will cast it to integer: {int(num_slice_groups)}"
            )
        num_slice_groups: int = int(num_slice_groups)
        print(f"Number of Slice Groups: {num_slice_groups}")

        # Get motion flags 
        intravolume_motion_flags: list[int] = []
        for aquisition_num, displacement_value in enumerate(intravolume_displacements):
            volume_num: int = aquisition_num // num_slice_groups
            if displacement_value > mm_threshold:
                if not volume_num in intravolume_motion_flags:
                    intravolume_motion_flags.append(volume_num)
        print(f"{len(intravolume_motion_flags)} of {num_volumes} Volumes Flagged for Intravolume Motion")

        framewise_motion_flags: list[int] = []
        for volume_num, displacement_value in enumerate(framewise_displacements):
            if displacement_value > mm_threshold:
                if not displacement_value in framewise_motion_flags:
                    framewise_motion_flags.append(volume_num)
        print(f"{len(framewise_motion_flags)} of {num_volumes} Volumes Flagged for Framewise Motion")

        # Plot results 
        fig, (axis1, axis2) = plt.subplots(
            nrows=2, ncols=1, 
            sharex=True, gridspec_kw={'height_ratios': [1, 4]},
            figsize=(8, 4))
        fig.suptitle("Frame-Wise vs Intra-Frame-Wise Motion", fontsize=10, fontweight='bold')

        sorted_intravolume_displacements: dict[int, list[float]] = {}
        for aquisition_num, displacement_value in enumerate(intravolume_displacements):
            volume_num: int = aquisition_num // num_slice_groups
            if not volume_num in sorted_intravolume_displacements:
                sorted_intravolume_displacements[volume_num] = []
            sorted_intravolume_displacements[volume_num].append(displacement_value)

        all_mins: list[float] = []
        all_maxes: list[float] = []
        for volume_num, intra_disp_list in sorted_intravolume_displacements.items(): # pyright: ignore[reportAssignmentType]
            intra_disp_list: pd.Series = pd.Series(intra_disp_list)
            
            all_mins.append(intra_disp_list.min())

            all_maxes.append(intra_disp_list.max())
            
            axis2.fill_between(
                [volume_num, volume_num + 1],
                intra_disp_list.quantile(0.05), y2=intra_disp_list.quantile(0.95),
                color='red', alpha=0.25, lw=0,
                label='Intra-Frame-Wise (Per-Volume 5% and 95% Quartiles)' if volume_num == 0 else None
            )

        axis2.set_xlabel("Volume Number")

        axis2.set_title("Motion Traces", fontsize=10, loc='left')
        axis2.plot(
            [volume_num + 0.5 for volume_num in range(num_volumes)],
            all_maxes,
            label='Intra-Frame-Wise (Per-Volume Max)',
            color='red',
            linewidth=2
        )
        axis2.scatter(
            [volume_num + 0.5 for volume_num in range(num_volumes)],
            all_maxes,
            color='red',
            s=5
        )
        axis2.plot(
            [volume_num + 0.5 for volume_num in range(num_volumes)],
            framewise_displacements,
            label='Frame-Wise',
            color='blue',
            linewidth=2
        )
        axis2.scatter(
            [volume_num + 0.5 for volume_num in range(num_volumes)],
            framewise_displacements,
            color='blue',
            s=5
        )
        axis2.axhline(
            y=mm_threshold,
            linestyle="--",
            color="black",
            label=f"Motion Threshold: {mm_threshold} mm"
        )
        axis2.set_ylabel("Displacement (mm)")
        axis2.legend(fontsize=8, loc='upper left')

        axis1.eventplot(
            [volume_num + 0.5 for volume_num in framewise_motion_flags], 
            color='blue',
            alpha=0.5,
            label="Frame-Wise Motion Flag"
        )
        axis1.eventplot(
            [volume_num + 0.5 for volume_num in intravolume_motion_flags], 
            color='red',
            alpha=0.5,
            label="Intra-Frame-Wise Motion Flag"
        )
        axis1.set_title("Motion Flags", fontsize=10, loc='left')
        axis1.set_ylim(0.5, 1.5)
        axis1.get_yaxis().set_visible(False)
        axis1.legend(fontsize=8, loc='upper left')
        

        fig.tight_layout()
        plt.subplots_adjust(
            left=0.083,
            bottom=0.11,
            right=0.981,
            top=0.898,
            wspace=0.2,
            hspace=0.232
        )

        plt.savefig(output_graph_path)
        print(f"Output Plot Saved to: {output_graph_path}")
        plt.show()

            

    def load_intravolume_displacements(self, displacements_text_file: str) -> list[float]:
        displacements: list[float] = []
        with open(displacements_text_file, mode='r') as file:
            for line in file:
                if not line.strip():
                    continue 
                else:
                    displacements.append(float(line.strip()))
        return displacements


    def load_fmriprep_displacements(self, fmriprep_confounds_tsv_file: str) -> list[float]:
        df: pd.DataFrame = pd.read_csv(fmriprep_confounds_tsv_file, sep='\t') 
        if not 'framewise_displacement' in df.keys():
            raise KeyError(
                f"The key: 'framewise_displacement' not found in your .tsv file: {fmriprep_confounds_tsv_file}"
            )
        return list(df['framewise_displacement'])


    def load_slice_timing(self, json_file_path: str) -> list[float]:
        with open(json_file_path, mode='r') as file:
            data: dict[str, Any] = json.load(file)
            if not 'SliceTiming' in data:
                raise KeyError(
                    f"The key: 'SliceTiming' was not found in your JSON file: {json_file_path}"
                )
            return sorted(list(set(data['SliceTiming'])))


    def get_repetition_time(self, json_file_path: str) -> float:
        with open(json_file_path, mode='r') as file:
            data: dict[str, Any] = json.load(file)
            if not 'RepetitionTime' in data:
                raise KeyError(
                    f"The key: 'RepetitionTime' was not found in your JSON file: {json_file_path}"
                )
            return float(data['RepetitionTime'])
        

if __name__ == "__main__":
    import argparse
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Compare Frame-Wise vs. Intra-Frame-Wise Motion"
    )
    parser.add_argument(
        "--intravolume_displacements_text_file",
        required=True,
        help="The displacements.txt file outputted by the Motion Characterization step."
    )
    parser.add_argument(
        "--fmriprep_confounds_tsv_file",
        required=True,
        help="The confounds .tsv file outputted by fMRIPrep (contains the 'framewise_displacement' column)"
    )
    parser.add_argument(
        "--json_file_path",
        required=True,
        help="The JSON Sidecar from fMRIPrep"
    )
    parser.add_argument(
        "--mm_threshold",
        type=float,
        default=0.4157,
        help="In mm. Default: 0.4157 mm"
    )
    parser.add_argument(
        "--output_graph_path",
        required=False,
        default="fd-vs-sd.png",
        help=f"Must be a .png file. Default: {os.path.abspath('fd-vs-sd.png')}"
    )
    args: argparse.Namespace = parser.parse_args()
    CompareFDvsSD(
        intravolume_displacements_text_file=os.path.abspath(args.intravolume_displacements_text_file),
        fmriprep_confounds_tsv_file=os.path.abspath(args.fmriprep_confounds_tsv_file),
        json_file_path=os.path.abspath(args.json_file_path),
        mm_threshold=args.mm_threshold,
        output_graph_path=os.path.abspath(args.output_graph_path)
    )