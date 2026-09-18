import os 
import json
import warnings
import pandas as pd 
from typing import Any
from matplotlib import pyplot as plt 
from matplotlib import patches as patches

class CompareFDvsSD:
    def __init__(self, 
                 intravolume_displacements_text_file: str, 
                 fmriprep_confounds_tsv_file: str,
                 json_file_path: str,
                 mm_threshold: float,
                 output_graph_path: str = "fd-vs-sd.png") -> None:

        slice_timing: list[float] = self.load_slice_timing(json_file_path)
        print(f"Slice Timing: {slice_timing}")

        t_r: float = self.get_repetition_time(json_file_path)
        print(f"Repetition Time: {t_r}s")

        intravolume_displacements: list[float] =  [0] + self.load_intravolume_displacements(intravolume_displacements_text_file)
        print(f"{len(intravolume_displacements)} Intravolume Displacement Values Loaded.")
        
        framewise_displacements: list[float] = self.load_fmriprep_displacements(fmriprep_confounds_tsv_file)
        print(f"{len(framewise_displacements)} Framewise Displacement Values Loaded.")

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

        intravolume_timing: list[float] = []
        for aquisition_num in range(len(intravolume_displacements)):
            volume_num: int = aquisition_num // num_slice_groups
            volume_start_time: float = volume_num * t_r
            slice_group_num: int = aquisition_num - (volume_num * num_slice_groups)
            slice_group_time_since_vol_start: float = slice_timing[slice_group_num]
            aquisition_time: float = volume_start_time + slice_group_time_since_vol_start
            intravolume_timing.append(aquisition_time)

        volume_timing: list[float] = [
            (volume_num + 1) * t_r
            for volume_num in range(num_volumes)
        ]
        
        plt.title("Frame-Wise vs. Intra-Frame-Wise Motion")
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement (mm)")

        """plt.scatter(
            intravolume_timing,
            intravolume_displacements,
            label="",
            s=0.75,
            color="red"
        )"""
        plt.plot(
            intravolume_timing,
            intravolume_displacements,
            label="Intravolume Displacements",
            alpha=0.75,
            color="red",
            linewidth=0.5
        )
        plt.scatter(
            volume_timing,
            framewise_displacements,
            label="",
            s=2,
            color="blue"
        )
        plt.plot(
            volume_timing,
            framewise_displacements,
            label="Framewise Displacements",
            alpha=0.75,
            color="blue",
            linewidth=1
        )
        plt.axhline(
            y=mm_threshold,
            linestyle="--",
            color="black",
            label=f"Motion Threshold: {mm_threshold} mm"
        )
        for i, volume_num in enumerate(framewise_motion_flags):
            plt.axvspan(
                volume_num * t_r,
                (volume_num + 1) * t_r,
                facecolor="blue",
                edgecolor='none',
                alpha=0.15,
                label=f"Framewise Motion Flags ({len(framewise_motion_flags)} / {num_volumes} Volumes)" if i == 0 else ""
            )

        for i, volume_num in enumerate(intravolume_motion_flags):
            plt.axvspan(
                volume_num * t_r,
                (volume_num + 1) * t_r,
                facecolor="red",
                edgecolor='none',
                alpha=0.15,
                label=f"Intravolume Motion Flags ({len(intravolume_motion_flags)} / {num_volumes} Volumes)" if i == 0 else ""
            )
        
        plt.grid()
        plt.legend(fontsize=8)
        plt.savefig(output_graph_path)
        plt.show()
        print(f"Plot at: {output_graph_path}")

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