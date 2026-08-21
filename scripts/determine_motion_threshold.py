import os 
import json
import math
import statistics
import numpy as np
from typing import Any 
import SimpleITK as sitk
import matplotlib.pyplot as plt 

class DetermineMotionThreshold:
    """
    Steps:
    1. Calculate the histogram of the number of volumes that exceed a threshold, for all thresholds.
        - Stop once you hit 0% of volumes
    2. Find the threshold that corresponds to 25% of the data being excluded.
        - In general, if 20-30% of the data is above the threshold, there may be too little data
          left for reliable analysis
    3. Calculate a mean and standard deviation of the remaining 75% of volumes.
    4. Get mm motion threshold via: mean + (2 * standard deviation)
    """

    def __init__(self, 
                 displacements_text_file: str, 
                 nifti_image_path: str,
                 json_file_path: str, 
                 output_directory: str = "outputs",
                 percent_of_volumes_to_scrub: int = 25) -> None:

        os.makedirs(output_directory, exist_ok=True)
        if not (0 < percent_of_volumes_to_scrub < 100):
            raise ValueError("arg: 'percent_of_volumes_to_scrub' must be between 0 and 100")

        """
        ====================================================================
        Get Statistics On the Input Data 
        ====================================================================
        """
        num_slice_groups: int = self.get_num_slice_groups(json_file_path)
        print(f"Number of Slice Groups Per Volume: {num_slice_groups}") 

        voxel_spacing: tuple[float, float, float] = self.get_voxel_spacing(nifti_image_path)
        print(f"Voxel Spacing: {voxel_spacing}")

        displacements: list[float] = self.load_displacement_values(displacements_text_file)
        print(f"{len(displacements)} Displacement Values Loaded")

        num_volumes: int = (len(displacements) + 1) // num_slice_groups


        """
        ====================================================================
        Group Slice Group Data By It's Volume
        ====================================================================
        """
        print("Grouping Displacements by Volume")
        grouped_displacements: dict[int, list[float]] = {}
        for aquisition_num, displacement_value in enumerate(displacements, start=1):
            volume_num: int = aquisition_num // num_slice_groups
            if not volume_num in grouped_displacements:
                grouped_displacements[volume_num] = []
            grouped_displacements[volume_num].append(displacement_value)


        """
        ====================================================================
        Calculate The Number of Volumes Flagged Per Threshold
        ====================================================================
        """
        print("Getting Number of Volumes Flagged Per Threshold")
        num_volumes_flagged_per_threshold: dict[tuple[float, float], int] = {}  # (percent_threshold, mm_threshold): num_volumes_flagged
        running_percent_threshold: float = 0
        while True:
            mm_threshold: float = self.get_threshold_in_mm(voxel_spacing, threshold_as_percent=running_percent_threshold) 
            print(f"Testing threshold: {running_percent_threshold}% of a voxel (or {mm_threshold} mm)  ")

            num_volumes_flagged_per_threshold[(running_percent_threshold, mm_threshold)] = 0

            for volume_num, displacement_values in grouped_displacements.items():
                if any(displacement_value > mm_threshold for displacement_value in displacement_values):
                    num_volumes_flagged_per_threshold[(running_percent_threshold, mm_threshold)] += 1

            print(f"{num_volumes_flagged_per_threshold[(running_percent_threshold, mm_threshold)]} of {num_volumes} Volumes Flagged")

            if num_volumes_flagged_per_threshold[(running_percent_threshold, mm_threshold)] == 0:
                print("Done with search.")
                break
            else:
                running_percent_threshold += 0.5


        """
        ====================================================================
        Export Data on Number of Flagged Volumes by Threshold to CSV
        ====================================================================
        """
        output_csv_path: str = os.path.join(output_directory, "num_volumes_flagged_per_threshold.csv")
        self.write_results_to_csv_file(
            num_volumes_flagged_per_threshold,
            output_csv_path=output_csv_path
        )
        print(f"Thresholds and Flag Counts Written to CSV: {output_csv_path}")

        """
        ====================================================================
        Get Which Volumes to Keep Based on Their Max Slice Group Displacement
        ====================================================================
        """
        # Get the remaining volumes
        # any positive int number less than 100
        sorted_volumes: list[tuple[float, int]]  = sorted(
            [
                (max(displacement_list), volume_num)
                for volume_num, displacement_list in grouped_displacements.items()
            ], 
            key = lambda x: x[0]
        )
        num_volumes_to_keep: int = int(num_volumes * (1 - (percent_of_volumes_to_scrub / 100)))
        remaining_volumes: list[tuple[float, int]] = sorted_volumes[:num_volumes_to_keep]
        print(f"At a Threshold of {percent_of_volumes_to_scrub}%, We Keep {num_volumes_to_keep} Volumes:")
        print(", ".join(f"Volume {volume_num} ({round(max_disp_val, 2)} mm)" for max_disp_val, volume_num in remaining_volumes))

        """
        ====================================================================
        Calculate the Motion Threshold
        ====================================================================
        """
        all_disp_values_from_remaining_volumes: list[float] = [] 
        for volume_num, displacement_list in grouped_displacements.items():
            if volume_num in  [volume_num for _, volume_num in remaining_volumes]:
                all_disp_values_from_remaining_volumes.extend(displacement_list)

        mm_mean: float = statistics.mean(all_disp_values_from_remaining_volumes)
        print(f"Remaining Data Displacement Mean: {mm_mean} mm")

        mm_std: float = np.std(all_disp_values_from_remaining_volumes, ddof=1) # pyright: ignore[reportAssignmentType]
        print(f"Remaining Data Displacements STD: {mm_std} mm")

        self.motion_threshold: float = mm_mean + (2 * mm_std)
        print(f"Motion Threshold: {self.motion_threshold} mm")


        """
        ====================================================================
        Plot Results to Annotated Histogram
        ====================================================================
        """
        # get the mm volume cutoff as the halfway point between
        # the last volume we kept and the first volume we rejected
        last_saved_mm_val: float = sorted_volumes[num_volumes_to_keep - 1][0]
        print(
            f"Last Saved Volume: {sorted_volumes[num_volumes_to_keep - 1][1]}, " \
            f"Max Displacement: {sorted_volumes[num_volumes_to_keep - 1][0]} mm"
        )
        first_rejected_mm_val: float = sorted_volumes[num_volumes_to_keep][0]
        print(
            f"First Rejected Volume: {sorted_volumes[num_volumes_to_keep][1]}, " \
            f"Max Displacement: {sorted_volumes[num_volumes_to_keep][0]} mm"
        )
        mm_volume_cutoff: float = (last_saved_mm_val + first_rejected_mm_val) / 2
        print(f"Using mm volume cutoff: {mm_volume_cutoff}")

        output_plot_path: str = os.path.join(output_directory, "num_volumes_flagged_per_threshold.png")
        self.plot_histogram(num_volumes_flagged_per_threshold, 
                            num_volumes=num_volumes,
                            voxel_spacing=voxel_spacing,
                            output_plot_path=output_plot_path,
                            mm_volume_cutoff=mm_volume_cutoff,
                            mm_motion_threshold=self.motion_threshold)
        print(f"Plot Saved to: {output_plot_path}")
        
        print(f'Done. The Motion Threshold Selected is: {self.motion_threshold} mm.')

    def get_num_slice_groups(self, 
                             json_file_path: str) -> int:
        with open(json_file_path, mode='r') as file:
            json_data: dict[str, Any] = json.load(file)
            if not 'SliceTiming' in json_data:
                raise KeyError(f"Key: 'SliceTiming' is not in your JSON file: {json_file_path}")
            else:
                # return number of unique timing values in json_data['SliceTiming']
                return len(set(json_data['SliceTiming']))


    def get_voxel_spacing(self, 
                          nifti_image_path: str) -> tuple[float, float, float]:
        spacing: tuple[float, float, float] | tuple[float, float, float] = \
            sitk.ReadImage(nifti_image_path).GetSpacing()

        if len(spacing) < 3:
            raise ValueError("Image must be at least 3D.") 

        return spacing


    def load_displacement_values(self, 
                                 displacements_text_file: str) -> list[float]:
        displacements: list[float] = []
        with open(displacements_text_file, mode='r') as file:
            for line in file:
                if not line.strip():
                    continue 
                else:
                    displacements.append(float(line.strip()))
        return displacements


    def get_threshold_in_mm(self, 
                            spacing: tuple[float, float, float], 
                            threshold_as_percent: float) -> float:
            """
            To get the diagonal of a rectangular prism (3D rectangle):
            d = sqrt(dim_x^2 + dim_y^2 + dim_z^2)
            
            To get the threshold in mm:
            mm_threshold = d * (threshold_as_percent_of_voxel / 100)
            """
            d_x, d_y, d_z = spacing[:3]
            diagonal: float = math.sqrt(d_x ** 2 + d_y ** 2 + d_z ** 2)
            return diagonal * (threshold_as_percent / 100)


    def write_results_to_csv_file(self, 
                                  num_volumes_flagged_per_threshold: dict[tuple[float, float], int],
                                  output_csv_path: str):
        with open(output_csv_path, mode='w') as file:
            file.write(f"percent_threshold,mm_threshold,num_volumes_flagged\n")
            for (threshold_percent, threshold_mm), displacement_values in num_volumes_flagged_per_threshold.items():
                file.write(f"{threshold_percent},{threshold_mm},{displacement_values}\n")


    def plot_histogram(self,
                       num_volumes_flagged_per_threshold: dict[tuple[float, float], int], 
                       num_volumes: int,
                       voxel_spacing: tuple[float, float, float],
                       output_plot_path: str,
                       mm_volume_cutoff: float,
                       mm_motion_threshold: float):

        bins: list[float] = [
            mm_threshold
            for percent_threshold, mm_threshold
            in num_volumes_flagged_per_threshold.keys()
        ]

        counts: list[int] = list(num_volumes_flagged_per_threshold.values())

        plt.figure(figsize=(10, 6))

        bar_width = bins[1] - bins[0]

        plt.bar(
            x=bins,
            height=counts,
            width=bar_width,
            edgecolor="black"
        )

        # horizontal line for where all volumes are flagged 
        plt.axhline(
            y=num_volumes,
            linestyle="--",
            color='blue',
            linewidth=1
        )
        plt.annotate(
            text=f"All Volumes Flagged",
            xy=(-0.0001, num_volumes + 2),
            fontsize=8,
            color="blue"
        )

        # vertical line for 25% of volumes rejected
        plt.axvline(
            x=mm_volume_cutoff,
            linestyle="--",
            color="red",
            label=f"~25% of Volumes Have Displacements Greater Than ~{round(mm_volume_cutoff, 4)} mm"
        )

        # vertical line for motion threshold 
        plt.axvline(
            x=mm_motion_threshold,
            linestyle="--",
            color="black",
            label=f"Motion Threshold: {round(mm_motion_threshold, 4)} mm"
        )

        plt.legend(loc='upper right', fontsize=8)
        plt.title("Number of Flagged Volumes per Motion Threshold")
        plt.xlabel("Motion Threshold (in mm)")
        plt.ylabel("Number of Volumes Flagged")
        plt.savefig(output_plot_path)
        # plt.show()


    def return_motion_threshold(self) -> float:
        return self.motion_threshold


if __name__ == "__main__":
    import argparse
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Get the appropriate motion threshold based on the amount of intravolume motion that was detected."
    )
    parser.add_argument(
        "--displacements_text_file",
        required=True,
        help=\
            ".txt file containing the mm displacements of every slice group to the one before it. " \
            "See: https://github.com/MeghanW23/intravolume_motion_pipeline/blob/main/scripts/calculate_displacements.py"
    )
    parser.add_argument(
        "--nifti_image_path",
        required=True,
        help="NiFTI Path outputted by dcm2niix. To determine the spacing of a single voxel."
    )
    parser.add_argument(
        "--json_file_path",
        required=True,
        help="JSON Sidecar outputted by dcm2niix. To determine the number of slice groups."
    )
    parser.add_argument(
        "--percent_of_volumes_to_scrub",
        required=False,
        type=int,
        default=25,
        help="Default: 25 Percent"
    )
    parser.add_argument(
        "--output_directory",
        required=False,
        default="outputs",
        help=f"Default: {os.path.abspath('outputs')}"
    )
    args: argparse.Namespace = parser.parse_args()
    DetermineMotionThreshold(
        displacements_text_file=os.path.abspath(args.displacements_text_file),
        nifti_image_path=os.path.abspath(args.nifti_image_path),
        json_file_path=os.path.abspath(args.json_file_path),
        output_directory=os.path.abspath(args.output_directory),
        percent_of_volumes_to_scrub=args.percent_of_volumes_to_scrub
    )