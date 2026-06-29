import os 
from matplotlib import patches, pyplot as plt 
from graph_transforms import GraphTransformDirectory

class CompareTransformDirectories:
    def __init__(self, 
                 slice_transform_directory, 
                 slice_transform_rotation_unit, 
                 slice_level_transform_suffix,
                 volume_transform_directory, 
                 volume_transform_rotation_unit, 
                 volume_level_transform_suffix,
                 json_path, 
                 working_directory, 
                 output_plot_path, 
                 threshold_as_percent = None,
                 plot_tile = "Intravolume vs. Framewise Motion Characterization"):

        os.makedirs(working_directory, exist_ok=True)

        """
        STEP ONE: Get information for slice level, then volume level data
        """
        print("Getting Slice-Level Data")
        slice_grapher = GraphTransformDirectory(
            transform_directory=slice_transform_directory,
            json_path=json_path,
            output_directory=os.path.join(working_directory, "slice-level"),
            input_rotation_unit=slice_transform_rotation_unit,
            threshold_as_percent_of_voxel=threshold_as_percent,
            transform_suffix=slice_level_transform_suffix,
            do_not_plot=True
        )

        print("Getting Volume-Level Data")
        volume_grapher = GraphTransformDirectory(
            transform_directory=volume_transform_directory,
            json_path=json_path,
            output_directory=os.path.join(working_directory, "volume-level"),
            input_rotation_unit=volume_transform_rotation_unit,
            threshold_as_percent_of_voxel=threshold_as_percent,
            transform_suffix=volume_level_transform_suffix,
            framewise_displacements=True,
            do_not_plot=True
        )

        """
        STEP TWO: format volume-level data so it matches the number of slice-level data points 
        """
        # create 'num_slice_groups' number of displacement values and parameter lists at each volume
        volume_level_formatted_displacements = []
        for displacement in volume_grapher.displacements:
            volume_level_formatted_displacements.extend([displacement] * slice_grapher.num_slice_groups)

        volume_level_formatted_parameters = []
        for parameter_list in volume_grapher.parameters:
            volume_level_formatted_parameters.extend([parameter_list] * slice_grapher.num_slice_groups)

        fig = plt.figure(figsize=(14, 8.5))
        fig.suptitle(plot_tile, fontweight='bold')

        """
        STEP THREE: Create the subplots (one for each of the parameters, one for the displacements)
        """
        slice_color = "red"
        volume_color = "blue"
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

        """
        STEP FOUR: Plot the parameters
        """
        dimensions = ["X Rotation", "Y Rotation", "Z Rotation", "X Translation", "Y Translation", "Z Translation"]
        for dimension_num, dimension_name in enumerate(dimensions):
            parameter_axes[dimension_num].set_title(dimension_name)
            parameter_axes[dimension_num].set_xlabel("Volume Number")
            parameter_axes[dimension_num].set_ylabel("Milimeters" if dimension_num > 2 else "Degrees")
            
            # plot slice-level 
            parameter_axes[dimension_num].plot(
                range(len(slice_grapher.parameters)),
                [parameter_list[dimension_num] for parameter_list in slice_grapher.parameters],
                label="Intravolume Motion",
                color=slice_color
            )
            # plot formatted volume-level
            parameter_axes[dimension_num].plot(
                range(len(volume_level_formatted_parameters)),
                [parameter_list[dimension_num] for parameter_list in volume_level_formatted_parameters],
                label="Framewise Motion",
                color=volume_color
            )

            parameter_axes[dimension_num].grid()
            parameter_axes[dimension_num].set_xticks(slice_grapher.xaxis_ticks)
            parameter_axes[dimension_num].set_xticklabels(slice_grapher.xaxis_tick_labels)

        """
        STEP FOUR: Plot the displacements
        """
        displacement_axes.set_title("Displacements")
        displacement_axes.set_xlabel("Volume Number")
        displacement_axes.set_ylabel("Milimeters")
        displacement_axes.plot(
            range(len(slice_grapher.displacements)),
            slice_grapher.displacements,
            color=slice_color
        )
        displacement_axes.plot(
            range(len(volume_level_formatted_displacements)),
            volume_level_formatted_displacements,
            color=volume_color
        )
        if threshold_as_percent:
            # plot threshold as gray dashed line 
            displacement_axes.axhline(
                y=slice_grapher.mm_displacement_threshold, 
                color='gray', 
                linestyle='--', 
                label=f"Motion Threshold:\n{slice_grapher.mm_displacement_threshold} mm / {threshold_as_percent}% of Voxel Width"
                )

        """
        STEP FIVE: Plot motion flags
        """
        # plot slice level motion flags
        flag_height = max(max(volume_grapher.displacements), max(slice_grapher.displacements))
        
        for i, motion_flagged_volume_num in enumerate(slice_grapher.motion_flagged_volumes):
            if i == 0:
                rect = patches.Rectangle(
                        xy=(motion_flagged_volume_num * slice_grapher.num_slice_groups, 0),
                        width=slice_grapher.num_slice_groups,
                        height=flag_height,
                        facecolor=slice_color,
                        edgecolor=slice_color,
                        alpha=0.25,
                        label=f"Intravolume Motion-Flagged Volume ({len(slice_grapher.motion_flagged_volumes)} Total Flagged Volumes)"
                    )
            else:
                rect = patches.Rectangle(
                        xy=(motion_flagged_volume_num * slice_grapher.num_slice_groups, 0),
                        width=slice_grapher.num_slice_groups,
                        height=flag_height,
                        facecolor=slice_color,
                        edgecolor=slice_color,
                        alpha=0.25
                    )
            displacement_axes.add_patch(rect)

        # plot volume level motion flags
        for i, motion_flagged_volume_num in enumerate(volume_grapher.motion_flagged_volumes):
            if i == 0:
                rect = patches.Rectangle(
                        xy=(motion_flagged_volume_num * slice_grapher.num_slice_groups, 0),
                        width=slice_grapher.num_slice_groups,
                        height=flag_height,
                        facecolor=volume_color,
                        edgecolor=volume_color,
                        alpha=0.25,
                        label=f"Framewise Motion-Flagged Volume ({len(volume_grapher.motion_flagged_volumes)} Total Flagged Volumes)"
                    )
            else:
                rect = patches.Rectangle(
                        xy=(motion_flagged_volume_num * slice_grapher.num_slice_groups, 0),
                        width=slice_grapher.num_slice_groups,
                        height=flag_height,
                        facecolor=volume_color,
                        edgecolor=volume_color,
                        alpha=0.25
                    )
            displacement_axes.add_patch(rect)
        
        displacement_axes.legend()
        displacement_axes.grid()
        displacement_axes.set_xticks(slice_grapher.xaxis_ticks)
        displacement_axes.set_xticklabels(slice_grapher.xaxis_tick_labels)

        """
        STEP SIX: Save and close 
        """
        handles, labels = parameter_axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 0.965)
        )

        plt.tight_layout()
        plt.savefig(output_plot_path)
        plt.close()
        print(f"Graph Outputted to: {output_plot_path}")
        

        


if __name__ == "__main__":
    """
    python scripts/graph_volume_vs_intravolume_motion_transforms.py \
        --slice_level_transform_directory p004_nfb2/transforms/slice_level_transforms \
        --slice_level_transform_rotation_unit versor \
        --volume_level_transform_directory p004_nfb2/transforms/volume_level_transforms \
        --volume_level_transform_rotation_unit versor \
        --json_path data/p004-ses-04_func-bold_task-NFB2_20250827181314_24.json \
        --working_directory p004-working \
        --plot_title "P004 Ses-04 NFB2: Intravolume vs. Framewise Motion Characterization" \
        --output_plot_path p004_nfb2/fd_vs_intravolume_motion.png \
        --threshold 25
    """

    """
    python scripts/graph_volume_vs_intravolume_motion_transforms.py \
        --slice_level_transform_directory p012_nfb3/transforms/slice_level_transforms \
        --slice_level_transform_rotation_unit versor \
        --volume_level_transform_directory p012_nfb3/transforms/volume_level_transforms \
        --volume_level_transform_rotation_unit versor \
        --json_path data/p012-ses-03_working_func-bold_task-NFB3_20251212183456_36.json \
        --working_directory p012-working \
        --plot_title "P012 Ses-03 NFB3: Intravolume vs. Framewise Motion Characterization" \
        --output_plot_path p012_nfb3/fd_vs_intravolume_motion.png \
        --threshold 25
    """

    default_plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fd_vs_intravolume_motion.png")
    default_working_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working")
    
    import argparse
    parser = argparse.ArgumentParser(
        description="Graph a transform of volume-level transforms vs. a transform of slice-level transforms"
    )
    parser.add_argument(
        "--slice_level_transform_directory",
        required=True,
        help="These transforms must characterize intravolume motion, not framewise motion."
    )
    parser.add_argument(
        "--slice_level_transform_rotation_unit",
        required=True,
        choices=['versor', 'radians', 'degrees']
    )
    parser.add_argument(
        "--slice_level_transform_suffix",
        required=False,
        choices=['.tfm', '.txt'],
        default='.tfm',
        help="The file extension of all the transforms in the directory. Default: '.tfm'"
    )
    parser.add_argument(
        "--volume_level_transform_directory",
        required=True,
        help="These transforms must characterize framewise motion, not intravolume motion."
    )
    parser.add_argument(
        "--volume_level_transform_rotation_unit",
        required=True,
        choices=['versor', 'radians', 'degrees']
    )
    parser.add_argument(
        "--volume_level_transform_suffix",
        required=False,
        choices=['.tfm', '.txt'],
        default='.tfm',
        help="The file extension of all the transforms in the directory. Default: '.tfm'"
    )
    parser.add_argument(
        "--json_path",
        required=True
    )
    parser.add_argument(
        "--working_directory",
        default=default_working_directory,
        help=f"Default: {default_working_directory}"
    )
    parser.add_argument(
        "--plot_title",
        default="Intravolume vs. Framewise Motion Characterization",
        help="Default: 'Intravolume vs. Framewise Motion Characterization'"
    )
    parser.add_argument(
        "--output_plot_path",
        default=default_plot_path,
        help=f"Default: {default_plot_path}"
    )
    parser.add_argument(
        "--threshold",
        required=False,
        type=int,
        default=25,
        help="Threshold as a percent of the width of a single voxel. Default: 25"
    )
    args = parser.parse_args()

    CompareTransformDirectories(
        slice_transform_directory=os.path.abspath(args.slice_level_transform_directory),
        slice_transform_rotation_unit=args.slice_level_transform_rotation_unit,
        slice_level_transform_suffix=args.slice_level_transform_suffix,
        volume_transform_directory=os.path.abspath(args.volume_level_transform_directory),
        volume_transform_rotation_unit=args.volume_level_transform_rotation_unit,
        volume_level_transform_suffix=args.volume_level_transform_suffix,
        json_path=os.path.abspath(args.json_path),
        working_directory=os.path.abspath(args.working_directory),
        plot_tile=args.plot_title,
        output_plot_path=os.path.abspath(args.output_plot_path),
        threshold_as_percent=args.threshold
    )
