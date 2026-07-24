import os
from matplotlib import patches, pyplot as plt 

from graph_transforms import GraphTransformDirectory

class CompareTransformDirectories:
    def __init__(self, 
                 transform_directories, 
                 transform_directory_labels,
                 transform_directory_colors,
                 transform_rotation_units, 
                 transform_suffixes,
                 json_paths, 
                 working_directory,
                 output_plot_path,
                 plot_title = "Motion Characterization Comparisons",
                 threshold_as_percent = None
                 ):

        """
        STEP ONE: Get information for transform data
        """
        graphers = []
        for directory_num, transform_directory in enumerate(transform_directories):
            grapher = GraphTransformDirectory(
                transform_directory=transform_directory,
                json_path=json_paths[directory_num],
                output_directory=working_directory,
                input_rotation_unit=transform_rotation_units[directory_num],
                threshold_as_percent_of_voxel=threshold_as_percent,
                transform_suffix=transform_suffixes[directory_num]

            )
            graphers.append(grapher)
        
        """
        STEP THREE: Create the subplots (one for each of the parameters, one for the displacements)
        """
        fig = plt.figure(figsize=(14, 8.5))
        fig.suptitle(plot_title, fontweight='bold')
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
            
            for grapher_num, grapher in enumerate(graphers):
                parameter_axes[dimension_num].plot(
                    range(len(grapher.parameters)),
                    [parameter_list[dimension_num] for parameter_list in grapher.parameters],
                    label=transform_directory_labels[grapher_num],
                    color=transform_directory_colors[grapher_num],
                    alpha=0.75
                )

            parameter_axes[dimension_num].grid()
            parameter_axes[dimension_num].set_xticks(graphers[0].xaxis_ticks)
            parameter_axes[dimension_num].set_xticklabels(graphers[0].xaxis_tick_labels)
        
        """
        STEP FOUR: Plot the displacements
        """
        displacement_axes.set_title("Displacements")
        displacement_axes.set_xlabel("Volume Number")
        displacement_axes.set_ylabel("Milimeters")
        for grapher_num, grapher in enumerate(graphers):
            displacement_axes.plot(
                range(len(grapher.displacements)),
                grapher.displacements,
                color=transform_directory_colors[grapher_num],
                alpha=0.75
            )
        if threshold_as_percent:
            # plot threshold as gray dashed line 
            displacement_axes.axhline(
                y=graphers[0].mm_displacement_threshold, 
                color='gray', 
                linestyle='--', 
                label=f"Motion Threshold:\n{graphers[0].mm_displacement_threshold} mm / {threshold_as_percent}% of Voxel Width"
                )
        """
        STEP FIVE: Plot motion flags
        """
        # plot slice level motion flags
        flag_height = max(max([grapher.displacements for grapher in graphers]))
        
        for grapher_num, grapher in enumerate(graphers):
            for i, motion_flagged_volume_num in enumerate(grapher.motion_flagged_volumes):
                if i == 0:
                    rect = patches.Rectangle(
                            xy=(motion_flagged_volume_num * grapher.num_slice_groups, 0),
                            width=grapher.num_slice_groups,
                            height=flag_height,
                            facecolor=transform_directory_colors[grapher_num],
                            edgecolor=transform_directory_colors[grapher_num],
                            alpha=0.25,
                            label=f"{transform_directory_labels[grapher_num]} Motion-Flagged Volume ({len(grapher.motion_flagged_volumes)} Total Flagged Volumes)"
                        )
                else:
                    rect = patches.Rectangle(
                            xy=(motion_flagged_volume_num * grapher.num_slice_groups, 0),
                            width=grapher.num_slice_groups,
                            height=flag_height,
                            facecolor=transform_directory_colors[grapher_num],
                            edgecolor=transform_directory_colors[grapher_num],
                            alpha=0.25
                        )
                displacement_axes.add_patch(rect)
        if threshold_as_percent:
            displacement_axes.legend()
        displacement_axes.grid()
        displacement_axes.set_xticks(graphers[0].xaxis_ticks)
        displacement_axes.set_xticklabels(graphers[0].xaxis_tick_labels)

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
    python graph_mutliple_transforms.py \
        --transform_directories data/slice_level_refvol2/outputs data/volume_level_refvol2/outputs \
        --json_paths data/p004-ses-04_func-bold_task-NFB2_20250827181314_24.json data/p004-ses-04_func-bold_task-NFB2_20250827181314_24.json \
        --transform_directory_labels "Initialized with Last Slice Group Transform" "Initialized with Volume Transform" \
        --transform_directory_colors "blue" "red" \
        --transform_rotation_units versor versor \
        --transform_suffixes .tfm .tfm \
        --plot_title 'Volume Level Initial Transform vs. Slice Level Initial Transform Results' \
        --threshold 25
    """

    default_plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motion_comparisons.png")
    default_working_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working")
    
    import argparse
    parser = argparse.ArgumentParser(
        description="Graph a transform of volume-level transforms vs. a transform of slice-level transforms"
    )
    parser.add_argument(
        "--transform_directories",
        required=True,
        nargs='+'
    )
    parser.add_argument(
        "--json_paths",
        required=True,
        help="Must match the order and number of transform directories.",
        nargs='+'

    )
    parser.add_argument(
        "--transform_directory_labels",
        required=True,
        help="Must match the order and number of transform directories.",
        nargs='+'

    )
    parser.add_argument(
        "--transform_directory_colors",
        required=True,
        help="Must match the order and number of transform directories.",
        nargs='+'

    )
    parser.add_argument(
        "--transform_rotation_units",
        required=True,
        choices=['versor', 'radians', 'degrees'],
        help="Must match the order and number of transform directories.",
        nargs='+'
    )
    parser.add_argument(
        "--transform_suffixes",
        required=False,
        choices=['.tfm', '.txt'],
        default='.tfm',
        help="The file extension of all the transforms in the directories. Default: '.tfm'. Must match the order and number of transform directories.",
        nargs='+'
    )
    parser.add_argument(
        "--working_directory",
        default=default_working_directory,
        help=f"Default: {default_working_directory}"
    )
    parser.add_argument(
        "--plot_title",
        default="Motion Characterization Comparisons",
        help="Default: 'Motion Characterization Comparisons'"
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
        default=None,
        help="Threshold as a percent of the width of a single voxel. Default: No Threshold"
    )
    args = parser.parse_args()

    CompareTransformDirectories(
        transform_directories=[
            os.path.abspath(transform_directory)
            for transform_directory in args.transform_directories
        ],
        json_paths=[
            os.path.abspath(json_path)
            for json_path in args.json_paths

        ],
        transform_directory_labels=args.transform_directory_labels,
        transform_directory_colors=args.transform_directory_colors,
        transform_rotation_units=args.transform_rotation_units,
        transform_suffixes=args.transform_suffixes,
        working_directory=os.path.abspath(args.working_directory),
        plot_title=args.plot_title,
        output_plot_path=args.output_plot_path,
        threshold_as_percent=args.threshold
    )