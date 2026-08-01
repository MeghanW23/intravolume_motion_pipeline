import os 
import json
import statistics
import numpy as np
import SimpleITK as sitk 
from collections import OrderedDict
from joblib import Parallel, delayed
from plotly import graph_objects as go
from plotly.subplots import make_subplots

class IdentifyReferenceVolume:
    def __init__(self, 
                 nifti_image_path: str , 
                 json_file_path: str, 
                 n_jobs: int = -1, 
                 output_directory: str | None = None) -> None:

        if output_directory:
            os.makedirs(output_directory, exist_ok=True)

        # Load data
        nifti_image: sitk.Image = sitk.ReadImage(nifti_image_path)
        dimensions: tuple[float, float, float] = nifti_image.GetSize()
        print(f"Input Image Dimensions: {dimensions}")

        if len(dimensions) != 4:
            raise ValueError("Your Input NiFTI Image must be 4D.")

        slice_timing: OrderedDict[float, list[int]] = self.get_slice_timing(json_file_path)
        print(f"Slice Timing: {slice_timing}")

        # Get all MI values 
        results: list[dict[int, list[float]]] = Parallel(
            n_jobs=n_jobs, return_as="list")(
            delayed(self.compare_volumes)(
                volume_num1=i - 1,
                volume_num2=i,
                nifti_image_path=nifti_image_path,
                slice_timing=slice_timing
            )
            for i in range(1, dimensions[-1])
        ) # pyright: ignore[reportAssignmentType]
        mutual_info_dict: dict[int, list[float]] = {}
        for result in results:
            mutual_info_dict[list(result.keys())[0]] = list(result.values())[0] # pyright: ignore[reportOptionalMemberAccess]

        # Save files to output directory 
        if output_directory:
            # Save to JSON file
            output_json_path: str = os.path.join(output_directory, "mutual_info_values.json")
            with open(output_json_path, mode='w') as file:
                json.dump(
                    mutual_info_dict,
                    fp=file
                )
            print(f"Saved Mutual Info Values to: {output_json_path}")

            self.plot_all_mutual_info_values(
                mutual_info_dict=mutual_info_dict,
                output_directory=output_directory,
                slice_timing=slice_timing,
                num_volumes=dimensions[-1],
                nifti_image_path=nifti_image_path
            )

        mi_means_per_volume: list[float] = [
            statistics.mean(volume_mi_list)
            for volume_mi_list in list(mutual_info_dict.values())
        ]      
        mi_min_per_volume: list[float] = [
            min(mi_list_in_volume)
            for mi_list_in_volume in list(mutual_info_dict.values())
        ]
        mi_range_per_volume: list[float] = [
            max(mi_list_in_volume) - min(mi_list_in_volume)
            for mi_list_in_volume in list(mutual_info_dict.values())
        ]

        if output_directory:
            output_min_per_vol_path: str = os.path.join(output_directory, "mutual_info_minimum_per_volume.txt")
            with open(output_min_per_vol_path, mode='w') as file:
                for value in mi_min_per_volume:
                    file.write(str(value) + '\n')
            print(f"Miniumum Mutual Info Per Volume Saved to: {output_min_per_vol_path}")

            output_range_per_vol_path: str = os.path.join(output_directory, "mutual_info_range_per_volume.txt")
            with open(output_range_per_vol_path, mode='w') as file:
                for value in mi_range_per_volume:
                    file.write(str(value) + '\n')
            print(f"Mutual Info Range Per Volume Saved to: {output_range_per_vol_path}")

            self.plot_min_and_range(
                output_directory=output_directory,
                nifti_image_path=nifti_image_path,
                num_volumes=dimensions[-1],
                mins=mi_min_per_volume,
                ranges=mi_range_per_volume
            )

        average_mean_per_volume: float = statistics.mean(mi_means_per_volume) 
        print(f"Average Mean of Mutual Info per Volume: {average_mean_per_volume}")

        min_num_volumes: int = 11
        threshold_mean: float = average_mean_per_volume  * 2
        while True:
            print(f"Threshold Value: {threshold_mean}")

            running_volumes: list[int] = []
            running_ranges: list[float] = []

            all_passing_volume_lists: list[list[int]]= []
            for volume_num, mi_mean in enumerate(mi_means_per_volume):
                if mi_mean >= threshold_mean:
                    running_volumes.append(volume_num)
                    running_ranges.append(mi_mean)
                else:
                    if running_volumes != []:
                        all_passing_volume_lists.append(running_volumes)
                    running_volumes: list[int] = []
                    running_ranges: list[float] = []
            
                    if running_volumes != []:
                        all_passing_volume_lists.append(running_volumes)

            if all(len(passing_volume_list) < min_num_volumes for passing_volume_list in all_passing_volume_lists): 
                print("Not enough consecutive passing volumes.")
                threshold_mean: float = threshold_mean - (average_mean_per_volume * 0.1)
                print(f"Changing Threshold Value to: {threshold_mean}")
                continue

            largest_passing_list_of_volumes: list[int] = []
            largest_num_volumes: int = 0
            for volume_list in all_passing_volume_lists:
                if len(volume_list) > largest_num_volumes:
                    largest_num_volumes: int = len(volume_list)
                    largest_passing_list_of_volumes: list[int] = volume_list
            print(f"Volumes Selected: {largest_passing_list_of_volumes}")

            self.reference_volume_index: int = int((largest_passing_list_of_volumes[-1] - largest_passing_list_of_volumes[0]) / 2) + largest_passing_list_of_volumes[0]
            print(f"Selected Reference Volume: {self.reference_volume_index}")
            break

        if output_directory:
            self.plot_selected_volume_graph(
                output_directory=output_directory,
                nifti_image_path=nifti_image_path,
                num_volumes=dimensions[-1],
                threshold_mean=threshold_mean,
                average_mean_per_volume=average_mean_per_volume,
                largest_passing_list_of_volumes=largest_passing_list_of_volumes,
                mi_means_per_volume=mi_means_per_volume
            )
            

    def get_slice_timing(self, json_path: str) -> OrderedDict[float, list[int]]:

        def find_matching_indexes(numbers: list[float]) -> dict[float, list[int]]:
            
            num_index_map: dict[float, list[int]] = {}
        
            for index, number in enumerate(numbers):
                if number in num_index_map:
                    num_index_map[number].append(index)
                else:
                    num_index_map[number] = [index]
        
            return {group_num: indexes for group_num, (number, indexes) in enumerate(num_index_map.items()) if len(indexes) > 1}

        slice_timing: list[float] = []
        with open(json_path) as f:
            slice_timing: list[float] = json.load(f)['SliceTiming']
        
        return OrderedDict(sorted(find_matching_indexes(slice_timing).items()))


    def extract_single_volume(self, 
                                volume_num: int, 
                                input_nifti_image_path: str,) -> sitk.Image:
        
            # cannot parallelize if passing a sitk.Image, must pass the path and load image
            input_nifti_image: sitk.Image = sitk.ReadImage(input_nifti_image_path)
            
            timeseries_dimensions: tuple[int, int, int, int] = input_nifti_image.GetSize()
    
            extractor: sitk.ExtractImageFilter = sitk.ExtractImageFilter()
            extractor.SetSize([
                timeseries_dimensions[0],
                timeseries_dimensions[1],
                timeseries_dimensions[2],
                0
            ])
            extractor.SetIndex([0, 0, 0, volume_num])
    
            return extractor.Execute(input_nifti_image)


    def extract_single_slice(self,
                             slice_num: int, 
                             input_nifti_image: sitk.Image) -> sitk.Image:
        
        volume_dimensions: tuple[int, int, int] = input_nifti_image.GetSize()
        
        return sitk.RegionOfInterest(
            input_nifti_image,
            size=[volume_dimensions[0], volume_dimensions[1], 1],
            index=[0, 0, slice_num]
        )

    
    def get_mutual_info(self, 
                        ref_data: np.ndarray, 
                        target_data: np.ndarray, 
                        nbins=64) -> float:
            """
            Function based off of: 
            https://matthew-brett.github.io/teaching/mutual_information.html
            """
    
            hist_2d, _, _ = np.histogram2d(
                ref_data.ravel(),
                target_data.ravel(),
                bins=nbins)
    
            # Convert bin counts in the joint hisogram to probability values
            # by dividing each bin count by the total number of samples 
            prob_xy: np.ndarray = hist_2d / float(np.sum(hist_2d))
    
            # A marginal distribution is the distribution of one variable ignoring the other
            # Compute the marginal for x over y
            prob_x: np.ndarray = np.sum(prob_xy, axis=1)
            # Compute the marginal for y over x
            prob_y: np.ndarray = np.sum(prob_xy, axis=0)
    
            # Compute the product of marginals
            # This is what the joint distribution would be if X and Y were independent.
            px_py: np.ndarray = prob_x[:, None] * prob_y[None, :]
    
            nzs: np.ndarray = prob_xy > 0 # Only non-zero pxy values contribute to the sum
    
            # prob_xy[nzs] -> gets the nonzero joint probabilities 
            # px_py[nzs] -> gets the matching independent joint probabilities
            return np.sum(prob_xy[nzs] * np.log(prob_xy[nzs] / px_py[nzs]))


    def compare_volumes(self, 
                        volume_num1: int, 
                        volume_num2: int,
                        nifti_image_path: str, 
                        slice_timing: OrderedDict[float, list[int]]
                        ) -> dict[int, list[float]]:
        
        volume_1: sitk.Image = self.extract_single_volume(volume_num1, nifti_image_path) 
        volume_2: sitk.Image = self.extract_single_volume(volume_num2, nifti_image_path) 

        all_mutual_info_values: list[float] = []
        for slice_group_num, slice_group_slice_nums in enumerate(slice_timing.values()):

            slice_group_mutual_info_values:  list[float] = []
            for slice_num in slice_group_slice_nums:
                slice_1: sitk.Image = self.extract_single_slice(
                    slice_num=slice_num,
                    input_nifti_image=volume_1
                )
                slice_2: sitk.Image = self.extract_single_slice(
                    slice_num=slice_num,
                    input_nifti_image=volume_2
                )
                mutual_information: float = self.get_mutual_info(
                    ref_data=sitk.GetArrayFromImage(slice_1),
                    target_data=sitk.GetArrayFromImage(slice_2)
                )
                print(f"Vol {volume_num1} vs. Vol {volume_num2} Slice Group {slice_group_num + 1} of {len(slice_timing)}: {mutual_information}")
                slice_group_mutual_info_values.append(mutual_information)
            all_mutual_info_values.append(statistics.mean(slice_group_mutual_info_values))

        return {volume_num1: all_mutual_info_values}


    def plot_all_mutual_info_values(self, 
                                    mutual_info_dict: dict[int, list[float]], 
                                    output_directory: str, 
                                    slice_timing: OrderedDict[float, list[int]], 
                                    num_volumes: int,
                                    nifti_image_path: str):

        output_plot_path: str = os.path.join(output_directory, "mutual_info_plot.html")
        mi_means_per_volume: list[float] = [
            statistics.mean(volume_mi_list)
            for volume_mi_list in list(mutual_info_dict.values())
        ]
        formatted_mi_means_per_volume: list[float] = []
        for mean_val in mi_means_per_volume:
            formatted_mi_means_per_volume.extend([mean_val] * len(slice_timing))

        formatted_volume_nums: list[float] = []
        for volume_num in range(num_volumes - 1):
            formatted_volume_nums.extend([volume_num] * len(slice_timing))

        fig = go.Figure()
        fig.update_layout(
            title=f"All Mutual Information Values: {os.path.basename(nifti_image_path)}",
            xaxis_title="Aquisition Num",
            yaxis_title="Mutual Information"
        )
        fig.add_trace(
            go.Scatter(
                x=list(range((num_volumes - 1) * len(slice_timing))),
                y=[
                    mi_val for volume_mi_list in list(mutual_info_dict.values()) for mi_val in volume_mi_list
                ],
                name="Slice Group Mutual Information",
                line=dict(color='lightblue'),
                customdata=np.column_stack([formatted_volume_nums]),
                hovertemplate=(
                    "<b>Aquisition Number</b>: %{x}<br>"
                    "<b>Mutual Information in Volume</b>: %{y}<br>"
                    "<b>In Volume Number</b>: %{customdata[0]}"
                )
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(range((num_volumes - 1) * len(slice_timing))),
                y=formatted_mi_means_per_volume,
                name="Average Mutual Information Value in Volume",
                line=dict(color='blue'),
                customdata=np.column_stack([formatted_volume_nums]),
                hovertemplate=(
                    "<b>Volume Number</b>: %{customdata[0]}<br>"
                    "<b>Average Mutual Information in Volume</b>: %{y}"
                )
            )
        )
        fig.write_html(output_plot_path)
        print(f"Mutual Info Plot Saved to: {output_plot_path}")


    def plot_min_and_range(self, output_directory: str, nifti_image_path: str, num_volumes: int, mins: list[float], ranges: list[float]):
        # plot 2 subplots with min and range mi per volume 
        output_plot_path: str = os.path.join(output_directory, "min_and_range_mi_vals.html")
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=[
                f"Minimum Mutual Info Value per Volume in {os.path.basename(nifti_image_path)}", 
                f"Range of Mutual Info Values per Volume in {os.path.basename(nifti_image_path)}"],
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        fig.update_layout(showlegend=False)
        fig.add_trace(
            go.Scatter(
                x=list(range(num_volumes - 1)),
                y=mins,
                name="Minimum Mutual Info Values per Volume"
            ),
            row=1,
            col=1
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(num_volumes - 1)),
                y=ranges,
                name="Range of Mutual Info Values per Volume"
            ),
            row=2,
            col=1
        )
        fig.update_xaxes(title_text="Volume Number", row=2)
        fig.update_yaxes(title_text="Mutual Information", row=1, col=1)
        fig.update_yaxes(title_text="Mutual Information", row=2, col=1)
        fig.write_html(output_plot_path)
        print(f"Plot Min and Range of Mutual Info Values per Volume: {output_plot_path}")


    def plot_selected_volume_graph(self, 
                                   output_directory: str,
                                   nifti_image_path: str,
                                   num_volumes: int, 
                                   threshold_mean: float, 
                                   average_mean_per_volume: float, 
                                   largest_passing_list_of_volumes: list[int], 
                                   mi_means_per_volume: list[float]):
        output_plot_path: str = os.path.join(output_directory, "ref-vol-selection-plot.html")
        fig = go.Figure()
        fig.update_layout(
            title=f"All Mutual Info Values per Volume vs. Selected Volume: {os.path.basename(nifti_image_path)}",
            xaxis_title="Volume Number",
            yaxis_title="Mutual Information"
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(num_volumes - 1)),
                y=[threshold_mean] * len(list(range(num_volumes - 1))), 
                name=f"Threshold: {average_mean_per_volume}",
                mode='lines',
                line=dict(color="black", dash="dash")
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(num_volumes - 1)),
                y=mi_means_per_volume,
                name="Mean Mutual Info Value in a Volume",
                mode="lines+markers",
                hoverinfo="skip",
                line=dict(color="gray")
            )
        )
        fig.add_trace(
            go.Scatter(
                x=largest_passing_list_of_volumes,
                y=[mi_means_per_volume[volume_num] for volume_num in largest_passing_list_of_volumes],
                name="Selected Passing Volumes",
                mode="lines+markers",
                line=dict(color="black")
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[self.reference_volume_index],
                y=[mi_means_per_volume[self.reference_volume_index]],
                name=f"Selected Reference Volume: {self.reference_volume_index}",
                mode="lines+markers",
                line=dict(color="red")
                
            )
        )
        fig.write_html(output_plot_path)
        print(f"Reference Volume Selection Plot At: {output_plot_path}")

    def return_selected_reference_volume_index(self) -> int:
        return self.reference_volume_index
    
if __name__ == "__main__":
    import argparse
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description = \
            "Identify a motion-free reference volume via comparing " \
            "the mutual information of slice groups across sequential volumes."
    )
    parser.add_argument(
        "--nifti_image_path",
        required=True,
        help="Must be 4D."
    )
    parser.add_argument(
        "--json_file_path",
        required=True
    )
    parser.add_argument(
        "--n_jobs",
        required=False,
        type=int,
        default=-1,
        help="Default = -1 (All CPU Cores)."
    )
    parser.add_argument(
        "--output_directory",
        required=False,
        default=None,
        help="If None, no output files will be created."
    )
    args: argparse.Namespace = parser.parse_args()
    IdentifyReferenceVolume(
        nifti_image_path=os.path.abspath(args.nifti_image_path),
        json_file_path=os.path.abspath(args.json_file_path),
        output_directory=\
            os.path.abspath(args.output_directory) 
            if args.output_directory else None,
        n_jobs=args.n_jobs
    )