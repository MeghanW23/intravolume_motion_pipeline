import os 
import json
import warnings
from typing import Any
from collections import OrderedDict

import numpy as np

class GetSliceTiming:

    def __init__(self, 
                 json_data: str | dict[str, Any], # type: ignore
                 output_json_timing_path: str | None = None,
                 output_txt_slice_order_path: str | None = None) -> None:
        
        if isinstance(json_data, str):
            print(f"Loading JSON Data from JSON File: {json_data}")
            with open(json_data, mode='r') as file:
                json_data: dict[str, Any] = json.load(file)
            print("JSON Data is Loaded")
        
        if "SliceTiming" not in json_data:
            raise KeyError("'SliceTiming' not found in JSON data.")

        self.slice_timing: OrderedDict[float, list[int]] = OrderedDict(
            sorted(self.find_matching_indexes(json_data['SliceTiming']).items())
        )

        if output_json_timing_path:
            print(f"Saving Slice Timing to: {output_json_timing_path}")
            self.save_to_json(
                self.slice_timing,
                output_json_timing_path=output_json_timing_path
            )
            print(f"Slice Timing JSON path saved.")

        if output_txt_slice_order_path:
            print(f"Saving Slice Aquisition Timing to: {output_txt_slice_order_path}")
            self.save_to_txt_file(
                self.slice_timing,
                output_txt_slice_order_path=output_txt_slice_order_path
            )
            print(f"Slice Timing Text File path saved.")

    def find_matching_indexes(self, numbers: list[float]) -> dict[float, list[int]]:
        
        num_index_map: dict[float, list[int]] = {}
    
        for index, number in enumerate(numbers):
            if number in num_index_map:
                num_index_map[number].append(index)
            else:
                num_index_map[number] = [index]
        
        if len(num_index_map) == len(numbers):
            warnings.warn(
                message=(
                    "Could not find matching slice timing values for any slices "
                    "in your JSON file. This pipeline is designed for SMS-Accelerated "
                    "fMRI data. Is your data SMS-Accelerated?"
                ),
                category=UserWarning
            )
            
            return {
                number: indexes 
                for number, indexes 
                in num_index_map.items()
            }
        else:
            return {
                number: indexes 
                for number, indexes 
                in num_index_map.items() 
                if len(indexes) > 1
            }


    def return_slice_timing(self) -> OrderedDict[float, list[int]]:
        return self.slice_timing



    def save_to_json(self,
                     slice_timing_data: OrderedDict[float, list[int]], 
                     output_json_timing_path: str):
        with open(output_json_timing_path, mode='w') as file:
            json.dump(
                slice_timing_data,
                fp=file
            )

    def save_to_txt_file(self,
                         slice_timing_data: OrderedDict[float, list[int]], 
                         output_txt_slice_order_path: str):
        sms_factor: int = len(list(slice_timing_data.values())[0])
        with open(output_txt_slice_order_path, mode='w') as file:
            for slice_group_indices in list(slice_timing_data.values()):
                for slice_num in slice_group_indices:
                    file.write(f"{str(slice_num)}\n")


    def print_slice_timing(self): 
        for slice_group_num, (slice_group_time, slice_nums) in \
        enumerate(self.slice_timing.items()): 
            print("\n---------")
            print(f"Slice Group {'{:02d}'.format(slice_group_num + 1)} of {'{:02d}'.format(len(self.slice_timing))}:")
            print(f"Time of Aquistion (from TR Start): {slice_group_time}s")
            print(f"Indices of Slices In Slice Group: {slice_nums}")


    def get_num_slice_groups(self) -> int:
        return len(self.slice_timing)
    
    
if __name__ == "__main__":
    import argparse
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Get the Slice Timing for SMS-Accelerated fMRI Data."
    )
    parser.add_argument("--json_file_path", required=True)
    parser.add_argument("--output_json_timing_path", required=False, default=None,
                        help=\
                            "Add a path if you'd like to save the slice timing" \
                            " to a JSON File. Else, no files will be created." \
                            " Default = None")
    parser.add_argument("--output_txt_slice_order_path", required=False, default=None,
                        help=\
                            "Add a path if you'd like to save the slice order times" \
                            " to a .txt file. Else, no files will be created. " \
                            " Default = None")
    args: argparse.Namespace = parser.parse_args()
    GetSliceTiming(
        json_data=os.path.abspath(args.json_file_path),
        output_json_timing_path=os.path.abspath(args.output_json_timing_path) 
                        if args.output_json_timing_path else
                        None,
        output_txt_slice_order_path=os.path.abspath(args.output_txt_slice_order_path) 
                        if args.output_txt_slice_order_path else
                        None
    ).print_slice_timing()