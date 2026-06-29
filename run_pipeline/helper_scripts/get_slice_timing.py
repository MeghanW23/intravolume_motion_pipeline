
import argparse
from collections import OrderedDict
import json
import os

class GetSliceTiming():
    
    def __init__(self, json_path):
        
        slice_timing = []
        with open(json_path) as f:
            slice_timing = json.load(f)['SliceTiming']

        slice_timing = OrderedDict(sorted(self.find_matching_indexes(slice_timing).items()))
        print("Slice Timing:")
        for i, key in enumerate(slice_timing):
            print(f"Slice Group {i}:")
            print(f"\tTime Of Aquisition (Since Aquisition Start): {key}")
            print(f"\tSlices In Group: {slice_timing[key]}")
    

    def find_matching_indexes(self, numbers):

        num_index_map = {}

        for index, number in enumerate(numbers):
            if number in num_index_map:
                num_index_map[number].append(index)
            else:
                num_index_map[number] = [index]

        return {number: indexes for number, indexes in num_index_map.items() if len(indexes) > 1}

        
parser = argparse.ArgumentParser(description="To get the slice timing from a JSON file.")
parser.add_argument("--json_filepath", required=True)
args = parser.parse_args()

GetSliceTiming(json_path=os.path.abspath(args.json_filepath))