import os 
import sys
import json
from typing import Any
import yaml
from glob import glob
# add script directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))
from manage_configuration_files import Configurations

class PostPipelineGraphing:
    def __init__(self, configuration_file: str) -> None:
        if not os.path.exists(configuration_file):
            raise FileNotFoundError(
                f"Could not find configuration file: {configuration_file}"
            )
        """
        ========================================
        LOAD CONFIGURATION FILE 
        ========================================
        """
        print(f"Loading Input Configuration File: {configuration_file}")
        configurations = None
        with open(configuration_file, mode='r') as file:
            configurations = Configurations(**yaml.safe_load(file))

        raw_nifti_path, raw_json_path = self.find_raw_func_data(configurations.OUTPUT_DIRECTORY_PATH)
        
        

    def find_raw_func_data(self, output_directory: str) -> tuple[str, str]:
        print("Looking for raw func JSON File...")

        json_file: str = ""
        nifti_file: str =""

        all_json_files: list[str] = glob(os.path.join(output_directory, "*.json"))
        for json_file in all_json_files:
            if "slice_timing" in os.path.basename(json_file):
                continue 
            else:
                with open(json_file, mode='r') as file:
                    data: dict[str, Any] = json.load(file)
                    if not 'ConversionSoftware' in data: 
                        continue 
                    elif data['ConversionSoftware'] != 'dcm2niix':
                        continue 
                    else:
                        print(f"Found Raw JSON File: {json_file}")
        nifti_file: str = json_file.replace(".json", ".nii.gz")
        if not os.path.exists(nifti_file):
            raise FileNotFoundError(f"Could not find Raw NiFTI File at: {nifti_file}")
        else:
            print(f"Found Raw NiFTI File: {nifti_file}")
        return nifti_file, json_file
        

if __name__ == "__main__":
    import argparse
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description = \
            "Wrapper for running post-pipeline graphing and analysis script. " \
            "NOTE: This script will look for each file as it was written by the pipeline. "
    )
    parser.add_argument(
        "--configuration_file",
        required=True,
        help="The .yaml file used to run the pipeline"
    )
    args: argparse.Namespace = parser.parse_args()
    PostPipelineGraphing(os.path.abspath(args.configuration_file))
