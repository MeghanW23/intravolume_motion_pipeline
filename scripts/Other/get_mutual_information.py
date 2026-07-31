import argparse
import math
import os 
import statistics
import subprocess
import shutil
import numpy as np 
import SimpleITK as sitk 
import pandas as pd 
from glob import glob 
import matplotlib.pyplot as plt 


class GetMutualInformation:
    def __init__(self, 
                 slice_paths: list[str],
                 reference_volume_path: str,
                 working_dir: str,
                 global_optima: list[float],
                 num_steps: int = 10,
                 rotation_step_size: float = 0.0174533 / 3,
                 translation_step_size: float = 0.3,
                 save_working_files: bool = False):
        
        # Path Setup 
        self.working_dir: str = working_dir
        os.makedirs(self.working_dir, exist_ok=True)

        self.slice_paths: list[str] = slice_paths
        self.reference_volume_path: str = reference_volume_path
        
        # Configurations 
        self.global_optima: list[float] = global_optima
        self.rotation_step_size: float = rotation_step_size # rad (0.0174533 rad = 1 degree)
        self.translation_step_size: float = translation_step_size # mm
        self.search_dimensions: list[str] = [
            "X_Rotation",
            "Y_Rotation",
            "Z_Rotation",
            "X_Translation",
            "Y_Translation",
            "Z_Translation"
        ]

        print(f"Copying Input Files into the Working Directory...")
        for file in self.slice_paths + [self.reference_volume_path]:
            if file != None:
                shutil.copy(
                    src=file,
                    dst=os.path.join(working_dir, os.path.basename(file))
                )

        self.fixed_parameters: list[float] = self._get_fixed_parameters()
        print(f"Fixed Parameters: {self.fixed_parameters}")

        self.mutual_information_values: dict[str, dict[int, float]] = {} # dict[step, mi_value]

        for dimension_index, dimension in enumerate(self.search_dimensions):
            print(f"Starting Dimension: {dimension}")

            step_size: float = None 
            if "Rotation" in dimension:
                step_size: float = self.rotation_step_size
            elif "Translation" in dimension:
                step_size: float = self.translation_step_size

            self.mutual_information_values[dimension] = {}

            print(f"Dimension Step Size: {step_size}")

            for step in range(-num_steps + 1, num_steps + 1):
                net_step: int = step + num_steps

                parameter_value: float = step * step_size
                
                output_label: str = f"step-{'{:04d}'.format(net_step)}_dim-{'{:04d}'.format(dimension_index)}-alignment"

                transform_path: str = self._make_transform(dimension_index=dimension_index,
                                                           parameter_value=parameter_value,
                                                           output_label=output_label)
                print(f"Created Transform: {transform_path}")

                slice_group_mis = []
                for slice_path in self.slice_paths:
                    
                    resampled_ref_to_moving_slice = self.apply_transform(
                        transform=sitk.ReadTransform(transform_path),
                        reference_volume=sitk.ReadImage(reference_volume_path),
                        slice=sitk.ReadImage(slice_path)
                    )
                    slice_group_mis.append(
                        self._get_mutual_info(
                            ref_data=sitk.GetArrayFromImage(resampled_ref_to_moving_slice),
                            target_data=sitk.GetArrayFromImage(sitk.ReadImage(slice_path)))
                        )

                mutual_info = statistics.mean(slice_group_mis)
                print(f"Mutual Info: {mutual_info}")

                self.mutual_information_values[dimension][step_size * step] = mutual_info

                if not save_working_files:
                    self._clear_files(output_label=output_label)

    
    def _clear_files(self, 
                     output_label: str): 
        
        for file in glob(os.path.join(self.working_dir, f"*{output_label}*")):
            os.remove(file)


    def _run_command(self,
                     command: list[str]): 

        print(f"Running Command:\n{command}")
        result: subprocess.CompletedProcess = subprocess.run(
            command, 
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Command:\n{command}\nreturned non-zero return code: {result.returncode}")
            print(f"Stdout:\n{result.stdout}")
            print(f"Stderr:\n{result.stderr}")
            os._exit(1)

        else:
            print("Command Ran Sucessfully")
            print(f"Stdout:\n{result.stdout}")


    def _get_fixed_parameters(self) -> list[float]:
        self._run_command([
            "docker", "run", "--rm",
            "-v", f"{self.working_dir}:/data",
            "crl/sms-mi-reg",
            "crl-identity-transform-at-volume-center.py",
            "--refvolume", os.path.basename(self.reference_volume_path),
            "--transformfile", "identity-centered.tfm"
        ])

        with open(file=os.path.join(self.working_dir, "identity-centered.tfm"), mode='r') as f:
            for line in f: 
                if "FixedParameters" in line:
                    return [float(val) for val in line.split(" ")[1:]] 


    def _make_transform(self, 
                        dimension_index: int, 
                        parameter_value: float, 
                        output_label: str) -> str:
        
        global_max_param: float = self.global_optima[dimension_index]
        parameter_value: float = global_max_param + parameter_value

        transform: sitk.VersorRigid3DTransform = sitk.VersorRigid3DTransform()
        transform.SetIdentity() 

        transform.SetFixedParameters(self.fixed_parameters)

        # set rotation 
        if dimension_index < 3:

            axis: list[float] = list(np.zeros(3))
            axis[dimension_index] = 1

            transform.SetRotation(axis, parameter_value)
        
        # set translation
        else:
            translation_params: list[float] = list(np.zeros(3))
            translation_params[dimension_index - 3] = parameter_value

            transform.SetTranslation(translation_params)
        
        filepath: str = os.path.join(self.working_dir, f"INPUT_{output_label}.tfm")


        sitk.WriteTransform(
            transform=transform,
            filename=filepath
        )

        return filepath

    def apply_transform(self, transform, reference_volume, slice):

        output_pixel_type = slice.GetPixelID()
        resampleFilter = sitk.ResampleImageFilter()
        resampleFilter.SetInterpolator(sitk.sitkLinear)
        resampleFilter.SetTransform(transform)
        resampleFilter.SetOutputPixelType(output_pixel_type)
        resampleFilter.SetDefaultPixelValue(0.0)
        resampleFilter.SetReferenceImage(slice)
        
        return resampleFilter.Execute(reference_volume)

    def _get_mutual_info_smsmireg(self,  output_label: str, transform_path: str):

        """
        Usage: sms-mi-reg [--help] [--version] [--optimizer VAR] [--maxiter VAR] referenceVolume inputTransform outputTransformLabel inputSlices

        Positional arguments:
        referenceVolume       The volume that is moving to be aligned to the slices.
        inputTransform        The transform to initialize the alignment.
        outputTransformLabel  Name phrase used in the construction of the output transform file name.
        inputSlices           The list of file names of the fixed target slices. [nargs: 1 or more]

        Optional arguments:
        -h, --help            shows help message and exits
        -v, --version         prints version information and exits
        --optimizer           Choice of optimizer (LN_COBYLA, LN_BOBYQA, LN_NELDERMEAD, LN_SBPLX). Default is LN_BOBYQA. [default: "LN_BOBYQA"]
        --maxiter             Maximum number of optimizer iterations. Default is 1000. [default: 1000]
        """

        self._run_command(
            ["docker", "run", "--rm", 
             "-v", f"{self.working_dir}:/data",
             "crl/sms-mi-reg",
             "sms-mi-reg",
             os.path.basename(self.reference_volume_path),
             os.path.basename(transform_path),
             output_label] +
             [os.path.basename(slice_path) for slice_path in self.slice_paths] +
             ["--maxiter", "1"]
        )

        regTrace_filepath: str = os.path.join(self.working_dir, f"regTrace_{output_label}.csv")

        return float(pd.read_csv(regTrace_filepath)['MutualInfo'][0])
    
    def _get_mutual_info(self, ref_data, target_data, nbins=64):
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

    def export_to_csv(self) -> str:

        output_dir: str = os.path.join(self.working_dir, "output_csvs")
        os.makedirs(output_dir, exist_ok=True)

        for dimension in self.mutual_information_values:
            csv_path: str = os.path.join(output_dir, f"{dimension.lower()}.csv")

            with open(file=csv_path, mode='w') as f:
                f.write("parameter,mutual_info,\n")

                for parameter, mutual_infomation in self.mutual_information_values[dimension].items():
                    f.write(f"{parameter},{mutual_infomation},\n")

        print(f"Find CSVs at: {output_dir}")


    def graph_mutual_info(self, 
                          figure_title: str = "Mutual Information Graph",
                          output_filepath: str = None, 
                          show_graph: bool = False, 
                          fig_width: float = 14, 
                          fig_height: float = 8.5):

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(fig_width, fig_height))
        fig.suptitle(figure_title, fontweight='bold')
        fig.supylabel("Mutual Information")

        for i, dimension in enumerate(self.mutual_information_values):

            row: int = 0 if i < 3 else 1
            column: int = i if i < 3 else i - 3
            
            axes[row, column].set_title(dimension.replace("_", " "))
            axes[row, column].plot(
                self.mutual_information_values[dimension].keys(),
                self.mutual_information_values[dimension].values(),
                linewidth=0.5
            )
            axes[row, column].scatter(
                self.mutual_information_values[dimension].keys(),
                self.mutual_information_values[dimension].values(),
                s=2
            )
            axes[row, column].set_xlabel("Radians" if "Rotation" in dimension else "Millimeters")
            axes[row, column].grid(True)
        
        plt.tight_layout()
        
        if output_filepath:
            plt.savefig(output_filepath)
       
        if show_graph:
            plt.show() 

        plt.close()
    

if __name__ == '__main__':
    """
    python scripts/get_mutual_information.py \
        --slice_paths working/target-slice_outputs-012.nii working/target-slice_outputs-027.nii working/target-slice_outputs-042.nii working/target-slice_outputs-057.nii \
        --reference_volume_path UPSAMPLED_volume_outputs-0000.nii \
        --working_dir mutual_info_test_working_UPSAMPLED_REF_VOL
    """

    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--slice_paths", 
                        nargs="+",
                        required = True)
    parser.add_argument("--reference_volume_path", 
                        required = True)
    parser.add_argument("--working_dir", 
                        required = False, 
                        default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working_dir"))
    parser.add_argument("--global_optima",
                        required=False,
                        type=float,
                        nargs=6,
                        default=[0, 0, 0, 0, 0, 0])
    args: argparse.Namespace = parser.parse_args()


    mi_maker: GetMutualInformation = GetMutualInformation(
        slice_paths=[os.path.abspath(path) for path in args.slice_paths],
        reference_volume_path=os.path.abspath(args.reference_volume_path),
        working_dir=os.path.abspath(args.working_dir),
        global_optima=args.global_optima
    )
    mi_maker.export_to_csv()
    mi_maker.graph_mutual_info(show_graph=True, output_filepath="mi_graph.png")