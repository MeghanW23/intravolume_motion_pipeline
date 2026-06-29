import os
import statistics
import numpy as np
import SimpleITK as sitk 
from itertools import product

class ExhaustiveSearch:

    def __init__(self, 
                 slice_group, 
                 reference_volume, 
                 output_directory,
                 fixed_parameters = None, 
                 num_steps_per_dim = 5,
                 rotation_step_size = 0.0174533, # radians
                 translation_step_size = 0.5, # milimeters
                 output_files_prefix = "mutual_info",
                 verbose = True
                 ):
        os.makedirs(output_directory, exist_ok=True)

        if isinstance(reference_volume, str):
            reference_volume = sitk.ReadImage(reference_volume)
        
        if any(isinstance(input_slice, str) for input_slice in slice_group):
            slice_group = [sitk.ReadImage(input_slice) for input_slice in slice_group]
        
        if not fixed_parameters:
            fixed_parameters = self.get_fixed_parameters(reference_volume)

        rot_steps  = np.arange(-num_steps_per_dim // 2, num_steps_per_dim // 2) * rotation_step_size
        trans_steps = np.arange(-num_steps_per_dim // 2, num_steps_per_dim // 2) * translation_step_size

        self.param_grid = [
            rot_steps,   # X Rotation
            rot_steps,   # Y Rotation
            rot_steps,   # Z Rotation
            trans_steps, # X Translation
            trans_steps, # Y Translation
            trans_steps, # Z Translation
        ]

        mutual_info_results = {}
        num_steps_total = num_steps_per_dim**len(self.param_grid)
        for i, param in enumerate(product(*self.param_grid)):
            if verbose:
                print(f"Slice Group {output_files_prefix}:\tStep {i} of {num_steps_total} - Parameters: {[str(param_val) for param_val in param]}")
                print(f"Slice Group {output_files_prefix}:\t{round((i/num_steps_total) * 100, 2)}% Done")
            transform = sitk.Euler3DTransform()
            transform.SetParameters(param)
            transform.SetFixedParameters(fixed_parameters + [0])

            mutual_information = statistics.mean([
                self.get_mutual_information(
                    ref_data=sitk.GetArrayFromImage(
                        self.apply_transform(
                            transform=transform,
                            reference_volume=reference_volume,
                            slice=input_slice
                        )
                    ),
                    target_data=sitk.GetArrayFromImage(input_slice)
                )
                for input_slice in slice_group
            ])
            if verbose:
                print(f"Slice Group {output_files_prefix}:\tMutual Information: {mutual_information}")

            mutual_info_results[param] = mutual_information
        
        with open(os.path.join(output_directory, f"{output_files_prefix}.txt"), mode='w') as file:
            for param, mutual_info in mutual_info_results.items():
                file.write(f"{', '.join([str(param_val) for param_val in param])} : {mutual_info}" + '\n')
        
        self.parameters_with_max_mutual_information, self.max_mutual_information = sorted([(parameter_list, mi) for parameter_list, mi in mutual_info_results.items()], key= lambda x: x[1])[-1]
        if verbose:
            print(f"Slice Group {output_files_prefix}:\tParameters with Maximum Mutual Information: {', '.join([str(parameter_value) for parameter_value in self.parameters_with_max_mutual_information])}")
            print(f"Slice Group {output_files_prefix}:\tMaximum Mutual Information: {self.max_mutual_information}")
    
    
    def get_fixed_parameters(self, volume):
        
        return list(volume.TransformContinuousIndexToPhysicalPoint([(index-1)/2.0 for index in volume.GetSize()]))         

    
    def apply_transform(self, transform, reference_volume, slice):

        output_pixel_type = slice.GetPixelID()
        resampleFilter = sitk.ResampleImageFilter()
        resampleFilter.SetInterpolator(sitk.sitkLinear)
        resampleFilter.SetTransform(transform)
        resampleFilter.SetOutputPixelType(output_pixel_type)
        resampleFilter.SetDefaultPixelValue(0.0)
        resampleFilter.SetReferenceImage(slice)
        
        return resampleFilter.Execute(reference_volume)


    def get_mutual_information(self, ref_data, target_data, nbins=64):
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


    def return_best_parameters(self):
        return self.parameters_with_max_mutual_information


if __name__ == '__main__':
    
    import argparse

    default_output_directory = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Motion Characterization of an fMRI Slice Group using an Exhaustive 6-Dimensional Parameter Search")
    parser.add_argument("--slice_group_paths", required=True, nargs='+')
    parser.add_argument("--reference_volume_path", required=True)
    parser.add_argument("--output_directory", required=False, default=default_output_directory, help=f"Default: {default_output_directory}")
    args = parser.parse_args()

    ExhaustiveSearch(
        slice_group=[os.path.abspath(slice_path) for slice_path in args.slice_group_paths],
        reference_volume=os.path.abspath(args.reference_volume_path),
        output_directory=os.path.abspath(args.output_directory)
    )
