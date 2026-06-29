import os
import statistics 
import numpy as np
import SimpleITK as sitk
from scipy.optimize import minimize

class ParameterSearch:
    
    def __init__(self, 
                 reference_volume, 
                 slices, 
                 initial_versor_parameters, 
                 fixed_parameters,
                 output_transform_filename,
                 output_directory,
                 max_iterations = 200,
                 verbose=False):
        self.verbose = verbose
        os.makedirs(output_directory, exist_ok=True)

        # Load data as SimpleITK Images if not already loaded 
        if isinstance(reference_volume, str):
            reference_volume = sitk.ReadImage(reference_volume)
        elif not isinstance(reference_volume, sitk.Image):
            print("arg: 'reference_volume' must be a SimpleITK Image or an Existing NiFTI File Path")
            exit(0)
        
        if all(isinstance(slice, str) for slice in slices):
            slices = [sitk.ReadImage(slice_path) for slice_path in sorted(slices)]
        elif not all(isinstance(slice, sitk.Image) for slice in slices):
            print("arg: 'slices' must be a list of SimpleITK Images or a list of Existing NiFTI File Paths")
            exit(0)
        
        # Run optimization, minimize the cost function (the negative mutual information value)
        result = minimize(
            fun=self.objective,
            x0=initial_versor_parameters,
            args=(fixed_parameters, reference_volume, slices),
            method='Powell',
            options={'maxiter': max_iterations, 'disp': True}   
        )

        # Create output transform
        self.found_parameters = list(result.x)
        self.output_transform_path = os.path.join(output_directory, output_transform_filename)
        sitk.WriteTransform(
            transform=self.create_transform(
                versor_parameters=self.found_parameters,
                fixed_parameters=fixed_parameters),
            filename=self.output_transform_path
        )


    def objective(self, versor_parameters, fixed_parameters, reference_volume, slices):
        
        # 1. Create the transform with the parameters we will test
        transform = self.create_transform(
            versor_parameters=versor_parameters,
            fixed_parameters=fixed_parameters
        )

        mutual_information_values_at_slice_group = []
        for slice in slices:
            
            # 2. Apply the transform to the slices
            resampled_ref_to_moving_slice = self.apply_transform(
                transform=transform,
                reference_volume=reference_volume,
                slice=slice
            )

            # 3. Evaluate the parameters by getting the mutual information 
            # between the resampled slice and the target slice
            mutual_information_values_at_slice_group.append(
                self.get_mutual_information(
                    ref_data=sitk.GetArrayFromImage(resampled_ref_to_moving_slice),
                    target_data=sitk.GetArrayFromImage(slice)
                )
            )
        
        # 4. Mean across slices in the slice group
        mutual_information = statistics.mean(mutual_information_values_at_slice_group) 

        if self.verbose:
            print(f"Mutual Information: {mutual_information}\t\tParameters: {', '.join([str(round(parameter, 4)) for parameter in list(versor_parameters)])}")

        # 5. Return negative of mutual information (our optimizer is looking for the minimum of the cost function)
        return -mutual_information


    def create_transform(self, versor_parameters, fixed_parameters):
        
        axis = [versor_parameters[0], versor_parameters[1], versor_parameters[2]]
        angle = np.linalg.norm(axis)

        transform = sitk.VersorRigid3DTransform()
        # No rotation
        if angle == 0:
            transform.SetRotation([1, 0, 0], 0.0)  # arbitrary axis, zero angle
        else:
            axis = [x / angle for x in axis]
            transform.SetRotation(axis, angle)

        transform.SetTranslation([versor_parameters[3], versor_parameters[4], versor_parameters[5]])
        transform.SetCenter(fixed_parameters) 

        return transform


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


    def get_output_parameters(self):
        return self.found_parameters
    

    def get_output_transform_path(self):
        return self.output_transform_path
        

if __name__ == '__main__':
    import argparse

    default_output_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

    parser = argparse.ArgumentParser(
        description="Motion Characterization of a Slice Group from a Reference Volume via Maximization of Mutual Information"
    )
    parser.add_argument(
        "--reference_volume_path", 
        required=True
    )
    parser.add_argument(
        "--slice_group_paths", 
        required=True,
          nargs='+'
    )
    parser.add_argument(
        "--initial_versor_parameters",
        required=True,
        nargs='+',
        type=float,
        help="VERSOR PARAMETERS ONLY"
    )
    parser.add_argument(
        "--fixed_parameters",
        required=True,
        nargs='+',
        type=float
    )
    parser.add_argument(
        "--output_transform_filename", 
        required=True
    )
    parser.add_argument(
        "--output_directory", 
        required=False,
        default=default_output_directory,
        help=f"Default: {default_output_directory}"
    )
    args = parser.parse_args()

    ParameterSearch(
        reference_volume=os.path.abspath(args.reference_volume_path),
        slices=[
            os.path.abspath(slice_path)
            for slice_path in args.slice_group_paths
        ],
        initial_versor_parameters=args.initial_versor_parameters,
        fixed_parameters=args.fixed_parameters,
        output_directory=os.path.abspath(args.output_directory),
        output_transform_filename=args.output_transform_filename
    )