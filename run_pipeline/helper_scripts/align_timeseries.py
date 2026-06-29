import os
import json
import joblib
import argparse
import statistics
import numpy as np
import nibabel as nib
from glob import glob 
import SimpleITK as sitk
from collections import OrderedDict
from matplotlib import pyplot as plt
from scipy.interpolate import NearestNDInterpolator

class AlignTimeseries:
    
    def __init__(self, 
                 nifti_path, 
                 json_path, 
                 transform_directory, 
                 working_directory, 
                 output_directory, 
                 reference_volume_index = 0, 
                 n_jobs = None):
        
        print(f"Input Nifti Image Path: {nifti_path}")
        print(f"Input JSON Path: {json_path}")
        print(f"Transform Directory Path: {transform_directory}")
        print(f"Working Directory: {working_directory}")
        print(f"Output Directory: {output_directory}")
        print(f"Reference Volume Index: {reference_volume_index}")

        os.makedirs(working_directory, exist_ok=True)
        os.makedirs(output_directory, exist_ok=True)

        """
        --------------------------------------------------------------------------
        STEP 1: Get Slice Timing
        --------------------------------------------------------------------------
        """
        slice_timing = self.get_slice_timing(json_path=json_path)
        num_slice_groups = len(set(slice_timing))
        print(f"Slice Timing: {slice_timing}")
        print(f"Number of Slice Groups: {num_slice_groups}")


        """
        --------------------------------------------------------------------------
        STEP 2: Get All Transform Paths
        --------------------------------------------------------------------------
        """
        transforms = [
            transform_path 
            for transform_path in sorted(glob(os.path.join(transform_directory, "*.tfm")))
            if not "identity" in transform_path # remove identity transform from transform list 
        ]
        print(f"I Found {len(transforms)} Transforms in the Transform Directory")
        

        """
        --------------------------------------------------------------------------
        STEP 3: Get Timeseries Dimensions
        --------------------------------------------------------------------------
        """
        dimensions = sitk.ReadImage(nifti_path).GetSize()
        print("Dimensions: " + ', '.join(str(dim_num) for dim_num in dimensions))
        if len(transforms) != num_slice_groups * dimensions[3]:
            print("\nERROR: len(transforms) != num_slice_groups * dimensions[3]")
            exit(0)


        """
        --------------------------------------------------------------------------
        STEP 4: Extract Volumes
        --------------------------------------------------------------------------
        """
        print(f"Extracting Volumes from Input Nifti Image")
        joblib.Parallel(
            n_jobs=n_jobs if n_jobs else min(dimensions[3], os.cpu_count())
            )(
            joblib.delayed(self.extract_volumes)(
                nifti_path=nifti_path, 
                volume_num=volume_index, 
                working_directory=working_directory
            )
            for volume_index in range(dimensions[3])
        )
        volume_paths = sorted(glob(os.path.join(working_directory, f"volume_outputs-*")))


        """
        --------------------------------------------------------------------------
        STEP 5: Get Reference Volume Path
        --------------------------------------------------------------------------
        """
        reference_volume_path = volume_paths[reference_volume_index]
        print(f"Reference Volume Path: {reference_volume_path}")

       
        """
        --------------------------------------------------------------------------
        STEP 6: Extract Slices
        --------------------------------------------------------------------------
        """
        print(f"Extracting Slices from Input Nifti Image")
        for volume_index, volume_path in enumerate(sorted(glob(os.path.join(working_directory, f"volume_outputs-*")))):
            joblib.Parallel(
                n_jobs=n_jobs if n_jobs else min(dimensions[2], os.cpu_count())
                )(
                joblib.delayed(self.extract_slices)(
                    volume_nifti_path=volume_path,
                    volume_num=volume_index,
                    num_volumes=dimensions[3],
                    slice_num=slice_index,
                    working_directory=working_directory
                )
                for slice_index in range(dimensions[2])
            )
        slice_paths = sorted(glob(os.path.join(working_directory, f"slice_outputs-*")))
        print(f"I Found {len(slice_paths)} Slice Paths")


        """
        --------------------------------------------------------------------------
        STEP 7: Apply Transforms to Slice Groups
        --------------------------------------------------------------------------
        """
        joblib.Parallel(
            n_jobs=n_jobs if n_jobs else min(dimensions[2], os.cpu_count())
            )(
            joblib.delayed(self.apply_transform_to_full_slice_group)(
                aquisition_num=aquisition_num,
                transform_path=transform_path,
                num_slice_groups=num_slice_groups,
                slice_timing=slice_timing,
                working_directory=working_directory,
                reference_volume_path=reference_volume_path,
                transforms=transforms

            )
            for aquisition_num, transform_path in enumerate(transforms)
        )


        """
        --------------------------------------------------------------------------
        STEP 8: Calculate and Plot Mutual Information Measurements
        --------------------------------------------------------------------------
        """
        self.run_all_mutual_information_steps(
            working_directory=working_directory,
            output_directory=output_directory,
            num_aquisitions=len(transforms),
            num_slice_groups=num_slice_groups,
            reference_volume_index=reference_volume_index,
            slice_timing=slice_timing,
            n_jobs=n_jobs
        )
        

        """
        --------------------------------------------------------------------------
        STEP 9: Merge 2D Resampled Slices into 3D Resampled Volumes
        --------------------------------------------------------------------------
        """
        joblib.Parallel(
            n_jobs=n_jobs if n_jobs else min(dimensions[2], os.cpu_count())
            )(
            joblib.delayed(self.merge_slices_to_volumes)(
                volume_num=volume_num,
                num_volumes=len(volume_paths),
                volume_path=volume_path,
                working_directory=working_directory

            )
            for volume_num, volume_path in enumerate(volume_paths)
        )


        """
        --------------------------------------------------------------------------
        STEP 10: Merge Resampled 3D Volumes into a 4D Resampled Timeseries
        --------------------------------------------------------------------------
        """
        resampled_time_series_path = os.path.join(output_directory, f"resampled_{os.path.basename(nifti_path)}")
        self.merge_volumes_to_timeseries(
            volume_paths=sorted(glob(os.path.join(working_directory, "resampled_volume_outputs-*"))),
            output_file_path=resampled_time_series_path
        )
        print(f"Resampled Timeseries at: {resampled_time_series_path}")
        

        print("\nDone.\n")


    def get_slice_timing(self, json_path):

        def find_matching_indexes(numbers):
            
            num_index_map = {}
        
            for index, number in enumerate(numbers):
                if number in num_index_map:
                    num_index_map[number].append(index)
                else:
                    num_index_map[number] = [index]
        
            return {group_num: indexes for group_num, (number, indexes) in enumerate(num_index_map.items()) if len(indexes) > 1}

        slice_timing = []
        with open(json_path) as f:
            slice_timing = json.load(f)['SliceTiming']
        
        return OrderedDict(sorted(find_matching_indexes(slice_timing).items()))


    def extract_volumes(self, nifti_path, volume_num, working_directory, output_file_prefix = "volume_outputs"):

        loaded_nifti = sitk.ReadImage(nifti_path)
        volume_size = list(loaded_nifti.GetSize())[:3] + [0]
        print(f"Extracting Volume {volume_num} of {loaded_nifti.GetSize()[3]} Total Volumes")

        extract = sitk.ExtractImageFilter()
        extract.SetSize(volume_size)
        extract.SetIndex((0,0,0,volume_num))
        volume_image = extract.Execute(loaded_nifti)

        output_file = os.path.join(working_directory, f"{output_file_prefix}-{'{:04d}'.format(volume_num)}.nii")
        writer = sitk.ImageFileWriter()
        writer.SetFileName(output_file)
        writer.Execute(volume_image)

        return output_file
        

    def extract_slices(self, volume_nifti_path, volume_num, num_volumes, slice_num, working_directory, output_file_prefix = "slice_outputs"):

        loaded_nifti_volume = sitk.ReadImage(volume_nifti_path)
        slice_size = list(loaded_nifti_volume.GetSize())[:2] + [1]
        print(f"Extracting Slice {slice_num} of {loaded_nifti_volume.GetSize()[2]} from Volume {volume_num} of {num_volumes} Total Volumes")

        slice_volume = sitk.RegionOfInterest(
            loaded_nifti_volume, 
            slice_size, 
            (0, 0, slice_num) 
        )
        
        output_file = os.path.join(working_directory, f"{output_file_prefix}-{'{:04d}'.format(volume_num)}-{'{:03d}'.format(slice_num)}.nii")
        writer = sitk.ImageFileWriter()
        writer.SetFileName(output_file)
        writer.Execute(slice_volume)
        
        return output_file

        
    def invert_transform(self, transform_path, output_transform_path):
        
        inverse_transform = sitk.ReadTransform(transform_path).GetInverse()

        sitk.WriteTransform(
            transform=inverse_transform,
            filename=output_transform_path
        )

    
    def apply_transform_to_single_slice(self, transform_path, slice_path, reference_volume_path, output_slice_path):
        
        reader = sitk.ImageFileReader()
        reader.SetFileName(reference_volume_path)
        refImage = reader.Execute();

        sliceReader = sitk.ImageFileReader()
        sliceReader.SetFileName(slice_path)
        sliceImage = sliceReader.Execute();

        transform = sitk.ReadTransform(transform_path)

        output_pixel_type = sliceReader.GetPixelID()
        resampleFilter = sitk.ResampleImageFilter()
        resampleFilter.SetInterpolator(sitk.sitkLinear)
        resampleFilter.SetTransform(transform)
        resampleFilter.SetOutputPixelType(output_pixel_type)
        resampleFilter.SetDefaultPixelValue(0.0)
        resampleFilter.SetReferenceImage(sliceImage)
        newImage = resampleFilter.Execute(refImage)

        writer = sitk.ImageFileWriter()
        writer.SetFileName(output_slice_path)
        writer.Execute(newImage) 
    

    def apply_transform_to_full_slice_group(self, aquisition_num, transform_path, num_slice_groups, slice_timing, working_directory, reference_volume_path, transforms):
        
        print(f"\nAligning Slice Group for Aquisition {aquisition_num} of {len(transforms)}")
        volume_num = int(aquisition_num / num_slice_groups)

        slice_group_num = aquisition_num - (volume_num * num_slice_groups)

        slice_index_list = slice_timing[slice_group_num]

        print(f"Transform at this Aquisition:\n{transform_path}")
        
        slice_paths = sorted([
            os.path.join(working_directory, f"slice_outputs-{'{:04d}'.format(volume_num)}-{'{:03d}'.format(slice_index)}.nii")
            for slice_index in slice_index_list
        ])
        print("Slice Paths at this Aquisition:")
        print('\n'.join(slice_paths))

        for slice_path in slice_paths:
            
            # 1. INVERT TRANSFORM (so slice goes into reference volume space)
            inverted_transform_path = os.path.join(working_directory, f"inverted_{os.path.basename(transform_path)}")   
            self.invert_transform(
                transform_path=transform_path,
                output_transform_path=inverted_transform_path
            )

            # 2. RESAMPLE SLICE WITH INVERTED TRANSFORM
            resampled_slice_path = os.path.join(working_directory, f"resampled_{os.path.basename(slice_path)}")
            self.apply_transform_to_single_slice(
                transform_path=inverted_transform_path,
                slice_path=slice_path,
                reference_volume_path=reference_volume_path,
                output_slice_path=resampled_slice_path
            )

    
    def merge_slices_to_volumes(self, volume_num, num_volumes, volume_path, working_directory):
        

        ref_img = nib.load(volume_path)
        ref_data = nib.load(volume_path).get_fdata()

        accumulator = np.zeros_like(ref_data, dtype=np.float32)
        weight_map = np.zeros_like(ref_data, dtype=np.float32)

        slices_at_this_volume = sorted(glob(os.path.join(working_directory, f"resampled_slice_outputs-{'{:04d}'.format(volume_num)}-*")))
        print(f"Merging Slices ({len(slices_at_this_volume)} Total) into Volume {volume_num} of {num_volumes}")
        
        for slice_path in slices_at_this_volume:
            
            slice_data = nib.load(slice_path).get_fdata()
            slice_data[slice_data == 0] = np.nan

            mask = ~np.isnan(slice_data)

            # Get the slice index from filename
            slice_idx = int(os.path.basename(slice_path).split('-')[-1].split('.')[0])
            
            # Place 2D slice into correct Z-plane
            accumulator[:, :, slice_idx][mask[:, :, 0]] += slice_data[:, :, 0][mask[:, :, 0]]
            weight_map[:, :, slice_idx][mask[:, :, 0]] += 1.0

        valid_voxels = weight_map > 0

        reconstructed_data = np.full_like(ref_data, fill_value=np.nan, dtype=np.float32)
        reconstructed_data[valid_voxels] = (accumulator[valid_voxels] / weight_map[valid_voxels])

        interp_reconstructed_data = self.nn_interpolation(reconstructed_data, brain_mask=(ref_data != 0))

        reconstructed_img = nib.Nifti1Image(interp_reconstructed_data, ref_img.affine, ref_img.header)
        output_file_path = os.path.join(working_directory, f"resampled_{os.path.basename(volume_path)}.gz")
        nib.save(reconstructed_img, output_file_path)
    

    def nn_interpolation(self, volume_data, brain_mask):

        volume_data_copy = volume_data.copy()
        nan_mask = np.isnan(volume_data_copy)
        
        # Only fill NaNs inside brain mask
        fill_mask = nan_mask & brain_mask
        
        if not fill_mask.any():
            return volume_data_copy

        # coordinates of valid data
        coords =  np.array(np.nonzero(~nan_mask)).T.astype(np.float64)   # shape (N, 3)
        """if coords.shape[0] == 0:
            return np.zeros_like(volume_data_copy, dtype=np.float64)"""
        
        values = volume_data_copy[~nan_mask].astype(np.float64)
        nan_coords = np.array(np.nonzero(fill_mask)).T.astype(np.float64)
        
        """
        # Spline Interpolation 
        interpolator = RBFInterpolator(
            coords,
            values,
            kernel="thin_plate_spline", 
            degree=1,                   
            neighbors=32                 
        )"""

        # Nearest Neighbor Interpolation
        interpolator = NearestNDInterpolator(coords, values)

        # Coordinates to fill
        volume_data_copy[fill_mask] = interpolator(nan_coords)

        return volume_data_copy


    def merge_volumes_to_timeseries(self, volume_paths, output_file_path):
        
        print(f"Merging {len(volume_paths)} Volumes into Time Series")

        volumes = [sitk.ReadImage(volume_path) for volume_path in volume_paths]

        join_array = sitk.JoinSeries(volumes, 0.0, 1.0)

        writer = sitk.ImageFileWriter()
        writer.SetFileName(output_file_path)
        writer.Execute(join_array)


    def compute_mutual_information(self, ref_data, target_data, nbins: int = 32) -> float:
        """
        hist_2d: 2d joint histogram, where:
            - rows = intensity bins of image X
            - columns = intensity bins of image Y
            - values = number of pixel pairs that fall into each bin pair
        """
        hist_2d, x_edges, y_edges = np.histogram2d(
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


    def get_mutual_info_comparisons(self, working_directory, aquisition_num, num_aquisitions, num_slice_groups, reference_volume_index, slice_timing):
        
        print(f"Getting Mutual Information Values for Aquisition {aquisition_num} of {num_aquisitions}")
        
        volume_num = int(aquisition_num / num_slice_groups)
        slice_group_num = aquisition_num - (volume_num * num_slice_groups)
        
        raw_vs_resampled = []
        ref_vs_resampled = []
        ref_vs_raw = []
        for slice_index in slice_timing[slice_group_num]:
            
            raw_slice_path = os.path.join(working_directory, f"slice_outputs-{'{:04d}'.format(volume_num)}-{'{:03d}'.format(slice_index)}.nii")
            resampled_slice_path = os.path.join(working_directory, f"resampled_slice_outputs-{'{:04d}'.format(volume_num)}-{'{:03d}'.format(slice_index)}.nii")
            reference_slice_path = os.path.join(working_directory, f"slice_outputs-{'{:04d}'.format(reference_volume_index)}-{'{:03d}'.format(slice_index)}.nii")

            raw_vs_resampled.append(
                self.compute_mutual_information(
                    ref_data=nib.load(raw_slice_path).get_fdata(),
                    target_data=nib.load(resampled_slice_path).get_fdata(),
                )
            )

            ref_vs_resampled.append(
                self.compute_mutual_information(
                    ref_data=nib.load(reference_slice_path).get_fdata(),
                    target_data=nib.load(resampled_slice_path).get_fdata(),
                )
            )
            ref_vs_raw.append(
                self.compute_mutual_information(
                    ref_data=nib.load(reference_slice_path).get_fdata(),
                    target_data=nib.load(raw_slice_path).get_fdata(),
                )
            )
        
        return [
            (aquisition_num, statistics.mean(raw_vs_resampled)),
            (aquisition_num, statistics.mean(ref_vs_resampled)),
            (aquisition_num, statistics.mean(ref_vs_raw))
        ]


    def write_mutual_information_to_file(self, mutual_info_tups, output_file_path):
        
        with open(output_file_path, mode='w') as f:
            for aquisition_num, mutual_information in mutual_info_tups:
                f.write(f"{aquisition_num},{mutual_information}\n")
        print(f"Mutual Information Values at: {output_file_path}")        
        
         
    def run_all_mutual_information_steps(self, working_directory, output_directory, num_aquisitions, num_slice_groups, reference_volume_index, slice_timing, n_jobs):
        
        print(f"\nCalculating and Plotting Mutual Information Measurements")

        # 1. Calculate mutual information
        results = joblib.Parallel(
            n_jobs=n_jobs if n_jobs else min(num_aquisitions, os.cpu_count())
            )(
            joblib.delayed(self.get_mutual_info_comparisons)(
                working_directory=working_directory,
                aquisition_num=aquisition_num,
                num_aquisitions=num_aquisitions,
                num_slice_groups=num_slice_groups,
                reference_volume_index=reference_volume_index,
                slice_timing=slice_timing
            )
            for aquisition_num in range(num_aquisitions)
        ) 
        raw_vs_resampled_mi = sorted([r[0] for r in results], key=lambda x: x[0])
        ref_vs_resampled_mi = sorted([r[1] for r in results], key=lambda x: x[0])
        ref_vs_raw_mi = sorted([r[2] for r in results], key=lambda x: x[0])

        # 2. Write mutual information values to .txt files 
        self.write_mutual_information_to_file(mutual_info_tups=raw_vs_resampled_mi, output_file_path=os.path.join(output_directory, "raw_vs_resampled_mutual_information.txt"))
        self.write_mutual_information_to_file(mutual_info_tups=ref_vs_resampled_mi, output_file_path=os.path.join(output_directory, "ref_vs_resampled_mutual_information.txt"))
        self.write_mutual_information_to_file(mutual_info_tups=ref_vs_raw_mi, output_file_path=os.path.join(output_directory, "ref_vs_raw_mutual_information.txt"))

        # 3a. Plot 1
        plt.figure(figsize=(14, 8.5))
        plt.title("Mutual Information of Unmoved Slice vs. Reference Slice AND Resampled Slice vs. Reference Slice")
        plt.plot(
            [aquisition_num for aquisition_num, _ in ref_vs_raw_mi],
            [mutual_info_value for _, mutual_info_value in ref_vs_raw_mi],
            label="Unmoved Slice vs. Reference Slice"
        )
        plt.plot(
            [aquisition_num for aquisition_num, _ in ref_vs_resampled_mi],
            [mutual_info_value for _, mutual_info_value in ref_vs_resampled_mi],
            label="Resampled Slice vs. Reference Slice"
        )
        plt.legend()
        plt.grid()
        plt.tight_layout()

        plot_path = os.path.join(output_directory, f"raw_and_resampled_vs_ref_mutual_information_plotted.png")
        plt.savefig(plot_path)
        print(f"Mutual Information of Unmoved Slice vs. Reference Slice AND Resampled Slice vs. Reference Slice Plotted At: {plot_path}")

        plt.close()

        # 3b. Plot 2
        plt.figure(figsize=(14, 8.5))
        plt.title("Mutual Information of Unmoved Slice vs. Resampled Slice")
        plt.plot(
            [aquisition_num for aquisition_num, _ in raw_vs_resampled_mi],
            [mutual_info_value for _, mutual_info_value in raw_vs_resampled_mi]
        )
        plt.grid()
        plt.tight_layout()
        plot_path = os.path.join(output_directory, f"raw_vs_resampled_mutual_information_plotted.png")
        plt.savefig(plot_path)
        print(f"Mutual Information of Unmoved Slice vs. Resampled Slice Plotted at: {plot_path}")
        plt.close()

        
if __name__ == '__main__':

    default_output_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    default_working_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "working")

    parser = argparse.ArgumentParser(description="Align Timeseries using Sms-Mi-Reg Transforms")
    parser.add_argument(
        "--nifti_image_path",
        required=True, 
    )
    parser.add_argument(
        "--json_file_path",
        required=True, 
    )
    parser.add_argument(
        "--transform_directory_path", 
        required=True
    )
    parser.add_argument(
        "--reference_volume_index", 
        required=False,
        default=0,
        type=int,
        help="Index of the chosen 3d reference volume in the timeseries."
    )
    parser.add_argument(
        "--working_directory",
        required=False, 
        default=default_working_directory, 
        help=f"Default: {default_working_directory}"
    )
    parser.add_argument(
        "--output_directory",
        required=False, 
        default=default_output_directory, 
        help=f"Default: {default_output_directory}"
    )
    args = parser.parse_args()

    AlignTimeseries(
        nifti_path=os.path.abspath(args.nifti_image_path),
        json_path=os.path.abspath(args.json_file_path),
        transform_directory=os.path.abspath(args.transform_directory_path),
        working_directory=os.path.abspath(args.working_directory),
        output_directory=os.path.abspath(args.output_directory)
    )
