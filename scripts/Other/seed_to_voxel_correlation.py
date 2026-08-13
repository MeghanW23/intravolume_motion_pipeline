import os
import json
import numpy as np
import nibabel as nib
from typing import Any
from nilearn.maskers import NiftiSpheresMasker
from nilearn.interfaces.fmriprep import load_confounds

class SeedToVoxelCorrelation:
    def __init__(self, 
                 nifti_image_path: str, 
                 json_file_path: str,
                 seed_coords: tuple[int, int, int] = (0, -52, 18),
                 seed_radius: float = 8) -> None:
        """
        Producing single subject maps of seed-to-voxel correlation
        Implements: https://nilearn.github.io/dev/auto_examples/03_connectivity/plot_seed_to_voxel_correlation.html 
        """

        print(f"Input NiFTI Image Path: {nifti_image_path}")
        print(f"Input JSON File Path: {json_file_path}")
        print(f"ROI Seed Coordinates: {seed_coords}")
        print(f"ROI Seed Radius: {seed_radius} mm")


        print(f"Loading NiFTI Image: {os.path.basename(nifti_image_path)}")
        img: nib.Nifti1Image = nib.load(nifti_image_path) # pyright: ignore[reportAssignmentType, reportPrivateImportUsage]
        img_data: np.ndarray = img.get_fdata()
        img_dimensions: list[float] = list(img_data.shape)
        print(f"Image Dimensions: {img_dimensions}")
        if len(img_dimensions) != 4:
            raise ValueError("Input NiFTI Image Must be 4D.")


        t_r: float = self.get_repetition_time(json_file_path)
        print(f"Repetition Time: {t_r}s")


        print("Loading Confounds")
        confounds, sample_mask = load_confounds(
            img_files=nifti_image_path,
            strategy=["high_pass", "motion", "wm_csf"],
            motion="basic",
            wm_csf="basic",
        )
        

        print("Extracting the time series from the functional imaging within the sphere")
        seed_masker: NiftiSpheresMasker = NiftiSpheresMasker(
            seeds=[seed_coords],

            # Indicates, in millimeters, the radius for the sphere around the seed.
            radius=seed_radius,

            detrend=True,
            standardize_confounds=True,
            low_pass=0.1,
            high_pass=0.01,
            t_r=t_r,
            memory="nilearn_cache",
            memory_level=1,
            verbose=1
        )


        print("Extracting the mean time series within seed region (while regressing out confounds).")
        seed_time_series: np.ndarray = seed_masker.fit_transform(
            img, 
            confounds=[confounds]
        )

    def get_repetition_time(self, json_file_path: str) -> float:
        with open(json_file_path, mode='r') as file:
            data: dict[str, Any] = json.load(file)
            if not 'RepetitionTime' in data:
                raise KeyError(
                    f"Could not find key: 'RepetitionTime' in your JSON File: {json_file_path}"
                )
            return float(data['RepetitionTime'])
        
         

if __name__ == "__main__":
    import argparse
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Produce seed-to-voxel correlation maps for a single subject, single run fMRI NiFTI Timeseries."
    )
    parser.add_argument(
        "--nifti_image_path",
        required=True,
        help="Must be 4D and must be post-fMRIPrep."
    )
    parser.add_argument(
        "--json_file_path",
        required=True,
        help="Must have key: 'RepetitionTime'"
    )
    parser.add_argument(
        "--seed_coords",
        required=False,
        type=int,
        nargs=3,
        default=(0, -52, 18),
        help=\
            "Input 3 coordinates to your seed. Default is (0, -52, 18). " \
            "These are in the Posterior Cingulate Cortex (PCC), " \
            "considered part of the Default Mode Network."
    )
    parser.add_argument(
        "--seed_radius",
        required=False,
        type=float,
        default=8,
        help="Indicates, in millimeters, the radius for the sphere around the seed. Default: 8mm"
    )
    args: argparse.Namespace = parser.parse_args()
    SeedToVoxelCorrelation(
        nifti_image_path=os.path.abspath(args.nifti_image_path),
        json_file_path=os.path.abspath(args.json_file_path),
        seed_coords=tuple(args.seed_coords),
        seed_radius=args.seed_radius
    )


