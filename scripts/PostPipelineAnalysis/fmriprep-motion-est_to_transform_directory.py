import os 
import numpy as np
import pandas as pd 
import SimpleITK as sitk


class ConvertfMRIPrepTsvToTransformDirectory:
    def __init__(self,
                 fmriprep_confounds_file: str,
                 fixed_parameters: list[float],
                 output_directory: str = 'fmriprep_transforms') -> None:

        os.makedirs(output_directory, exist_ok=True)
        
        confounds_df: pd.DataFrame = pd.read_csv(fmriprep_confounds_file, sep='\t')

        for frame_num in range(len(confounds_df['framewise_displacement'])):

            # Make Euler3DTransform
            euler_transform: sitk.Euler3DTransform = sitk.Euler3DTransform()
            euler_transform.SetFixedParameters(fixed_parameters + [0])
            euler_transform.SetParameters([
                confounds_df['rot_x'][frame_num],
                confounds_df['rot_y'][frame_num],
                confounds_df['rot_z'][frame_num],
                confounds_df['trans_x'][frame_num],
                confounds_df['trans_y'][frame_num],
                confounds_df['trans_z'][frame_num]
            ])

            # Convert to Versor3D Transform
            matrix: np.ndarray = euler_transform.GetMatrix()
            translation: np.ndarray = euler_transform.GetTranslation()
            center: np.ndarray= euler_transform.GetCenter()
            versor_transform: sitk.VersorRigid3DTransform = sitk.VersorRigid3DTransform()
            versor_transform.SetFixedParameters(center)
            versor_transform.SetTranslation(translation)
            versor_transform.SetMatrix(matrix)

            sitk.WriteTransform(
                versor_transform,
                os.path.join(output_directory, f"transform-{'{:04d}'.format(frame_num)}.tfm")
            )

if __name__ == "__main__":
    import argparse
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=\
            "Convert a fMRIPrep desc-confounds_timeseries.tsv file's motion estimates " \
            "to a directory of SimpleITK Versor3DTransforms."
    )
    parser.add_argument(
        "--fmriprep_confounds_file",
        required=True,
        help="the fMRIPrep desc-confounds_timeseries.tsv file."
    )
    parser.add_argument(
        "--output_directory",
        required=False,
        default="fmriprep_transforms",
        help=f"Default: {os.path.abspath('fmriprep_transforms')}"
    )
    parser.add_argument(
        "--fixed_parameters",
        required=True,
        nargs=3,
        type=float,
        help=\
            "The fixed center of rotation. See any transforms outputted by our motion correction software, " \
            "or run the identity transform creation script to get the fixed parameters. "
    )
    args: argparse.Namespace = parser.parse_args()
    ConvertfMRIPrepTsvToTransformDirectory(
        fmriprep_confounds_file=os.path.abspath(args.fmriprep_confounds_file),
        output_directory=os.path.abspath(args.output_directory),
        fixed_parameters=args.fixed_parameters

    )
    