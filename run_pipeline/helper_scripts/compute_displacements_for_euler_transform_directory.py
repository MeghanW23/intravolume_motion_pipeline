import os 
import numpy as np 
from glob import glob 
import SimpleITK as sitk 

class ComputeDisplacementsFromEulerTransformDirectory:
    def __init__(self, transform_directory, output_displacements_path):
        
        print(f"Transform Directory: {transform_directory}")
        print(f"Output Displacements File Path: {output_displacements_path}")

        transforms = sorted(glob(os.path.join(transform_directory, "*tfm")))
        print(f"I Found {len(transforms)} Transforms Total.")


        displacements = []
        for i in range(len(transforms)):
            if i == 0:
                continue 
            
            print(f"Calculating Displacement between Aqusition {i - 1} and {i}.")
            displacements.append(self.compute_displacement(
                transform1=sitk.ReadTransform(transforms[i - 1]),
                transform2=sitk.ReadTransform(transforms[i])
            ))

        print(f"I Created {len(displacements)} Displacement Values.")

        self.write_displacement_path(
            displacements=displacements,
            output_displacements_path=output_displacements_path
        )
        print(f"Done. Displacements at: {output_displacements_path}")


    def compute_displacement(self, transform1, transform2, radius=50):

        A0 = np.asarray(transform2.GetMatrix()).reshape(3, 3)
        c0 = np.asarray(transform2.GetCenter())
        t0 = np.asarray(transform2.GetTranslation())

        A1 = np.asarray(transform1.GetInverse().GetMatrix()).reshape(3, 3)
        c1 = np.asarray(transform1.GetInverse().GetCenter())
        t1 = np.asarray(transform1.GetInverse().GetTranslation())

        combined_mat = np.dot(A0,A1)
        combined_center = c1
        combined_translation = np.dot(A0, t1+c1-c0) + t0+c0-c1
        combined_affine = sitk.AffineTransform(combined_mat.flatten(), combined_translation, combined_center)

        versorrigid3d = sitk.VersorRigid3DTransform()
        versorrigid3d.SetCenter(combined_center)
        versorrigid3d.SetTranslation(combined_translation)
        versorrigid3d.SetMatrix(combined_mat.flatten())

        # First three parameters are rotation angles in radians.
        # Second three parameters are translations.
        euler3d = sitk.Euler3DTransform()
        euler3d.SetCenter(combined_center)
        euler3d.SetTranslation(combined_translation)
        euler3d.SetMatrix(combined_mat.flatten())

        # Compute the displacement:
        parms = np.asarray( euler3d.GetParameters() )
        return \
            abs(parms[0]*radius) + abs(parms[1]*radius) + \
            abs(parms[2]*radius) + abs(parms[3]) + abs(parms[4]) + abs(parms[5])


    def write_displacement_path(self, displacements, output_displacements_path):

        with open(output_displacements_path, mode='w') as file:
            for displacement in displacements:
                file.write(str(displacement) + '\n')
        


if __name__ == '__main__':

    """
    python /lab-share/Neuro-Cohen-e2/Groups/IRB-P00049401/mw_motion_pipeline/run_pipeline/helper_scripts/compute_displacements_for_euler_transform_directory.py \
        --transform_directory parameter_euler_transforms/ \
        --output_displacement_file_path displacements.txt
    """

    import argparse

    parser = argparse.ArgumentParser(description="Calculate the displacements between a directory of Euler3DTransforms")
    parser.add_argument("--transform_directory", required=True, help='The transforms must be ordered so that python function "sort()" will order the transforms correctly.')
    parser.add_argument("--output_displacement_file_path", required=True, help='The text file the displacements will be written in to.')
    args = parser.parse_args()

    ComputeDisplacementsFromEulerTransformDirectory(
        transform_directory=os.path.abspath(args.transform_directory),
        output_displacements_path=os.path.abspath(args.output_displacement_file_path)
    )