import os 
import math 
import numpy as np 
from glob import glob
import SimpleITK as sitk 

class CalculateDisplacements:
    def __init__(self, 
                 transform_directory: str,
                 output_file_path: str,
                 file_extension: str = '.tfm',
                 head_radius: float = 50) -> None:

        transform_paths: list[str] = sorted(glob(
            os.path.join(transform_directory, "*" + file_extension.strip())
        ))
        print(f"{len(transform_paths)} Transform Paths found in directory: {transform_directory}")
        if len(transform_paths) == 0:
            raise FileNotFoundError(
                 f"No Files found with file extension: '{file_extension.strip()}' "
                 f"in directory: {transform_directory}"
            )
        
        print("Calculating Displacements")
        all_displacements: list[float] = []
        for transform_num, _ in enumerate(transform_paths):
            if transform_num == 0:
                continue

            displacement: float = self.calculate_displacement(
                transform1_path=transform_paths[transform_num - 1],
                transform2_path=transform_paths[transform_num],
                head_radius=head_radius
            )
            
            print(
                f"Displacement Between: " + \
                transform_paths[transform_num - 1] + \
                " and " + \
                transform_paths[transform_num] + \
                f": {round(displacement, 4)}mm"
            )

            all_displacements.append(displacement)

        self.write_displacements_to_file(
            all_displacements,
            output_file_path=output_file_path
        )
        print(f"Displacements written to: {output_file_path}")
    
    def calculate_displacement(self, 
                               transform1_path: str, 
                               transform2_path: str, 
                               head_radius: float = 50
                               ) -> float:
        
        transform1: sitk.VersorRigid3DTransform | sitk.Euler3DTransform = \
            sitk.ReadTransform(transform1_path)
        transform2: sitk.VersorRigid3DTransform | sitk.Euler3DTransform = \
            sitk.ReadTransform(transform2_path)
        
        composed_affine: sitk.AffineTransform = self.compose_transforms(transform1, transform2)
        versorrigid3d: sitk.VersorRigid3DTransform = self.convert_affine_to_versorrigid(composed_affine)
        
        parms: np.ndarray = np.asarray( versorrigid3d.GetParameters() )
        versormagsquared: float = parms[0]*parms[0] + parms[1]*parms[1] + parms[2]*parms[2]
        versormag: float = math.sqrt(versormagsquared)
        wsquared: float = 1 - versormagsquared
        w : float = math.sqrt(wsquared)
        angle: float = 2.0 * math.atan2( versormag, w )
        deltarotationmm: float = float(head_radius) * float(angle)
        deltatranslationsquared: float = abs(parms[3])*abs(parms[3]) + abs(parms[4])*abs(parms[4]) + abs(parms[5])*abs(parms[5])
        deltatranslation: float = math.sqrt(deltatranslationsquared)
        totalmotion: float = deltarotationmm + deltatranslation

        return totalmotion
    

    def compose_transforms(self, 
                           transform1: sitk.VersorRigid3DTransform | sitk.Euler3DTransform, 
                           transform2: sitk.VersorRigid3DTransform | sitk.Euler3DTransform
                           ) -> sitk.AffineTransform:
            
            transform1_inverse: sitk.VersorRigid3DTransform | sitk.Euler3DTransform = transform1.GetInverse()

            A0: np.ndarray = np.asarray(transform2.GetMatrix()).reshape(3,3)
            c0: np.ndarray = np.asarray(transform2.GetCenter())
            t0: np.ndarray = np.asarray(transform2.GetTranslation())

            A1: np.ndarray = np.asarray(transform1_inverse.GetMatrix()).reshape(3,3)
            c1: np.ndarray = np.asarray(transform1_inverse.GetCenter())
            t1: np.ndarray = np.asarray(transform1_inverse.GetTranslation())

            combined_mat: np.ndarray = np.dot(A0,A1)
            combined_center: np.ndarray = c1
            combined_translation: np.ndarray = np.dot(A0, t1+c1-c0) + t0+c0-c1
            combined_affine: sitk.AffineTransform = sitk.AffineTransform(
                combined_mat.flatten(), 
                combined_translation, 
                combined_center
            )

            return combined_affine


    def convert_affine_to_versorrigid(self, 
                                      affinetransform: sitk.AffineTransform
                                      ) -> sitk.VersorRigid3DTransform:
        
        versorrigid3d: sitk.VersorRigid3DTransform = sitk.VersorRigid3DTransform()
        
        versorrigid3d.SetCenter( affinetransform.GetCenter() )
        versorrigid3d.SetTranslation( affinetransform.GetTranslation() )
        versorrigid3d.SetMatrix( affinetransform.GetMatrix() )
        
        return versorrigid3d
    
    def write_displacements_to_file(self, 
                                    displacements: list[float],
                                    output_file_path: str) -> None:
       
        with open(output_file_path, mode='w') as file:
            for displacement in displacements:
                file.write(f"{displacement}\n")
        

if __name__ == "__main__":
    import argparse
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
         description=\
            "Calculate mm displacements from a directory of " + \
            "SimpleITK 3D Rigid-Body Transforms. " + \
            "i.e. VersorRigid3DTransform or Euler3DTransform."
    )
    parser.add_argument(
        "--transform_directory",
        required=True
    )
    parser.add_argument(
        "--file_extension",
        required=False,
        default='.tfm',
        help=\
            "The file extension of the transform files" + \
            "in the transform directory. Default: '.tfm'. " + \
            "the order of transform files must be sortable via sorted()."
    )
    parser.add_argument(
        "--output_file_path",
        required=False,
        default='displacements.txt',
        help=\
            "The file path to write the displacements to." + \
            f" Default: '{os.path.abspath('displacements.txt')}'."
    )
    parser.add_argument(
        "--head_radius",
        required=False,
        type=float,
        default=50,
        help="Radius, in mm, of the participant's head. Default: 50mm."
    )
    args: argparse.Namespace = parser.parse_args()
    CalculateDisplacements(
        transform_directory=os.path.abspath(args.transform_directory),
        file_extension=args.file_extension,
        output_file_path=os.path.abspath(args.output_file_path),
        head_radius=args.head_radius
    )