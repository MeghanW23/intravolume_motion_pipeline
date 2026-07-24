import os 
import math 
from glob import glob
import SimpleITK as sitk

class WriteParametersToTextFile:
    def __init__(self,
                 transform_directory: str,
                 output_file_path: str,
                 transform_suffix: str = '.tfm',
                 output_rotation_unit: str = 'radian') -> None:

        print("Reading in transforms")
        transforms: list[sitk.VersorRigid3DTransform | sitk.Euler3DTransform] = \
            self.find_transform_paths(transform_directory, transform_suffix=transform_suffix)
        print(f"{len(transforms)} Total Transforms Found")
        
        print(f"Converting parameters to {output_rotation_unit}")
        all_converted_parameters: list[list[float]] = []
        for transform in transforms:
            rx, ry, rz, tx, ty, tz = transform.GetParameters()
            print(f"\nInput parameters: {[rx, ry, rz, tx, ty, tz]}")
            if isinstance(transform, sitk.VersorRigid3DTransform):
                print("Transform is a VersorRigid3DTransform")
                if output_rotation_unit == 'degrees':
                    output_rotations: list[float] = self.versor_to_degrees(rx, ry, rz)
                    all_converted_parameters.append(output_rotations + [tx, ty, tz])
                
                elif output_rotation_unit == 'radian':
                    output_rotations: list[float] = self.versor_to_radians(rx, ry, rz)
                    all_converted_parameters.append(output_rotations + [tx, ty, tz]) 
                
                elif output_rotation_unit == 'versor':
                    all_converted_parameters.append([rx, ry, rz, tx, ty, tz])
                

            elif isinstance(transform, sitk.Euler3DTransform):
                print("Your Transform is a Euler3DTransform")
                if output_rotation_unit == 'degrees':
                    output_rotations: list[float] = self.radians_to_degrees(rx, ry, rz)
                    all_converted_parameters.append(output_rotations + [tx, ty, tz]) 
                
                elif output_rotation_unit == 'radian':
                    all_converted_parameters.append([rx, ry, rz, tx, ty, tz])
                
                elif output_rotation_unit == 'versor':
                    output_rotations: list[float] = self.radians_to_versor(rx, ry, rz)
                    all_converted_parameters.append(output_rotations + [tx, ty, tz]) 
            else:
                raise ValueError(
                    "The input transforms must be one of two types:"
                    "\n1. SimpleITK's VersorRigid3DTransform"
                    "\n2. SimpleITK's Euler3DTransform"
                )
            print(f"Converted to parameters: {all_converted_parameters[-1]}")
        print(f"Converted {len(all_converted_parameters)} sets of parameters")

        print(f"Writing converted parameters to file: {output_file_path}")
        self.write_to_textfile(
            parameters=all_converted_parameters,
            output_file_path=output_file_path
        )
        print(f"Parameters written to: {output_file_path}")


    def find_transform_paths(self, 
                        transform_directory: str,
                        transform_suffix: str
                        ) -> list[sitk.VersorRigid3DTransform | sitk.Euler3DTransform]:

        transforms: list[sitk.VersorRigid3DTransform | sitk.Euler3DTransform] = [
            sitk.ReadTransform(transform_path) 
            for transform_path in sorted(glob(os.path.join(transform_directory, f"*{transform_suffix}")))
            if not "identity" in os.path.basename(transform_path)
        ]
        if not transforms:
            raise FileNotFoundError(
                f" No Transforms Found Matching: {os.path.join(transform_directory, f'*{transform_suffix}')}"
            )
        return transforms


    def versor_to_degrees(self, x: float, y: float, z: float) -> list[float]:
        
        # reconstruct quaternion
        w: float = math.sqrt(max(0.0, 1 - x*x - y*y - z*z))

        # rotation matrix
        r00: float = 1 - 2*(y*y + z*z)
        r01: float = 2*(x*y - z*w)

        r10: float = 2*(x*y + z*w)
        r11: float = 1 - 2*(x*x + z*z)

        r20: float = 2*(x*z - y*w)
        r21: float = 2*(y*z + x*w)
        r22: float = 1 - 2*(x*x + y*y)

        # euler angles
        ry: float = math.asin(-r20)

        if abs(r20) != 1:
            rx: float = math.atan2(r21, r22)
            rz: float = math.atan2(r10, r00)
        else:
            rx: float = 0
            rz: float = math.atan2(-r01, r11)

        return [
            math.degrees(rx),
            math.degrees(ry),
            math.degrees(rz)
        ]


    def versor_to_radians(self, x: float, y: float, z: float) -> list[float]:
        # reconstruct quaternion
        w: float = math.sqrt(max(0.0, 1 - x*x - y*y - z*z))

        # rotation matrix
        r00: float = 1 - 2*(y*y + z*z)
        r01: float = 2*(x*y - z*w)

        r10: float = 2*(x*y + z*w)
        r11: float = 1 - 2*(x*x + z*z)

        r20: float = 2*(x*z - y*w)
        r21: float = 2*(y*z + x*w)
        r22: float = 1 - 2*(x*x + y*y)

        # euler angles
        ry: float = math.asin(-r20)

        if abs(r20) != 1:
            rx: float = math.atan2(r21, r22)
            rz: float = math.atan2(r10, r00)
        else:
            rx: float = 0
            rz: float = math.atan2(-r01, r11)

        return [rx, ry, rz]


    def radians_to_degrees(self, x: float, y: float, z: float) -> list[float]:
        return [
            math.degrees(x),
            math.degrees(y),
            math.degrees(z)
        ]
        
    def radians_to_versor(self, rx: float, ry: float, rz: float) -> list[float]:
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)

        # rotation matrix R = Rz(rz) * Ry(ry) * Rx(rx)
        r00: float = cy * cz
        r01: float = cz * sy * sx - sz * cx
        r02: float = cz * sy * cx + sz * sx

        r10: float = cy * sz
        r11: float = sz * sy * sx + cz * cx
        r12: float = sz * sy * cx - cz * sx

        r20: float = -sy
        r21: float = cy * sx
        r22: float = cy * cx

        # rotation matrix -> quaternion (Shepperd's method)
        trace: float = r00 + r11 + r22

        if trace > 0:
            s: float = 0.5 / math.sqrt(trace + 1.0)
            w: float = 0.25 / s
            x: float = (r21 - r12) * s
            y: float = (r02 - r20) * s
            z: float = (r10 - r01) * s
        elif r00 > r11 and r00 > r22:
            s: float = 2.0 * math.sqrt(1.0 + r00 - r11 - r22)
            w: float = (r21 - r12) / s
            x: float = 0.25 * s
            y: float = (r01 + r10) / s
            z: float = (r02 + r20) / s
        elif r11 > r22:
            s: float = 2.0 * math.sqrt(1.0 + r11 - r00 - r22)
            w: float = (r02 - r20) / s
            x: float = (r01 + r10) / s
            y: float = 0.25 * s
            z: float = (r12 + r21) / s
        else:
            s: float = 2.0 * math.sqrt(1.0 + r22 - r00 - r11)
            w: float = (r10 - r01) / s
            x: float = (r02 + r20) / s
            y: float = (r12 + r21) / s
            z: float = 0.25 * s

        # keep w >= 0 to match versor_to_radians convention
        if w < 0:
            x, y, z, w = -x, -y, -z, -w

        return [x, y, z]
    
    def write_to_textfile(self, parameters: list[list[float]], output_file_path: str):
        with open(output_file_path, mode='w') as file:
            for parameter_list in parameters:
                file.write(
                    ', '.join([str(param_value) for param_value in parameter_list])
                    + "\n"
                )

if __name__ == "__main__":
    import argparse
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=\
            "Takes in a directory of transforms, outputs the parameters " +\
            "to a text file in the rotation unit of your choice."
    )
    parser.add_argument(
        "--transform_directory",
        required=True,
        help="Must be a directory of SimpleITK Versor 3D Transforms or Euler3D Transforms."
    )
    parser.add_argument(
        "--transform_suffix",
        required=False,
        default='.tfm',
        help=\
            "The file extension of the transforms in the directory. " + \
            "Default: '.tfm'."
    )
    parser.add_argument(
        "--output_file_path",
        required=True,
        help="File extension must be: '.txt'."
    )
    parser.add_argument(
        "--output_rotation_unit",
        required=True,
        choices=['versor', 'radian', 'degrees'],
        help="Choices: 'versor', 'radian', 'degrees'."
    )
    args: argparse.Namespace = parser.parse_args()
    WriteParametersToTextFile(
        transform_directory=os.path.abspath(args.transform_directory),
        output_file_path=os.path.abspath(args.output_file_path),
        transform_suffix=args.transform_suffix,
        output_rotation_unit=args.output_rotation_unit
    )