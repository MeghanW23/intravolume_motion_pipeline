import os
import sys

class WriteEulerTransformsFromParameterFile:
    def __init__(self, parameter_file, fixed_parameters, output_directory):
        
        print(f"Parameter Text File: {parameter_file}")
        
        print(f"Fixed Parameters: {fixed_parameters}")
        if len(fixed_parameters) != 3:
            print(f"Fixed parameters must have 3 values.")
            sys.exit(1)

        print(f"Output Directory for Transforms: {output_directory}")
        os.makedirs(output_directory, exist_ok=True)


        parameters = self.read_parameters(parameter_file)
        if any([len(parameter_list) != 6 for parameter_list in parameters]):
            print(f"Every aquisition's parameters must have 6 values.")
            sys.exit(1)
        print(f"There are {len(parameters)} Aquisitions Total in this Parameter File.")


        for aquisition_num, parameter_list in enumerate(parameters):
            output_transform_path = os.path.join(output_directory, f"aquisition-{'{:05d}'.format(aquisition_num)}.tfm")
            self.write_transform(
                parameters=parameter_list,
                fixed_parameters=fixed_parameters,
                output_file=output_transform_path)
            print(f"Wrote Transform File: {output_transform_path}")
        
        print("Done.")

            
    def write_transform(self, parameters, fixed_parameters, output_file):
        with open(output_file, mode='w') as file:
            file.write("#Insight Transform File V1.0\n")
            file.write("#Transform 0\n")
            file.write("Transform: Euler3DTransform_double_3_3\n")
            file.write("Parameters: " + ' '.join([str(float(param)) for param in parameters]) + '\n')
            file.write("FixedParameters: " + ' '.join([str(param) for param in fixed_parameters]))
            file.write('\n') 


    def read_parameters(self, parameter_file):
        parameters = []
        with open(parameter_file, mode='r') as file:
            for line in file:
                parameters.append([
                    float(value.strip())
                    for value in line.strip().split(' ')
                    if value.strip()
                ])
        return parameters



if __name__ == '__main__':

    """
    python ./lab-share/Neuro-Cohen-e2/Groups/IRB-P00049401/mw_motion_pipeline/run_pipeline/helper_scripts/write_euler_transforms_from_parameter_file.py \
        --parameter_file parameters.txt \
        --fixed_parameters -3.9139926433563232 -12.933999300003052 2.904700994491577 \
        --transform_directory_path ./parameter_euler_transforms
    """

    import argparse

    parser = argparse.ArgumentParser(description="Write a parameters.txt text file to a directory of transforms")
    parser.add_argument("--parameter_file", help="Rotation Parameters must be in RADIANS", required=True)
    parser.add_argument("--fixed_parameters", required=True, nargs='+', type=float)
    parser.add_argument("--transform_directory_path", required=True)
    args = parser.parse_args()

    WriteEulerTransformsFromParameterFile(
        parameter_file=os.path.abspath(args.parameter_file),
        fixed_parameters=args.fixed_parameters,
        output_directory=os.path.abspath(args.transform_directory_path)
    )