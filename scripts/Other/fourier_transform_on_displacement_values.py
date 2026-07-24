import os 
import json
import math
import numpy as np
from glob import glob
import nibabel as nib
import SimpleITK as sitk 
from scipy.signal import find_peaks
from  plotly import graph_objects as go
class FourierTransform:

    def __init__(self, transform_directory, nifti_image_path, json_file_path, output_file_path, transform_suffix = ".tfm", input_rotation_unit = "versor"):
        
        """
        1. Load Input Data 
        """
        print("Loading JSON Data")
        json_data = {}
        with open(json_file_path, mode='r') as f:
            json_data = json.load(f)
        
        print("Loading Nifti Data")
        nifti_image = nib.load(nifti_image_path)
        

        """
        2. Get Displacements from Transform Directory
        """
        print("Extracting Displacements from Transform Files")
        # Get and load transforms from transform directory 
        transforms = self.find_transform_paths(transform_directory, transform_suffix)
        
        # Get parameters from files and convert rotation unit to degrees
        self.parameters = self.extract_parameters(transforms, input_rotation_unit)
        
        # Calculate Displacements
        self.displacements = [0] + [self.compute_displacement(transform1=transforms[i - 1], transform2=transforms[i]) for i in range(len(transforms)) if i != 0]


        """
        3. Define Sampling Parameters
        """
        num_aquisitions = len(transforms)
        print(f"{num_aquisitions} Total Aquisitions")
        
        num_volumes = nifti_image.shape[-1]
        print(f"{num_volumes} Total Volumes")

        tr =  json_data['RepetitionTime']
        print(f"{tr}s Repetition Time")

        duration = tr * num_volumes
        print(f"Total Duration of Run: {round(duration, 2)} seconds (repetition time * number of volumes)")
       
        time = np.linspace(0, duration, num_aquisitions, endpoint=False) # Time axis
        print(f"X-Axis (Time in seconds, {len(time)} Total Points): {time}")

        """
        4. Compute FFT 

        FFT expects complex-valued inputs and computes both positive and negative frequencies, 
        while RFFT is optimized specifically for real-valued inputs and 
        skips computing redundant negative frequencies
        """

        # remove mean from data to prevent big spike at the beginning
        meanfree_signal = np.array(self.displacements) - np.mean(np.array(self.displacements))

        # Compute the one-dimensional discrete Fourier Transform for real input
        fft_values = np.fft.rfft(meanfree_signal)
        print(f"FFT Values: {fft_values}")
        print(len(fft_values))

        # Absolute values of fft_values
        magnitude = np.abs(fft_values)
        print(f"Magnitude: {magnitude}")

        # Return the Discrete Fourier Transform sample frequencies
        # n : int
        #   Window length.
        # d : scalar, optional
        #   Sample spacing (inverse of the sampling rate). Defaults to 1.
        frequencies = np.fft.rfftfreq(n=len(self.displacements), d=tr)
        print(f"Frequencies: {frequencies}")

        """
        5. Plot Results
        """
        fig = go.Figure()
        fig.update_layout(
            title=f"Frequency x Magnitude: {os.path.basename(nifti_image_path).replace('.nii.gz', '')}",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Magnitude"
        )
        fig.add_trace(
            go.Scatter(
                x=frequencies[1:],
                y=magnitude[1:]
            )
        )
        fig.show()
        fig.write_html(output_file_path)

        print(f"Output Graph at: {output_file_path}")

    
    def find_transform_paths(self, transform_directory, transform_suffix):

        transforms = [
            sitk.ReadTransform(transform_path) 
            for transform_path in sorted(glob(os.path.join(transform_directory, f"*{transform_suffix}")))
            if not "identity" in os.path.basename(transform_path)
        ]
        if not transforms:
            print(f"\nERROR: No Transforms Found Matching: {os.path.join(transform_directory, f'*{transform_suffix}')}")
            exit(0)
        
        return transforms


    def extract_parameters(self, transforms, input_rotation_unit):
        parameters = [list(transform.GetParameters()) for transform in transforms]
        if input_rotation_unit == 'versor':
            return [
                self.versor_to_degrees(rx, ry, rz) + [tx, ty, tz]
                for rx, ry, rz, tx, ty, tz in parameters
            ]

        elif input_rotation_unit == 'radians':
            return [
                self.radians_to_degrees(rx, ry, rz) + [tx, ty, tz]
                for rx, ry, rz, tx, ty, tz in parameters
            ]
        
        elif input_rotation_unit == 'degrees':
            return parameters

        else:
            print(f"\nERROR: input_rotation_unit must be one of the following options: 'versor', 'radians', 'degrees'.")
            exit(0)


    def versor_to_degrees(self, x, y, z):
        
        # reconstruct quaternion
        w = math.sqrt(max(0.0, 1 - x*x - y*y - z*z))

        # rotation matrix
        r00 = 1 - 2*(y*y + z*z)
        r01 = 2*(x*y - z*w)

        r10 = 2*(x*y + z*w)
        r11 = 1 - 2*(x*x + z*z)

        r20 = 2*(x*z - y*w)
        r21 = 2*(y*z + x*w)
        r22 = 1 - 2*(x*x + y*y)

        # euler angles
        ry = math.asin(-r20)

        if abs(r20) != 1:
            rx = math.atan2(r21, r22)
            rz = math.atan2(r10, r00)
        else:
            rx = 0
            rz = math.atan2(-r01, r11)

        return [
            math.degrees(rx),
            math.degrees(ry),
            math.degrees(rz)
        ]


    def radians_to_degrees(self, x, y, z):
        return [
            math.degrees(x),
            math.degrees(y),
            math.degrees(z)
        ]


    def write_parameters_to_text_file(self, parameters, output_text_file_path):
        with open(output_text_file_path, mode='w') as f:
            for parameter_list in parameters:
                f.write(' '.join([str(parameter) for parameter in parameter_list]) + '\n')  


    def compute_displacement(self, transform1, transform2, radius = 50):

        A0 = np.asarray(transform2.GetMatrix()).reshape(3, 3)
        c0 = np.asarray(transform2.GetCenter())
        t0 = np.asarray(transform2.GetTranslation())

        A1 = np.asarray(transform1.GetInverse().GetMatrix()).reshape(3, 3)
        c1 = np.asarray(transform1.GetInverse().GetCenter())
        t1 = np.asarray(transform1.GetInverse().GetTranslation())

        combined_mat = np.dot(A0,A1)
        combined_translation = np.dot(A0, t1+c1-c0) + t0+c0-c1

        versorrigid3d = sitk.VersorRigid3DTransform()
        versorrigid3d.SetCenter(c1)
        versorrigid3d.SetTranslation(combined_translation)
        versorrigid3d.SetMatrix(combined_mat.flatten())

        euler3d = sitk.Euler3DTransform()
        euler3d.SetCenter(c1)
        euler3d.SetTranslation(combined_translation)
        euler3d.SetMatrix(combined_mat.flatten())

        parms = np.asarray(euler3d.GetParameters())
        
        return np.sqrt(
            (parms[0]*radius)**2 +
            (parms[1]*radius)**2 +
            (parms[2]*radius)**2 +
            parms[3]**2 +
            parms[4]**2 +
            parms[5]**2
        )
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Perform Fourier Transform on the Displacement Values from a Directory of Versor3d SimpleITK Transforms")
    parser.add_argument("--transform_directory", required=True)
    parser.add_argument("--nifti_image_path", required=True)
    parser.add_argument("--json_file_path", required=True)
    parser.add_argument("--output_file_path", required=False, default="fourier_transform.html", help="Default: 'fourier_transform.html'")
    args = parser.parse_args()

    FourierTransform(
        transform_directory=os.path.abspath(args.transform_directory),
        nifti_image_path=os.path.abspath(args.nifti_image_path),
        json_file_path=os.path.abspath(args.json_file_path),
        output_file_path=os.path.abspath(args.output_file_path)
    )

