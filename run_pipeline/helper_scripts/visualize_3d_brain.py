import json
import math
import os
import vedo
import numpy as np
from glob import glob 
from typing import OrderedDict


class Visualizer():

    def __init__(self, transform_dir, template_brain_path, json_path):
        # Input Paths
        self.transform_dir: str = transform_dir
        if not os.path.exists(self.transform_dir):
            raise FileNotFoundError(f"{self.transform_dir} Does Not Exist.")
        
        self.transform_paths: list[str] = sorted(glob(f"{self.transform_dir}/*alignTransform_align-*.tfm"))
        
        
        # Load JSON Data 
        self.json_data: dict[str, any] = {}
        with open(json_path) as f:
            self.json_data: dict[str, any] = json.load(f)

        print(f"Phase Encoding Direction: {self.json_data['PhaseEncodingDirection']}")
        
        # Load brain as vedo Mesh object and get it's attributes
        self.template_brain: vedo.Mesh = vedo.load(template_brain_path).isosurface(5)
        self.template_brain.color("grey")

        self.center_x, self.center_y, self.center_z = self.get_fixed_center(transform_path=self.transform_paths[0])

        self.slice_axis = self.get_slice_axis(self.json_data['ImageOrientationPatientDICOM'])

        min, max = self.get_min_and_max_locations(bounds=self.template_brain.bounds())
        self.slice_positions = np.linspace(min, max, num=len(self.json_data['SliceTiming']) + 1)
        
        print(f"Slice Positions: {self.slice_positions}")

        # Create plot vedo object 
        self.plot: vedo.Plotter = vedo.Plotter(title="Slice-Level Motion Visualizer")

        self.slice_dict, self.slice_dict_copy = self.create_slices(num_slices=len(self.json_data['SliceTiming']))
        
        self.slice_timing_dict: dict[str, any] = OrderedDict(self.find_matching_indexes(numbers=self.json_data['SliceTiming']).items())
        print(f"Slice Timings:\n{self.slice_timing_dict}")
        
        self.motion_slider: vedo.addons.Slider2D = self.plot.add_slider(
            sliderfunc=self.on_slider_change,
            xmin=1, 
            xmax=100, 
            value=1,
            pos=[(0.51, 0.06), (0.95, 0.06)],  # bottom-right corner
            title="Motion Magnitude"
        )

        self.aquisition_slider: vedo.addons.Slider2D = self.plot.add_slider(
            sliderfunc=self.on_slider_change,
            xmin=1, 
            xmax=len(self.slice_timing_dict), 
            value=1,
             pos=[(0.05, 0.06), (0.49, 0.06)],  # bottom-left
            title="Slice Group Number",
        )

        self.volume_slider: vedo.addons.Slider2D = self.plot.add_slider(
            sliderfunc=self.on_slider_change,
            xmin=1, 
            xmax=round(len(self.transform_paths) / len(self.slice_timing_dict)), 
            value=1,
            pos=[(0.05, 0.16), (0.49, 0.16)],  # stacked just above acquisition
            title="Volume Number", 
        )

        self.text: vedo.Text2D = vedo.Text2D(f"ProtocolName: {self.json_data['ProtocolName']}\nFixed Rotation (X,Y,Z): {round(self.center_x, 5)} rad, {round(self.center_y, 5)} rad, {round(self.center_z, 5)} rad\nRotation (X,Y,Z): 0.0 rad, 0.0 rad, 0.0 rad\nTranslation (X,Y,Z): 0.0 mm, 0.0 mm, 0.0mm", pos="top-left")
        self.plot += self.text

        # apply first aquisition to the brain 
        self.update_slices(scale=1, aquisition_num=0)

        self.plot.show(axes=1)


    def get_min_and_max_locations(self, bounds: list[float]) -> tuple[float, float]:
        axis_index = self.slice_axis.index(1.0)
        min_val = bounds[axis_index * 2]
        max_val = bounds[axis_index * 2 + 1]
        return min_val, max_val

    def on_slider_change(self, _: vedo.addons.Slider2D, event: str):
        volume_num: int = round(self.volume_slider.GetRepresentation().GetValue() - 1)
        volume_num_in_aquisition_format: int = volume_num * len(self.slice_timing_dict)
        slice_group: int = round(self.aquisition_slider.GetRepresentation().GetValue() - 1)

        self.update_slices(
            scale=self.motion_slider.GetRepresentation().GetValue(),
            aquisition_num=volume_num_in_aquisition_format + slice_group)


    def update_slices(self, scale: float, aquisition_num: int):
        current_volume: str = '{:04d}'.format(int(aquisition_num / len(self.slice_timing_dict)))
        current_slice_group: str = '{:04d}'.format(aquisition_num % len(self.slice_timing_dict))

        print(f" -------- Updating --------")
        print(f"Scale: {scale}, Aquisition Number: {aquisition_num}, Current Volume: {int(current_volume) + 1}, Current Slice Group: {int(current_slice_group) + 1}")

        transform_path: str = glob(os.path.join(self.transform_dir, f"*alignTransform_align-{current_volume}-{current_slice_group}.tfm"))[0]

        slices_indices: list[int] = self.get_slice_indices_at_this_transform(transform_file=transform_path)
        print(f"Slice Indices: {slices_indices}")
        
        r_x, r_y, r_z, t_x, t_y, t_z = self.get_transform_values(transform_path=transform_path)

        self.text.text(f"Protocol Name: {self.json_data['ProtocolName']}\nFixed Rotation (X,Y,Z): {round(self.center_x, 5)} rad, {round(self.center_y, 5)} rad, {round(self.center_z, 5)} rad\nRotation (X,Y,Z): {round(r_x, 5)} rad, {round(r_y, 5)} rad, {round(r_z, 5)} rad\nTranslation (X,Y,Z): {round(t_x, 5)} mm, {round(t_y, 5)} mm, {round(t_z, 5)} mm")

        for index, _ in self.slice_dict.items():
            self.plot.remove(self.slice_dict[index])
            self.slice_dict[index] = self.slice_dict_copy[index].copy()
            self.slice_dict[index].alpha(0.2)
            self.plot += self.slice_dict[index]

        for slice_index in slices_indices:
            self.plot.remove(self.slice_dict[slice_index])

            self.slice_dict[slice_index] = self.slice_dict_copy[slice_index].copy()
            self.slice_dict[slice_index] = self.apply_transform(
                slice=self.slice_dict[slice_index],
                r_x=r_x * scale,
                r_y=r_y * scale,
                r_z=r_z * scale,
                t_x=t_x * scale,
                t_y=t_y * scale,
                t_z=t_z * scale)

            self.slice_dict[slice_index].color("red")

            self.plot += self.slice_dict[slice_index]
            

    def find_matching_indexes(self, numbers: list[float]) -> dict[float, list[int]]:

        num_index_map: dict[float, list[int]] = {}
    
        for index, number in enumerate(numbers):
            if number in num_index_map:
                num_index_map[number].append(index)
            else:
                num_index_map[number] = [index]
    
        results: dict[float, list[int]] = {number: indexes for number, indexes in num_index_map.items() if len(indexes) > 1}

        return results
    

    def get_slice_indices_at_this_transform(self, transform_file: str) -> list[int]:
        slice_index: int = int(os.path.basename(transform_file).split("-")[-1].replace(".tfm", ""))
        keys_list = list(self.slice_timing_dict.keys())
        slice_indices: list[int] = self.slice_timing_dict[keys_list[slice_index]]
        return slice_indices


    def get_slice_axis(self, ImageOrientationPatientDICOM: list[int]) -> list[float]:
        row_direction = ImageOrientationPatientDICOM[:3]
        column_direction = ImageOrientationPatientDICOM[3:]
        slice_dir = np.cross(row_direction, column_direction)
        axis_num = int(np.argmax(np.abs(slice_dir)))
        axis = [0.0, 0.0, 0.0]
        axis[axis_num] = 1.0
        return axis

    def get_origin(self, index: int) -> tuple[list[float, float, float], list[float, float, float]]:
        
        moving_location_start: float = self.slice_positions[index]
        moving_location_end: float = self.slice_positions[index + 1]

        origin1 = [self.center_x, self.center_y, self.center_z]
        origin2 = [self.center_x, self.center_y, self.center_z]

        axis_index = self.slice_axis.index(1.0)
        origin1[axis_index] = moving_location_start
        origin2[axis_index] = moving_location_end

        return origin1, origin2

    def create_slices(self, num_slices: int) -> tuple[dict[int, vedo.Mesh], dict[int, vedo.Mesh]]:

        slices: dict[int, vedo.Mesh] = {}
        slice_copy: dict[int, vedo.Mesh] = {}

        # Determine if positions are ascending or descending
        ascending = self.slice_positions[1] > self.slice_positions[0]
        forward_normal = self.slice_axis if ascending else [-v for v in self.slice_axis]

        for i in range(num_slices):

            origin_coordinates1, origin_coordinates2 = self.get_origin(index=i)

            print(f"Creating Slice {i + 1}/{num_slices} at: ({round(origin_coordinates1[0], 2)}, {round(origin_coordinates1[1], 2)}, {round(origin_coordinates1[2], 2)}) to ({round(origin_coordinates2[0], 2)}, {round(origin_coordinates2[1], 2)}, {round(origin_coordinates2[2], 2)})")

            cut_brain: vedo.Mesh = self.template_brain.copy()

            # first cut: remove everything behind origin1
            if i != 0:
                cut_brain = cut_brain.cut_with_plane(
                    origin_coordinates1,
                    normal=forward_normal
                )

            # second cut: remove everything ahead of origin2
            if i != num_slices - 1:
                cut_brain = cut_brain.cut_with_plane(
                    origin_coordinates2,
                    normal=forward_normal,
                    invert=True
                )

            cut_brain.cap()

            slices[i] = cut_brain
            slice_copy[i] = cut_brain.copy()

            self.plot += slices[i]

        return slices, slice_copy


    def get_transform_values(self, transform_path: str) -> tuple[float, float, float, float, float, float]:
        with open(file=transform_path, mode="r") as file:
            for line in file:
                if "Parameters:" in line and not "FixedParameters:" in line:
                    all_params: list[str] = line.split()[1:]
                    rx, ry, rz = self.versor_to_radians(float(all_params[0]), float(all_params[1]), float(all_params[2]))
                    return rx, ry, rz, float(all_params[3]), float(all_params[4]), float(all_params[5])


    def versor_to_radians(self, x, y, z):
        
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

        return rx, ry, rz


    def get_fixed_center(self, transform_path: str) -> tuple[float, float, float]:
         with open(file=transform_path, mode="r") as file:
            for line in file:
                if "FixedParameters:" in line:
                    all_params: list[str] = line.split()[1:]
                    return float(all_params[0]), float(all_params[1]), float(all_params[2])
         

    def apply_transform(self, slice: vedo.Mesh, r_x: float, r_y: float, r_z: float, t_x: float, t_y: float, t_z: float) -> vedo.Mesh:
        slice.rotate_x(r_x, around=[self.center_x, self.center_y, self.center_z])
        slice.rotate_y(r_y, around=[self.center_x, self.center_y, self.center_z])
        slice.rotate_z(r_z, around=[self.center_x, self.center_y, self.center_z])

        slice.shift(t_x, t_y, t_z)

        return slice

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Visualize the 3D Motion of a Slice Group from a Directory of SimpleItk Versor3D Transforms.")
    parser.add_argument(
        "--transform_directory", 
        required=True,
        help="Input a path of SimpleITK Versor3D Parameters"
    )
    parser.add_argument(
        "--json_path", 
        required=True
    )
    parser.add_argument(
        "--template_brain_path", 
        required=False, 
        default=f"{os.environ['FSLDIR']}/data/standard/MNI152_T1_2mm_brain.nii.gz",
        help=f"3D Template Brain as a NiFTI Image. Default is FSL's T1 2mm MNI brain at: {os.environ['FSLDIR']}/data/standard/MNI152_T1_2mm_brain.nii.gz"
    )

    args = parser.parse_args()
    Visualizer(
        transform_dir=os.path.abspath(args.transform_directory),
        template_brain_path=os.path.abspath(args.template_brain_path),
        json_path=os.path.abspath(args.json_path)
    )
    
    
    
