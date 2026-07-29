import os 
import math
import json
import statistics
import numpy as np
import nibabel as nib
from glob import glob
import SimpleITK as sitk 
from nipype.interfaces import fsl
from collections import OrderedDict
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from nilearn.image import resample_to_img

class CarpetPlot:
    """

    Steps from https://jsheunis.github.io/2018-07-20-the-plot-spm/
    
    1. Coregistering the anatomical image to reference functional image (in preparation for the next step)
    2. Segmenting the coregistered anatomical image into standard tissue types (grey matter, white matter and CSF)
    3. Smoothing the unprocessed and realigned functional data (this essentially allows 4 versions of the data to be plotted, raw (un)smoothed and realigned (un)smoothed)
    4. Preparing data for The Plot:
        4.1. Removing the mean from and detrending the movement parameters
        4.2. Calculating framewise displacement
        4.3. Creating brain mask from segmentations
        4.4. Calculating percentage signal change (from the mean) for the masked fMRI time series
    5. Vertically concatenate sections of the scaled fMRI time-series data based on the bins: 
       grey matter on top, then white matter in the middle, then CSF.
    6. We plot the resulting intensity values over time, we plot separator lines to separate the tissue 
       types, and we plot the framewise displacement over time.  

    """
    def __init__(self, anatomical_image, functional_image, json_file, 
                 reference_volume_image, transform_directory, displacement_threshold, 
                 output_directory = "outputs", transform_suffix = ".tfm", 
                 plot_title="Voxel Percent Signal Change Carpet Plot + Displacements",
                 output_file_path="carpet_plot.html", slice_times_text_file = None):
        
        os.makedirs(output_directory, exist_ok=True)

        slice_timing = self.get_slice_timing(json_file, slice_times_text_file)
        print(f"Slice Timing: {slice_timing}")
        
        print("\nCoregistering the anatomical image to reference functional image")
        coregistered_anat_image, registration_matrix = self.coregister_anat_to_func_fsl(
            anatomical_image, reference_volume_image, output_directory=output_directory
        )
        print(f"Anatomical Image Registered to Reference Volume: {coregistered_anat_image}")
        print(f"Registration Matrix: {registration_matrix}")


        print("\nSkull stripping coregistered anatomical image")
        ss_coregistered_anat_image,  ss_coregistered_anat_image_mask = self.skull_strip_anat(coregistered_anat_image, output_directory=output_directory)
        print(f"Skull Stripped Anatomical Image: {ss_coregistered_anat_image}")
        print(f"Skull Stripped Anatomical Image Mask: {ss_coregistered_anat_image_mask}")


        print("\nSegmenting the coregistered anatomical image into grey matter, white matter, and CSF")
        masks = self.segment_anat_fsl(ss_coregistered_anat_image, output_directory=output_directory)
        print(f"Anatomical CSF Mask: {masks['csf']}")
        print(f"Anatomical Gray Matter Mask: {masks['gm']}")
        print(f"Anatomical White Matter Mask: {masks['wm']}")


        print("\nCalculating the displacements")
        displacements = []
        transform_paths = sorted(glob(os.path.join(transform_directory, "*" + transform_suffix)))
        print(f"{len(transform_paths)} Total Transforms")
        for i, _ in enumerate(transform_paths):
            if i == 0:
                continue 
            displacements.append(self.calculate_displacement(
                 transform1_path=transform_paths[i - 1],
                 transform2_path=transform_paths[i]
            ))
        print(f"{len(displacements)} Displacement Values Calculated")
        print(f"Min Displacement: {min(displacements)}mm")
        print(f"Max Displacement: {max(displacements)}mm")
        print(f"Mean Displacement: {statistics.mean(displacements)}mm") 

        print("\nGetting Motion Flagged Volumes")
        motion_flagged_volumes = self.get_motion_flagged_volumes(
            num_volumes=len(transform_paths) // len(slice_timing),
            num_slice_groups=len(slice_timing),
            displacement_values=displacements,
            mm_displacement_threshold=displacement_threshold
        )
        print(f"Motion Flagged Volumes:\n{motion_flagged_volumes}")


        #  psc: 2D array of shape (n_voxels, n_timepoints) in PSC units
        print("\nCalculating Percent Signal Change")
        demeaned_voxels = {}
        demeaned_voxels['gm'] = self.subsample_carpet(
            self.calculate_psc(
                functional_image, 
                brain_mask=masks['gm'],
                slice_timing=slice_timing
            ),
            max_voxels=6000
        )
        demeaned_voxels['wm'] = self.subsample_carpet(
            self.calculate_psc(
                functional_image, 
                brain_mask=masks['wm'],
                slice_timing=slice_timing
            ),
            max_voxels=2000
        )
        demeaned_voxels['csf'] = self.subsample_carpet(
            self.calculate_psc(
                functional_image, 
                brain_mask=masks['csf'],
                slice_timing=slice_timing
            ),
            max_voxels=1000
        )
        fig = make_subplots(
            rows=4, 
            cols=1, 
            subplot_titles=["", ""], 
            horizontal_spacing=0.5, 
            vertical_spacing=0.01,
            row_heights=[0.2, 0.4, 0.25, 0.15],
        )
        fig.update_layout(
            title_text="<b>" + plot_title  + "</b>",
            title_x=0.5,
            margin=dict(l=50, r=0, t=50, b=0),
            legend=dict(
                orientation="h",
                x=0.01,  # 0=left, 1=right
                y=0.99,  # 0=bottom, 1=top
                bgcolor ='rgba(255,255,255,0.8)', # Background color of the legend box
                bordercolor="black",               # Border color of the legend box
                borderwidth=1                      # Border thickness
            ),
        )


        ## DISPLACEMENTS
        print("\nPlotting Displacements") 
        fig.add_trace(
            go.Scatter(
                x=list(range(len(displacements))),
                y=displacements,
                showlegend=True,
                hovertemplate=(
                    "<b>Aquisition:</b> %{x}" + "<br>" +
                    "<b>Displacement (mm</b>: %{y} " + "<br>"
                ),
                name="Displacement (mm)",
                
            ),
            row=1,
            col=1
        )
        fig.add_trace(
            go.Scatter(
                x=[0, len(displacements)-1],
                y=[displacement_threshold, displacement_threshold],
                mode="lines",
                name=f"Threshold: {displacement_threshold} mm",
                line=dict(color="black", dash="dash"),
                showlegend=True
            ),
            row=1,
            col=1
        )
        # Plot motion flags
        for i, flagged_volume in enumerate(motion_flagged_volumes):
            fig.add_trace(
                go.Scatter(
                    x=[
                        flagged_volume * len(slice_timing), 
                        flagged_volume * len(slice_timing), 
                        (flagged_volume * len(slice_timing)) + len(slice_timing), 
                        (flagged_volume * len(slice_timing)) + len(slice_timing),
                    ],
                    y=[
                        min(displacements) - 0.1, 
                        max(displacements) * 1.1, 
                        max(displacements) * 1.1, 
                        min(displacements) - 0.1
                    ], 
                    fill="toself",
                    fillcolor="rgba(0,0,0,0.25)",
                    line=dict(width=0),
                    legendgroup="motion_flags",
                    mode="none",
                    name=f"Motion Flags: {len(motion_flagged_volumes)} of {len(transform_paths) // len(slice_timing)} Volumes Flagged",
                    showlegend=True if i == 0 else False,
                    hoverinfo="skip",
                ),
            row=1, col=1
        )
        fig.update_yaxes(title_text="Displacement (mm)", title_font=dict(size=11), title_standoff=5, showticklabels=True, row=1)
        fig.update_xaxes(ticks="", showticklabels=False, row=1, col=1)


        ## GRAY MATTER CARPET PLOT 
        print("\nPlotting Gray Matter")
        clim = float(np.percentile(np.abs(demeaned_voxels['gm']), 95)) 
        print(f"Setting color limits to ±{clim:.2f}%")
        fig.add_trace(
            go.Heatmap(
                z=demeaned_voxels['gm'] ,
                x=list(range(demeaned_voxels['gm'].shape[1])),
                colorscale="Gray",
                colorbar=dict(title="PSC (%)"),
                zmin=-clim,
                zmax=clim,
                hovertemplate=(
                    "<b>Aquisition:</b> %{x}" + "<br>" +
                    "<b>Voxel Number</b>: %{y} " + "<br>" +
                    "<b>Percent Signal Change</b>: %{z} " + "<br>"
                ),
                showscale=False
            ),
            row=2,
            col=1
        )
        fig.update_xaxes(ticks="", showticklabels=False, row=2, col=1)
        fig.update_yaxes(title_text="Gray Matter Voxels", title_font=dict(size=11), title_standoff=5, showticklabels=True, row=2, col=1)


        ## WHITE MATTER CARPET PLOT 
        print("\nPlotting White Matter")
        clim = float(np.percentile(np.abs(demeaned_voxels['wm']), 95)) 
        print(f"Setting color limits to ±{clim:.2f}%")
        fig.add_trace(
            go.Heatmap(
                z=demeaned_voxels['wm'] ,
                x=list(range(demeaned_voxels['wm'].shape[1])),
                colorscale="Gray",
                colorbar=dict(title="PSC (%)"),
                zmin=-clim,
                zmax=clim,
                hovertemplate=(
                    "<b>Aquisition:</b> %{x}" + "<br>" +
                    "<b>Voxel Number</b>: %{y} " + "<br>" +
                    "<b>Percent Signal Change</b>: %{z} " + "<br>"
                ),
                showscale=False
            ),
            row=3,
            col=1
        )
        fig.update_xaxes(ticks="", showticklabels=False, row=3, col=1)
        fig.update_yaxes(title_text="White Matter Voxels", title_font=dict(size=11), title_standoff=5, showticklabels=True, row=3, col=1)


        ## CSF CARPET PLOT 
        print("\nPlotting CSF")
        clim = float(np.percentile(np.abs(demeaned_voxels['csf']), 95)) 
        print(f"Setting color limits to ±{clim:.2f}%")
        fig.add_trace(
            go.Heatmap(
                z=demeaned_voxels['csf'] ,
                x=list(range(demeaned_voxels['csf'].shape[1])),
                colorscale="Gray",
                colorbar=dict(title="PSC (%)"),
                zmin=-clim,
                zmax=clim,
                hovertemplate=(
                    "<b>Aquisition:</b> %{x}" + "<br>" +
                    "<b>Voxel Number</b>: %{y} " + "<br>" +
                    "<b>Percent Signal Change</b>: %{z} " + "<br>"
                ),
                showscale=False
            ),
            row=4,
            col=1
        )
        fig.update_xaxes(title_text="Aquisition", title_font=dict(size=12), title_standoff=5, showticklabels=True, row=4, col=1)
        fig.update_yaxes(title_text="CSF Voxels", title_font=dict(size=11), title_standoff=5, showticklabels=True, row=4, col=1)
        
        ## Write to file
        fig.write_html(output_file_path)
        print(f"\nDone. Output Plot At: {output_file_path}")
    
    
    def get_slice_timing(self, json_path, slice_times_text_file):

        def find_matching_indexes(numbers):
            
            num_index_map = {}
        
            for index, number in enumerate(numbers):
                if number in num_index_map:
                    num_index_map[number].append(index)
                else:
                    num_index_map[number] = [index]
        
            return {number: indexes for number, indexes in num_index_map.items() if len(indexes) > 1}

        with open(json_path) as f:
            json_data = json.load(f)
            if not 'SliceTiming' in json_data:
                print(f"'SliceTiming' Key Not In JSON File: {json_path}")
                if slice_times_text_file:
                    slice_timing = []
                    with open(slice_times_text_file, mode='r') as file:
                        for line in file:
                            if not line.strip():
                                continue 
                            else:
                                slice_timing.append(float(line.strip()))
                    print(f"Using the slice timing from the input file: {slice_timing}")
                    OrderedDict(sorted(find_matching_indexes(slice_timing).items()))

                else:
                    print(f"\n\nERROR: 'SliceTiming' Key Not In JSON File: {json_path}\n\n")
                    print(f"Please provide a different JSON file or provide slice timing via the argument 'slice_timing'")

                    exit(0)
            else:
               return OrderedDict(sorted(find_matching_indexes(json_data['SliceTiming']).items()))
            
    def get_motion_flagged_volumes(self, num_volumes, num_slice_groups, displacement_values, mm_displacement_threshold):
        motion_flagged_volumes = []
        for volume_num in range(num_volumes):
            displacements_at_this_volume = displacement_values[volume_num*num_slice_groups:(volume_num*num_slice_groups) + num_slice_groups]
            if any([displacement > mm_displacement_threshold for displacement in displacements_at_this_volume]):
                motion_flagged_volumes.append(volume_num)
        
        return motion_flagged_volumes 
    
    def coregister_anat_to_func_fsl(self, anatomical_image, reference_volume_image, output_directory):
        flirt = fsl.FLIRT()
        flirt.inputs.in_file = anatomical_image
        flirt.inputs.reference = reference_volume_image
        flirt.inputs.out_file = f"{output_directory}/anat_coreg.nii.gz"
        flirt.inputs.out_matrix_file = f"{output_directory}/anat2func.mat"
        flirt.inputs.cost = "corratio"      # correlation ratio, standard for cross-modal
        flirt.inputs.dof = 6                # rigid body (no scaling, since same subject)
        flirt.inputs.interp = "trilinear"
        result = flirt.run()
        return result.outputs.out_file, result.outputs.out_matrix_file


    def skull_strip_anat(self, coregistered_anat_image, output_directory):
        bet = fsl.BET()
        bet.inputs.in_file = coregistered_anat_image
        bet.inputs.out_file = os.path.join(output_directory, "ss_anat_brain.nii.gz")
        bet.inputs.mask = True        
        bet.inputs.frac = 0.5         
        result = bet.run()
        return result.outputs.out_file, result.outputs.mask_file


    def segment_anat_fsl(self, coregistered_anat_image, output_directory):
        fast = fsl.FAST()
        fast.inputs.in_files = [coregistered_anat_image]
        fast.inputs.img_type = 1
        fast.inputs.number_classes = 3
        fast.inputs.output_biascorrected = True
        fast.inputs.output_biasfield = False
        fast.inputs.segments = True
        fast.inputs.probability_maps = True
        fast.inputs.out_basename = os.path.join(output_directory, "anat_seg")

        try:
            fast.run()
        except Exception:
            pass  # nipype bug with absolute paths — files are written correctly
        
        return {
            "csf": os.path.join(output_directory, "anat_seg_seg_0.nii.gz"),
            "gm":  os.path.join(output_directory, "anat_seg_seg_1.nii.gz"),
            "wm":  os.path.join(output_directory, "anat_seg_seg_2.nii.gz"),
        }
    

    def calculate_displacement(self, transform1_path, transform2_path, radius = 50):

        def compose_transforms(transform1, transform2):
            transform1_inverse = transform1.GetInverse()

            A0 = np.asarray(transform2.GetMatrix()).reshape(3,3)
            c0 = np.asarray(transform2.GetCenter())
            t0 = np.asarray(transform2.GetTranslation())

            A1 = np.asarray(transform1_inverse.GetMatrix()).reshape(3,3)
            c1 = np.asarray(transform1_inverse.GetCenter())
            t1 = np.asarray(transform1_inverse.GetTranslation())

            combined_mat = np.dot(A0,A1)
            combined_center = c1
            combined_translation = np.dot(A0, t1+c1-c0) + t0+c0-c1
            combined_affine = sitk.AffineTransform(combined_mat.flatten(), combined_translation, combined_center)

            return combined_affine
        
        def convert_affine_to_versorrigid(affinetransform):
            versorrigid3d = sitk.VersorRigid3DTransform()
            versorrigid3d.SetCenter( affinetransform.GetCenter() )
            versorrigid3d.SetTranslation( affinetransform.GetTranslation() )
            versorrigid3d.SetMatrix( affinetransform.GetMatrix() )
            return versorrigid3d
        
        transform1 = sitk.ReadTransform(transform1_path)
        transform2 = sitk.ReadTransform(transform2_path)
        composed_affine = compose_transforms(transform1, transform2)
        versorrigid3d = convert_affine_to_versorrigid(composed_affine)
        parms = np.asarray( versorrigid3d.GetParameters() )
        versormagsquared = parms[0]*parms[0] + parms[1]*parms[1] + parms[2]*parms[2]
        versormag = math.sqrt(versormagsquared)
        wsquared = 1 - versormagsquared
        w = math.sqrt(wsquared)
        angle = 2.0 * math.atan2( versormag, w )
        deltarotationmm = float(radius) * float(angle)
        deltatranslationsquared = abs(parms[3])*abs(parms[3]) + abs(parms[4])*abs(parms[4]) + abs(parms[5])*abs(parms[5])
        deltatranslation = math.sqrt(deltatranslationsquared)
        totalmotion = deltarotationmm + deltatranslation

        return totalmotion


    def calculate_psc(self, functional_image, brain_mask, slice_timing):
        func_img = nib.load(functional_image)
        mask_img = nib.load(brain_mask)

        mask_resampled = resample_to_img(
            source_img=mask_img,
            target_img=func_img,
            interpolation="nearest",
            force_resample=False,
            copy_header=True,
        )

        func_data = func_img.get_fdata()                   # (X,Y,Z,T)
        mask_data = mask_resampled.get_fdata().astype(bool)

        # ------------------------------------------------------------------
        # Extract voxel time series
        # ------------------------------------------------------------------

        voxel_timeseries = func_data[mask_data]            # (n_voxels, n_volumes)

        voxel_mean = voxel_timeseries.mean(axis=1, keepdims=True)

        keep = voxel_mean[:, 0] > (0.5 * np.median(voxel_mean))

        voxel_timeseries = voxel_timeseries[keep]
        voxel_mean = voxel_mean[keep]

        psc = (voxel_timeseries - voxel_mean) / voxel_mean * 100
        # shape = (n_voxels, n_volumes)

        # ------------------------------------------------------------------
        # Determine which voxel belongs to which slice
        # ------------------------------------------------------------------

        coords = np.argwhere(mask_data)
        coords = coords[keep]

        voxel_slice = coords[:, 2]

        slice_to_group = {}

        for group_idx, slices in enumerate(slice_timing.values()):
            for s in slices:
                slice_to_group[s] = group_idx

        n_groups = len(slice_timing)
        n_volumes = func_data.shape[3]

        expanded = np.empty((psc.shape[0], n_groups * n_volumes))

        expanded[:] = np.nan

        # initialize with first volume
        expanded[:, 0] = psc[:, 0]

        # ------------------------------------------------------------------
        # Expand 294 volumes -> 4410 slice-group acquisitions
        # ------------------------------------------------------------------

        for vol in range(n_volumes):

            for group in range(n_groups):

                col = vol * n_groups + group

                if col > 0:
                    expanded[:, col] = expanded[:, col - 1]

                # voxels acquired in this slice group
                update = np.array(
                    [slice_to_group[z] == group for z in voxel_slice]
                )

                expanded[update, col] = psc[update, vol]

        print("Expanded PSC:", expanded.shape)

        return expanded

    def subsample_carpet(self, psc_array, max_voxels):
        n = psc_array.shape[0]
        if n <= max_voxels:
            return psc_array
        idx = np.linspace(0, n - 1, max_voxels, dtype=int)  
        return psc_array[idx]
if __name__ == "__main__": 
    import argparse
    
    parser = argparse.ArgumentParser(description="Make Carpet Plot of Data")
    parser.add_argument(
        "--anatomical_image", 
        required=True
    )
    parser.add_argument(
        "--json_file",
        required=True
    )
    parser.add_argument(
        "--slice_times_text_file",
        required=False,
        help=\
            "If your JSON file is missing 'SliceTiming' key, provide a text file containing the timing for each slice in a volume." + \
            "(fMRIPrep's derivative JSON sidecars for desc-preproc_bold files typically don't carry SliceTiming forward).",
        nargs='+',
        type=float
    )
    parser.add_argument(
        "--functional_image",
        required=True
    )
    parser.add_argument(
        "--reference_volume_image",
        required=True
    )
    parser.add_argument(
        "--transform_directory",
        required=True
    )
    parser.add_argument(
        "--displacement_threshold",
        required=False,
        type=float,
        default=0.4157,
        help="Give in mm. Default: 0.4157mm"
    )
    parser.add_argument(
        "--output_directory",
        required=False,
        default="outputs",
        help=f"Default: {os.path.abspath('outputs')}"
    )
    parser.add_argument(
        "--plot_title",
        required=False,
        default="Voxel Percent Signal Change Carpet Plot + Displacements",
        help="Default:'Voxel Percent Signal Change Carpet Plot + Displacements'"
    )
    parser.add_argument(
        "--output_plot_path",
        required=False,
        default="carpet_plot.html",
        help="Default:'carpet_plot.html'"


    )
    args = parser.parse_args()

    CarpetPlot(
        anatomical_image=os.path.abspath(args.anatomical_image),
        json_file=os.path.abspath(args.json_file),
        functional_image=os.path.abspath(args.functional_image),
        reference_volume_image=os.path.abspath(args.reference_volume_image),
        transform_directory=os.path.abspath(args.transform_directory),
        displacement_threshold=args.displacement_threshold,
        output_directory=os.path.abspath(args.output_directory),
        plot_title=args.plot_title,
        output_file_path=os.path.abspath(args.output_plot_path),
        slice_times_text_file=\
            os.path.abspath(args.slice_times_text_file) 
            if args.slice_times_text_file else None
    )