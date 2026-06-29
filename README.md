# Intra-Volume Motion Correction Pipeline

### Project Overview
Pharmacological fMRI (phMRI) is the primary tool for understanding how stimulant medications work in the brain to treat ADHD. However, the findings across studies are strikingly inconsistent. A key reason may be hiding in plain sight: stimulants reduce hyperactivity, which reduces head movement in the scanner. This means that "off vs. on medication" comparisons likely capture differences in motion artifact as much as, or more than, real changes in brain activity. The field has acknowledged this as a limitation but no prior study has measured it or corrected for it.

This project directly addresses the motion confound using this novel slice-by-slice motion correction pipeline. Standard correction methods treat each MRI volume as if it were captured in a single instant, but participants can move within a volume acquisition, and in ADHD, they frequently do. Our algorithm corrects motion at the level of individual slice groups rather than whole volumes, capturing within-volume jitter that standard methods miss. By comparing findings with and without this superior correction, we can separate true treatment effects from motion artifacts and begin to identify accurate neuroanatomical targets for ADHD intervention.

This project was developed in collaboration with the [Computational Radiology Lab](http://crl.med.harvard.edu/). 

### How to Run The Pipeline

1. Create a config file for your data by making a copy of [run_pipeline/config_files/TEMPLATE_CONFIG.env](https://github.com/MeghanW23/intravolume_motion_pipeline/blob/master/run_pipeline/config_files/TEMPLATE_CONFIG.env). These config files are used to point to the input data and pipeline configurations. For additional information on how to use a config file, there are notes in the TEMPLATE_CONFIG.env file.
2. [run_pipeline/submit_job.sh](https://github.com/MeghanW23/intravolume_motion_pipeline/blob/master/run_pipeline/submit_job.sh) starts the pipeline. It receives one or more config files as command line inputs. Pass the config file(s) to submit_job.sh and start a slurm job like this:
```
sbatch submit_job.sh <your config file(s)>
```
*Currently, it is much faster to run 1 job per single-run data. I recommend submitting a job per run, then running fMRIPrep separately afterwards.*

### About This Repository
*Please note that this pipeline is designed to be run on Boston Children's HPC cluster E3.*
This repository contains the pipeline for correcting slice-by-slice (intravolume) motion in [SMS Accelerated](https://pmc.ncbi.nlm.nih.gov/articles/PMC4915494/) fMRI data obtained from prospective studies conducted by the [Cohen Lab](https://bchcohenlab.com/). This pipeline can be broken down into 2 main steps:

#### 1. Motion Characterization
We must first characterize the slice-by-slice movement before we can correct for it. We characterize each slice group's motion via a 3D Rigid Body Transform detailing the movement, in 6 dimensions, from a 3D reference volume. This alignment is powered via [Computational Radiology Lab](http://crl.med.harvard.edu/)'s [Sms-Mi-Reg Optimizer](https://github.com/ComputationalRadiology/sms-mi-reg). 

We first use [run_pipeline/pipeline_scripts/get_reference_volume.py](https://github.com/MeghanW23/intravolume_motion_pipeline/blob/master/run_pipeline/pipeline_scripts/get_reference_volume.py) to select a motion-free reference volume. We then align each slice group in the timeseries to the reference volume via [run_pipeline/pipeline_scripts/motion_characterization.py](https://github.com/MeghanW23/intravolume_motion_pipeline/blob/master/run_pipeline/pipeline_scripts/motion_characterization.py).

For a step-by-step guide on what the motion characterization script does, please see the [PDF Guide](https://github.com/MeghanW23/intravolume_motion_pipeline/blob/master/other/motion-characterization-step-by-step.pdf).

#### 2. Motion Correction
Conventionally, motion-corrupted volumes are censored from fMRI time series. Where this censoring can result in discontinuities in the fMRI signal, which may lead to substantial alterations in fMRI analysis. Instead, we utilize a [Structured Matrix Completion Approach](https://pubmed.ncbi.nlm.nih.gov/34432631/), created by the [Computational Radiology Lab](http://crl.med.harvard.edu/), where we recover the missing entries from censoring based on structured low rank matrix completion.

The source code for the Structured Matrix Completion Approach can be found in the [rsfMRI_SMC_mc-ORIGINAL_CODE](https://github.com/bchimagine/rsfMRI_SMC_mc/tree/abe415496bc38fc7d590e49ff7a435117b0f97ec) directory. The code I edited for this pipeline can be found in the [rsfMRI_SMC_mc](https://github.com/MeghanW23/intravolume_motion_pipeline/tree/master/rsfMRI_SMC_mc) directory.

#### After running both motion correction and characterization, you will have the option to run fMRIPrep on the motion corrected NiFTI Timeseries.

### Pipeline Steps
1. [run_pipeline/submit_job.sh](https://github.com/MeghanW23/intravolume_motion_pipeline/blob/master/run_pipeline/submit_job.sh): Submits the job to slurm and runs the run_pipeline.py script. Takes in 1 or more config file(s).<br><br>
2. [run_pipeline/pipeline_scripts/run_pipeline.py](https://github.com/MeghanW23/intravolume_motion_pipeline/blob/master/run_pipeline/pipeline_scripts/run_pipeline.py): This script validates the inputs in the config file(s) and acts as a wrapper for each of the follows steps/scripts.<br><br>
3. [run_pipeline/pipeline_scripts/dicom_to_nifti.py](https://github.com/MeghanW23/intravolume_motion_pipeline/blob/master/run_pipeline/pipeline_scripts/dicom_to_nifti.py): Runs only if a DICOM Directory is given. It decompresses the DICOMS and runs dcm2niix.<br><br>
4. [run_pipeline/pipeline_scripts/motion_characterization.py](https://github.com/MeghanW23/intravolume_motion_pipeline/blob/master/run_pipeline/pipeline_scripts/motion_characterization.py): Characterizes motion in 6 dimensions for every aquisition. Unless a reference volume is listed in the config file, it will select a reference volume via [run_pipeline/pipeline_scripts/get_reference_volume.py](https://github.com/MeghanW23/intravolume_motion_pipeline/blob/master/run_pipeline/pipeline_scripts/get_reference_volume.py). It is very important a motion-free reference volume is selected and motion may not be visible to the naked eye. For a step-by-step guide on what the motion characterization script does, please see the [PDF Guide](https://github.com/MeghanW23/intravolume_motion_pipeline/blob/master/other/motion-characterization-step-by-step.pdf).<br><br>
5. [run_pipeline/pipeline_scripts/graph_transforms.py](https://github.com/MeghanW23/intravolume_motion_pipeline/blob/master/run_pipeline/pipeline_scripts/graph_transforms.py): This script graphs the outputs of motion_characterization.py. It also calculates displacements between each transform and the next, and outputs a parameters.txt and displacement.txt files with the values. The file parameters.txt will be inputted to the motion correction software.<br><br>
6. [run_pipeline/pipeline_scripts/remove_background.py](https://github.com/MeghanW23/intravolume_motion_pipeline/blob/master/run_pipeline/pipeline_scripts/remove_background.py): The motion correction software requires both the raw NiFTI data and the NiFTI data with the background removed. This script strips the background. *Please assure it has not stripped more than the background.*<br><br>
7. [run_pipeline/pipeline_scripts/start_motion_correction.py](https://github.com/MeghanW23/intravolume_motion_pipeline/blob/master/run_pipeline/pipeline_scripts/start_motion_correction.py): the motion correction software is mainly written in MATLAB. This python script starts the main MATLAB script.<br><br>
8. [run_pipeline/pipeline_scripts/run_fmriprep.py](https://github.com/MeghanW23/intravolume_motion_pipeline/blob/master/run_pipeline/pipeline_scripts/run_fmriprep.py): If you are running fMRIPrep, this script prepares the fmriprep directories and materials and starts fMRIPrep.
