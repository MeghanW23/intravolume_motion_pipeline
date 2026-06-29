# Intra-Volume Motion Correction Pipeline

### Project Overview
Pharmacological fMRI (phMRI) is the primary tool for understanding how stimulant medications work in the brain to treat ADHD. However, the findings across studies are strikingly inconsistent. A key reason may be hiding in plain sight: stimulants reduce hyperactivity, which reduces head movement in the scanner. This means that "off vs. on medication" comparisons likely capture differences in motion artifact as much as, or more than, real changes in brain activity. The field has acknowledged this as a limitation but no prior study has measured it or corrected for it.

This project directly addresses the motion confound using this novel slice-by-slice motion correction pipeline. Standard correction methods treat each MRI volume as if it were captured in a single instant, but participants can move within a volume acquisition, and in ADHD, they frequently do. Our algorithm corrects motion at the level of individual slice groups rather than whole volumes, capturing within-volume jitter that standard methods miss. By comparing findings with and without this superior correction, we can separate true treatment effects from motion artifacts and begin to identify accurate neuroanatomical targets for ADHD intervention.

### About this Repository
This repository contains the pipeline for correcting slice-by-slice (intravolume) motion in SMS Accelerated fMRI data
in the [Cohen Lab](https://bchcohenlab.com/)'s ADHD and ASD Cohorts. This pipeline can be broken down into 2 main steps:
<br>

**STEP ONE: Motion Characterization**: We must first characterize the slice-by-slice movement before we can correct for it. We characterize each slice group's motion via a 3D Rigid Body Transform detailing the movement, in 6 dimensions, from a 3D reference volume. This alignment is powered via Computational Radiology Lab's [Sms-Mi-Reg Optimizer](https://github.com/ComputationalRadiology/sms-mi-reg). 

We first use the [Reference Volume Selection Script](https://github.com/MeghanW23/intravolume_motion_pipeline/blob/master/run_pipeline/pipeline_scripts/get_reference_volume.py) to select a motion-free reference volume. We then align each slice group in the timeseries to the reference volume via the [Motion Characterization Script](https://github.com/MeghanW23/intravolume_motion_pipeline/blob/master/run_pipeline/pipeline_scripts/motion_characterization.py).

**STEP TWO: Motion Correction**: Conventionally, motion-corrupted volumes are censored from fMRI time series. Where this censoring can result in discontinuities in the fMRI signal, which may lead to substantial alterations in fMRI analysis. Instead, we utilize a [Structured Matrix Completion Approach](https://pubmed.ncbi.nlm.nih.gov/34432631/), where we recover the missing entries from censoring based on structured low rank matrix completion.

The source code for the Structured Matrix Completion Approach can be found in the [rsfMRI_SMC_mc-ORIGINAL_CODE](https://github.com/bchimagine/rsfMRI_SMC_mc/tree/abe415496bc38fc7d590e49ff7a435117b0f97ec) directory. The code I edited for this pipeline can be found in the [rsfMRI_SMC_mc](https://github.com/MeghanW23/intravolume_motion_pipeline/tree/master/rsfMRI_SMC_mc) directory.

**After running both motion correction and characterization, you will have the option to run fMRIPrep on the motion corrected NiFTI Timeseries.**