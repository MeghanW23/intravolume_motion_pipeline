# Intra-Volume Motion Correction Pipeline

### Project Overview
Pharmacological fMRI (phMRI) is the primary tool for understanding how stimulant medications work in the brain to treat ADHD. However, the findings across studies are strikingly inconsistent. Stimulants reduce hyperactivity, which reduces head movement in the scanner. This means that "off vs. on medication" comparisons likely capture differences in motion artifact as much as, or more than, real changes in brain activity. The field has acknowledged this as a limitation but no prior study has measured it or corrected for it.

This project directly addresses the motion confound using this novel slice-by-slice motion correction pipeline. Standard correction methods treat each MRI volume as if it were captured in a single instant, but participants can move within a volume acquisition, and in ADHD, they frequently do. Our algorithm corrects motion at the level of individual slice groups rather than whole volumes, capturing within-volume motion that standard methods miss. By comparing findings with and without this superior correction, we can separate true treatment effects from motion artifacts and begin to identify accurate neuroanatomical targets for ADHD intervention.

This project was developed in collaboration with the [Computational Radiology Lab](http://crl.med.harvard.edu/). 

### How to Run The Pipeline

1. Create a config file for your data by making a copy of [run_pipeline/config_files/TEMPLATE.yaml](https://github.com/MeghanW23/intravolume_motion_pipeline_v2/blob/main/run_pipeline/config_files/TEMPLATE.yaml). These config files are used to tell the pipeline what data to use, what steps to run, and what configurations to use. For additional information on how to use a config file, there are notes in the TEMPLATE.yaml file.
2. [run_pipeline/submit_job.sh](https://github.com/MeghanW23/intravolume_motion_pipeline_v2/blob/main/run_pipeline/submit_job.sh) starts the pipeline as a [Slurm](https://slurm.schedmd.com/overview.html) batch job. It receives one config file as a command line input. Pass the config file to submit_job.sh and start a slurm job like this:
```
sbatch submit_job.sh <your config file>
```

This project has submodules. To clone the project with submodules, given you have acecss to them, run:
```
git clone --recurse-submodules git@github.com:MeghanW23/intravolume_motion_pipeline_v2.git
```
