# Intravolume Motion Correction Pipeline

#### [Meghan Walsh](https://meghanw23.github.io/portfolio/) | [wmeghan0809@gmail.com](mailto:wmeghan0809@gmail.com)

## Project Overview
Pharmacological fMRI (phMRI) is the primary tool for understanding how stimulant medications work in the brain to treat ADHD. However, the findings across studies are strikingly inconsistent. Stimulants reduce hyperactivity, which reduces head movement in the scanner. This means that "off vs. on medication" comparisons likely capture differences in motion artifact as much as, or more than, real changes in brain activity. The field has acknowledged this as a limitation but no prior study has measured it or corrected for it.

This project directly addresses the motion confound using this novel slice-by-slice motion correction pipeline. Standard correction methods treat each MRI volume as if it were captured in a single instant, but participants can move within a volume acquisition, and in ADHD, they frequently do. Our algorithm corrects motion at the level of individual slice groups rather than whole volumes, capturing within-volume motion that standard methods miss. By comparing findings with and without this superior correction, we can separate true treatment effects from motion artifacts and begin to identify accurate neuroanatomical targets for ADHD intervention.

#### [Computational Radiology Lab](http://crl.med.harvard.edu/) | [Cohen Laboratory of Translational Neuroimaging](https://bchcohenlab.com/)

----

## Set up the Pipeline

***Note to Cohen Lab Researchers**: This repository is already existing and configured on E3 at: `/lab-share/Neuro-Cohen-e2/Groups/IRB-P00049401/intravolume_motion_pipeline`. You will need access to the Neurofeedback project PHI folder (`Groups/P00049401`) to access the repo.*


#### Clone the Repository
This project has submodules. *You will need to request access to any submodules that are currently private*.
To clone the project with submodules, given you have access to them, run:
```
# If connecting to GitHub with SSH 
git clone --recurse-submodules git@github.com:MeghanW23/intravolume_motion_pipeline_v2.git

# If connecting to GitHub over HTTPS
git clone --recurse-submodules https://github.com/MeghanW23/intravolume_motion_pipeline_v2.git
```

#### Edit TEMPLATE.yaml software paths
You will also need to edit the various software paths in the [run_pipeline/config_files/TEMPLATE.yaml](https://github.com/MeghanW23/intravolume_motion_pipeline_v2/blob/main/run_pipeline/config_files/TEMPLATE.yaml) file so that it points to the correct paths on your system.

#### Ensure you are running on an HPC
2. This pipeline is memory-intensive and long-running, so it is important to run on an [HPC](https://www.intel.com/content/www/us/en/learn/what-is-hpc.html) instead of locally.

#### Create the Conda environment
3. You will need to create a Conda (or potentially Python) environment to install the necessary packages. There is a environment.yml file you can use at `run_pipeline/environment.yml`. Once you have created the environment, don't forget to edit the following paths in your .yaml file(s):
  CONDA_ENV_NAME
  CONDA_INIT_PATH
  CONDA_ENV_PYTHON_PATH

----

## How to Run the Pipeline

#### 1. Create a configuration file
Create a config file for your data by making a copy of [run_pipeline/config_files/TEMPLATE.yaml](https://github.com/MeghanW23/intravolume_motion_pipeline_v2/blob/main/run_pipeline/config_files/TEMPLATE.yaml). These config files are used to tell the pipeline what data to use, what steps to run, and what configurations to use. For additional information on how to use a config file, there are notes in the TEMPLATE.yaml file.

#### 2. Run the pipeline
[run_pipeline/submit_job.sh](https://github.com/MeghanW23/intravolume_motion_pipeline_v2/blob/main/run_pipeline/submit_job.sh) starts the pipeline as a [Slurm](https://slurm.schedmd.com/overview.html) batch job. It receives one config file as a command line input and runs the pipeline wrapper script `run_pipeline/run_pipeline.py`. Pass the config file to `submit_job.sh` and start a slurm job like this:
```
sbatch submit_job.sh config_files/<your config file>
```
or you can run the main python script interactively (on a system with high memory / CPU requirements) using the main pipeline script: 

```
python run_pipeline.py --configuration_file config_files/<your config file>
```
