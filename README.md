# Intravolume Motion Correction Pipeline

#### [Meghan Walsh](https://meghanw23.github.io/portfolio/) | [wmeghan0809@gmail.com](mailto:wmeghan0809@gmail.com)

## Project Overview
Pharmacological fMRI (phMRI) is the primary tool for understanding how stimulant medications work in the brain to treat ADHD. However, the findings across studies are strikingly inconsistent. Stimulants reduce hyperactivity, which reduces head movement in the scanner. This means that "off vs. on medication" comparisons likely capture differences in motion artifact as much as, or more than, real changes in brain activity. The field has acknowledged this as a limitation but no prior study has measured it or corrected for it.

This project directly addresses the motion confound using this novel slice-by-slice motion correction pipeline. Standard correction methods treat each MRI volume as if it were captured in a single instant, but participants can move within a volume acquisition, and in ADHD, they frequently do. Our algorithm corrects motion at the level of individual slice groups rather than whole volumes, capturing within-volume motion that standard methods miss. By comparing findings with and without this superior correction, we can separate true treatment effects from motion artifacts and begin to identify accurate neuroanatomical targets for ADHD intervention.

#### [Computational Radiology Lab](http://crl.med.harvard.edu/) | [Cohen Laboratory of Translational Neuroimaging](https://bchcohenlab.com/)

----

## How to Set Up the Pipeline
#### Notes:
- This pipeline is memory-intensive and long-running, so it is important to run on an [HPC](https://www.intel.com/content/www/us/en/learn/what-is-hpc.html) instead of locally.
- You will need MATLAB installed for this pipeline. 

***Note to Cohen Lab Researchers**: This repository is already existing and configured on E3 at: `/lab-share/Neuro-Cohen-e2/Groups/IRB-P00049401/intravolume_motion_pipeline`. You will need access to the Neurofeedback project PHI folder (`Groups/P00049401`) to access the repo.*

### 1. Clone the Repository

This project has submodules. *You will need to request access to any submodules that are currently private*.
To clone the project with submodules, given you have access to them, run:
```
# If connecting to GitHub with SSH 
git clone --recurse-submodules git@github.com:MeghanW23/intravolume_motion_pipeline.git
```
```
# If connecting to GitHub over HTTPS
git clone --recurse-submodules https://github.com/MeghanW23/intravolume_motion_pipeline.git
```
### 2. Edit TEMPLATE.yaml software paths
You will need to edit the various software paths in the [`run_pipeline/config_files/TEMPLATE.yaml`](https://github.com/MeghanW23/intravolume_motion_pipeline/blob/main/run_pipeline/config_files/TEMPLATE.yaml) file so that it points to the correct paths on your system. Currently, the paths are set up for E3.
### 3. Create the Conda environment
You will need to create a Conda (or potentially Python) environment to install the necessary packages. There is a environment.yml file you can use at [`run_pipeline/environment.yml`](https://github.com/MeghanW23/intravolume_motion_pipeline/blob/main/run_pipeline/environment.yml). Once you have created the environment, don't forget to edit the following paths in your .yaml file(s):
<br>
<br>`CONDA_ENV_NAME`<br>
<br>`CONDA_INIT_PATH`<br>
<br>`CONDA_ENV_PYTHON_PATH`
### 4. Pull and/or Compile the Optimizer
Our optimizer, [`retro-motion-measurement`](https://github.com/ComputationalRadiology/sms-mi-reg/blob/main/retro-motion-measurement.cxx), characterizes the movement between a reference volume and each slice group acquisition via a [6-Dimension Rigid Body Transform](https://medium.com/@parkie0517/rigid-transformation-in-3d-space-translation-and-rotation-d701d8859ba8).
You can install the software via any of the three following options:

- #### Option One - Build the Docker Image:
  - The `crl/sms-mi-reg` Docker image contains the all the optimizer software and it's dependencies. Build the Docker container by running the [`build-docker.sh`](https://github.com/ComputationalRadiology/sms-mi-reg/blob/main/build-docker.sh) file:
    ```
    cd scripts/MotionCharacterization/sms-mi-reg/
    sudo chmod +x build-docker.sh
    sudo ./build-docker.sh
    ```
  - You could also follow the steps listed in [`scripts/MotionCharacterization/pull_docker_container.sh`](https://github.com/MeghanW23/intravolume_motion_pipeline/blob/main/scripts/MotionCharacterization/pull_docker_container.sh) to pull the container from the Computational Radiology Lab's Container Repository.
    - You will need access to the Container Repository.
    - Please ensure this image is the most recent version of the image.

- #### Option Two - Compile the CPP Code on Directly on the Host Machine:
  - Follow the steps in the [`scripts/MotionCharacterization/compile_sms-mi-reg.sh`](https://github.com/MeghanW23/intravolume_motion_pipeline/blob/main/scripts/MotionCharacterization/compile_sms-mi-reg.sh) script.
  - You will need to have [ITK](https://docs.itk.org/en/latest/download.html) installed. See: "Create the Conda environment" above.

- #### Option Three - Build the Singularity Image by Converting the Docker Image to a `.sif` file:
  - On many HPCs, including BCH's E3, you will not have `sudo` access. In these cases, many developers opt to use [Singularity](https://docs.sylabs.io/guides/2.6/user-guide/introduction.html), which does not require `sudo` privileges, instead of Docker.
  - The recommended workflow for getting the Singularity image includes:
    1. Pulling the Docker container onto a machine where you have docker access (your local machine, another server etc.), then
    2. Converting the Docker image to a Singularity image on your HPC.
  - An example of this pipeline can be found at [`scripts/MotionCharacterization/docker-to-sif_steps-for-e3.sh`](https://github.com/MeghanW23/intravolume_motion_pipeline/blob/main/scripts/MotionCharacterization/docker-to-sif_steps-for-e3.sh)
  - It is recommended to run the Docker to Singularity conversion in a batch job / high-resource environment. An example of a Slurm batch script to use can be found at: [`scripts/MotionCharacterization/docker-to-sif_batch-script.sh`](https://github.com/MeghanW23/intravolume_motion_pipeline/blob/main/scripts/MotionCharacterization/docker-to-sif_batch-script.sh)
  - *Make sure you are pulling the Docker image with the architecture of the machine you want to run the singularity image on.*

**Once you have pulled or built the optimizer, you will need to the following variables in your .yaml file(s)**
- `OPTIMIZER_RUN_ENVIRONMENT`: Enter the optimizer option you have chosen. Input: 'singularity', 'docker', or 'local'.
- `OPTIMIZER_SINGULARITY_IMAGE_PATH`: If using a Singularity image, please provide the Singularity .sif path here.
- `OPTIMIZER_EXECUTABLE_PATH`: If using the compiled `retro-motion-measurement.cxx` program, please provide the path to the compiled CPP code here.
If you are opting to use a Docker image, you will only need to set `OPTIMIZER_RUN_ENVIRONMENT` to 'docker'.

----

## How to Run the Pipeline

#### 1. Create a configuration file
Create a config file for your data by making a copy of [`run_pipeline/config_files/TEMPLATE.yaml`](https://github.com/MeghanW23/intravolume_motion_pipeline/blob/main/run_pipeline/config_files/TEMPLATE.yaml). These config files are used to tell the pipeline what data to use, what steps to run, and what configurations to use. For additional information on how to use a config file, there are notes in the TEMPLATE.yaml file.

#### 2. Run the pipeline
[`run_pipeline/submit_job.sh`](https://github.com/MeghanW23/intravolume_motion_pipeline/blob/main/run_pipeline/submit_job.sh) starts the pipeline as a [Slurm](https://slurm.schedmd.com/overview.html) batch job. It receives one config file as a command line input and runs the pipeline wrapper script `run_pipeline/run_pipeline.py`. Pass the config file to `submit_job.sh` and start a slurm job like this:
```
sbatch submit_job.sh config_files/<your config file>
```
or you can run the pipeline wrapper script (on a system with high memory / CPU resources) using the main pipeline script: 

```
python run_pipeline.py --configuration_file config_files/<your config file>
```
