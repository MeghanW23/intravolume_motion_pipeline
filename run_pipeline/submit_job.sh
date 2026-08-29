#!/bin/bash
#SBATCH --partition=bch-compute
#SBATCH --time=24:00:00
#SBATCH --job-name=intravolume_motion_pipeline
#SBATCH --output=logs/output_%j.out
#SBATCH --error=logs/output_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=meghan.walsh2@childrens.harvard.edu

# set -euo pipefail

config_file=$1
echo "Inputted File: ${config_file}"

# Load needed paths from .yaml file
BIOGRIDS_PATH=$(python3 -c "import yaml, sys; print(yaml.safe_load(open(sys.argv[1]))['BIOGRIDS_PATH'])" ${config_file})
CONDA_ENV_NAME=$(python3 -c "import yaml, sys; print(yaml.safe_load(open(sys.argv[1]))['CONDA_ENV_NAME'])" ${config_file})
CONDA_INIT_PATH=$(python3 -c "import yaml, sys; print(yaml.safe_load(open(sys.argv[1]))['CONDA_INIT_PATH'])" ${config_file})
CONDA_ENV_PYTHON_PATH=$(python3 -c "import yaml, sys; print(yaml.safe_load(open(sys.argv[1]))['CONDA_ENV_PYTHON_PATH'])" ${config_file})
FSLDIR_PATH=$(python3 -c "import yaml, sys; print(yaml.safe_load(open(sys.argv[1]))['FSLDIR_PATH'])" ${config_file})
OUTPUT_DIRECTORY_PATH=$(python3 -c "import yaml, sys; print(yaml.safe_load(open(sys.argv[1]))['OUTPUT_DIRECTORY_PATH'])" ${config_file})
mkdir -p ${OUTPUT_DIRECTORY_PATH}

# Source needed software paths
echo "Sourcing ${BIOGRIDS_PATH}"
source ${BIOGRIDS_PATH}

echo "Activating Motion Pipeline Conda Environment: ${CONDA_ENV_NAME}"
source "${CONDA_INIT_PATH}"
conda activate "${CONDA_ENV_NAME}"

echo "Loading FSL at Directory: ${FSLDIR_PATH}" 
source "${FSLDIR_PATH}/etc/fslconf/fsl.sh"
export PATH=$FSLDIR_PATH/bin:$PATH

# Configure threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$SLURM_CPUS_PER_TASK

# Run run_pipeline.py
SCRIPT_PATH="$(dirname "$(realpath "$0")")"

echo "Running the Main Pipeline Script Via:"
echo
echo "  srun ${CONDA_ENV_PYTHON_PATH} run_pipeline.py \\"
echo "      --configuration_file ${config_file} \\"
echo "      >> ${OUTPUT_DIRECTORY_PATH}output_${SLURM_JOB_ID}.out \\"
echo "      2>> ${OUTPUT_DIRECTORY_PATH}output_${SLURM_JOB_ID}.err"
echo

srun ${CONDA_ENV_PYTHON_PATH} run_pipeline.py --configuration_file ${config_file} \
    >> "${OUTPUT_DIRECTORY_PATH}output_${SLURM_JOB_ID}.out" \
    2>> "${OUTPUT_DIRECTORY_PATH}output_${SLURM_JOB_ID}.err"

