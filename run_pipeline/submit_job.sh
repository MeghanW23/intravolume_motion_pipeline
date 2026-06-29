#!/bin/bash
#SBATCH --partition=bch-compute
#SBATCH --time=18:00:00
#SBATCH --job-name=intravolume_motion_pipeline
#SBATCH --output=logs/output_%j.out
#SBATCH --error=logs/output_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=meghan.walsh2@childrens.harvard.edu

# set -euo pipefail

config_files=($@)
echo "Inputted Files: ${config_files[@]}"

echo "Getting Environment Variables from Inputted .env File: ${config_files[0]}"
source "${config_files[0]}"

echo "Sourcing ${BIOGRIDS_PATH}"
source ${BIOGRIDS_PATH}

echo "Activating Motion Pipeline Conda Environment: ${CONDA_ENV_NAME}"
source "${CONDA_INIT_PATH}"
conda activate "${CONDA_ENV_NAME}"

mkdir -p logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$SLURM_CPUS_PER_TASK

SCRIPT_PATH="$(dirname "$(realpath "$0")")"

echo "Running the Main Pipeline Script Via:"
echo " "
echo "  srun ${CONDA_ENV_PYTHON_PATH} ${RUN_PIPELINE_SCRIPT} \\" 
echo "      --configuration_files ${config_files[@]} \\" 
echo "      >> logs/output_${SLURM_JOB_ID}.out \\" 
echo "      2>> logs/output_${SLURM_JOB_ID}.err"
echo " "
srun ${CONDA_ENV_PYTHON_PATH} "${RUN_PIPELINE_SCRIPT}" --configuration_files "${config_files[@]}" \
    >> "logs/output_${SLURM_JOB_ID}.out" \
    2>> "logs/output_${SLURM_JOB_ID}.err"

