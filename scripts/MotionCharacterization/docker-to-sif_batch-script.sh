#!/bin/bash

# Please see official Slurm documentation for more on SBATCH arguments - https://slurm.schedmd.com/sbatch.html


#SBATCH --partition=bch-compute
#SBATCH --time=02:05:00
#SBATCH --job-name=sif-conversion
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=meghan.walsh2@childrens.harvard.edu
#SBATCH --output=output_%j.txt
#SBATCH --cpus-per-task=4
#SBATCH --mem=31000M


# ------------ Batch Processing Script ------------
# export TMPDIR=/lab-share/Rad-Warfield-e2/Public/
# Change this to compare local directory writing versus RC-FS writing speeds.
export TMPDIR=/home/ch246081
export APPTAINER_CACHEDIR=${TMPDIR}
export APPTAINER_TMPDIR=${TMPDIR}

# Write hostname to file
echo "Running job on node: "
hostname

# Write username to file
echo "Running job with user: "
whoami

# Write jobid to file
echo "This is SLURM_JOB_ID: "
echo $SLURM_JOB_ID

# rm ${TMPDIR}/sms-mi-reg.sif

echo `date`
APPTAINER_CACHEDIR=${TMPDIR} APPTAINER_TMPDIR=${TMPDIR} apptainer build ${TMPDIR}/sms-mi-reg.sif docker-archive:${TMPDIR}/sms-mi-reg.tar
echo `date`
