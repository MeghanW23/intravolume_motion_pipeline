#!/bin/bash

# Please see official Slurm documentation for more on SBATCH arguments - https://slurm.schedmd.com/sbatch.html

#SBATCH --partition=bch-compute                       # queue to be used - bch-compute is the default
#SBATCH --time=02:05:00                           # Running time (in hours-minutes-seconds) - only request what you need
#SBATCH --job-name=sif-conversion               # Job name - you should change this
#SBATCH --mail-type=BEGIN,END,FAIL            # send and email when the job begins, ends or fails
#SBATCH --mail-user=meghan.walsh2@childrens.harvard.edu    # Email address to send the job status - can use api endpoints, see wiki
#SBATCH --output=output_%j.txt                      # Name of the output file - this format is jobid
#SBATCH --cpus-per-task=4                   # Number of CPUs to request - default is 1 - change for multithreading
#SBATCH --mem=31000M                          # RAM to request - you should change this to meet your needs


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

rm ${TMPDIR}/slimm.sif

echo `date`
APPTAINER_CACHEDIR=${TMPDIR} APPTAINER_TMPDIR=${TMPDIR} apptainer build ${TMPDIR}/sms-mi-reg.sif docker-archive:${TMPDIR}/sms-mi-reg.tar
echo `date`