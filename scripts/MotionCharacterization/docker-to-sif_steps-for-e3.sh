# STEP ONE 
# Get the architecture of the host (e3)
# It should be: x86_64 / AMD64
# If you are using a different host, run: 
#   uname -m
# to get the architecture

# STEP TWO 
# Log into the machine where you can access both Docker and the internet
# This could be your local machine, Gammu, etc.
# 
# Log into BCH's GitLab using your glpat token:
cat glpat-access-token-file.txt | docker login ccts3.aws.chboston.org:5151 --username firstname.lastname --password-stdin

# STEP THREE
# Pull the image from the BCH Gitlab Container Registry 
docker pull \
    --platform <host-os>/<host-architecture> \
    ccts3.aws.chboston.org:5151/computationalradiology/sms-mi-reg \
    
# STEP FOUR 
# Tag the image 
docker tag \
    ccts3.aws.chboston.org:5151/computationalradiology/sms-mi-reg \
    crl/sms-mi-reg

# STEP FIVE 
# Save the image to a .tar file 
docker save \
    -o sms-mi-reg.tar \
    crl/sms-mi-reg:latest

# STEP SIX
# Send the image to host system 
rsync -ahvzP sms-mi-reg.tar \
ch246081@e3-login2.tch.harvard.edu:/lab-share/Neuro-Cohen-e2/Groups/IRB-P00049401/intravolume_motion_pipeline/scripts/MotionCharacterization/sms-mi-reg/build

# STEP SEVEN 
# Convert the .tar image to a .sif image on the host system 
singularity build sms-mi-reg.sif docker-archive://sms-mi-reg.tar

# STEP EIGHT
# Update the path in the config file: SMS_MI_REG_SINGULARITY_IMAGE_PATH