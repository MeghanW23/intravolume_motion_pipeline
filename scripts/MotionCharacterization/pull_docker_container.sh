#!/bin/bash

# Create a file containing your gitlab personal access token.
cat glpat-access-token-file.txt | docker login ccts3.aws.chboston.org:5151 --username firstname.lastname --password-stdin

# To pull the container from the gitlab container registry:
docker pull ccts3.aws.chboston.org:5151/computationalradiology/sms-mi-reg

# Tag the container 
docker tag ccts3.aws.chboston.org:5151/computationalradiology/sms-mi-reg crl/sms-mi-reg

