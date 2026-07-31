#!/bin/bash

# I have been using E3's rsfMRI_SMC_mc_env as the development env
# It has ITK installed, which is required to compile sms-mi-reg.cpp
conda activate rsfMRI_SMC_mc_env

mkdir -p sms-mi-reg/build
cd sms-mi-reg/build 
cmake ..
make -j$(nproc)