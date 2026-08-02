#!/bin/bash

# Whatever environment you use to run this script, make sure it has
# ITK installed, as it is required to compile sms-mi-reg.cpp

mkdir -p sms-mi-reg/build
cd sms-mi-reg/build 
cmake ..
make -j$(nproc)
