#!/bin/bash

# Whatever environment you use to run this script, make sure it has
# ITK installed, as it is required to compile the cpp/cxx files

mkdir -p sms-mi-reg/build
cd sms-mi-reg/build 
cmake ..

# if this command doesn't work, try running from the sms-mi-reg dir instead of the sms-mi-reg/build 
make -j$(nproc)
