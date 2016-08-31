#!/bin/bash

# User specific aliases and functions
module load apps/python/anaconda3-2.5.0
module load apps/java/1.8u71
export PYTHONPATH='/home/cop15rj/rishav-msc-project/*:'

echo command=$*
$*
