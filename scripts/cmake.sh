#!/bin/bash

#-------------------------------------------------------------------------------
#
# Script to launch cmake with automatic options
#
# Advanced Robotics Department
# Fondazione Istituto Italiano di Tecnologia
#
# Author: Jesus Ortiz
# email : jesus.ortiz@iit.it
# Date  : 07-Dec-2012
#
#-------------------------------------------------------------------------------

# Calculate the root folder and pass it as an option to CMake
ROOT_DIR=$(pwd)/$(dirname $0)/..

# Clean root path
OLDIFS="$IFS"
IFS='/'
END=false
while ! $END; do
  FOUND_BACK=false
  AUX=""
  ROOT_DIR_AUX=""
  for TOKEN in $ROOT_DIR; do
    if $FOUND_BACK; then
      ROOT_DIR_AUX=${ROOT_DIR_AUX}/${TOKEN}
    else
      if [ "$TOKEN" == ".." ]; then
        FOUND_BACK=true
      else
        if [ -n "$ROOT_DIR_AUX" ]; then
          ROOT_DIR_AUX=${ROOT_DIR_AUX}/${AUX}
        else
          ROOT_DIR_AUX=${AUX}
        fi
        AUX=$TOKEN
      fi
    fi
  done
  if ! $FOUND_BACK; then
    END=true
  else
    ROOT_DIR="/${ROOT_DIR_AUX}"
  fi
done
IFS="$OLDIFS"
OPTIONS="-DROOT_DIR=$ROOT_DIR"


# Launch cmake with the automatic options
echo "Launching CMake:"
echo "  cmake $OPTIONS"
cmake $ROOT_DIR $OPTIONS

