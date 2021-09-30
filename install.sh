#!/bin/bash

#==============================================
# INSTALLATION SCRIPT FOR FENICS_ICE
#-----------------------------------
# Installs fenics_ice in a new conda environment 
# called 'fenics_ice'.
#
# Run this script where you find it (i.e. top level
# dir of the fenics_ice repository!)
# Be sure to edit CONDA_HOME before running.
#
# The installation can be tested with:
#   pytest
# or
#   mpirun -n 2 pytest
#==============================================


# Assumes conda is available on system
# SET THIS!
CONDA_HOME=$HOME/miniconda3/

#----------------------------------
# Change things below this point at your peril.
#----------------------------------

export FENICS_ICE_BASE_DIR="$PWD"
export INSTALL_DIR=$(dirname "$PWD") #parent directory


source $CONDA_HOME/etc/profile.d/conda.sh

cd $INSTALL_DIR

# Create a conda environment for fenics_ice
conda env create -f $FENICS_ICE_BASE_DIR/environment.yml
conda activate fenics_ice

# Create env variable 'OMP_NUM_THREADS' to prevent OpenMP threading
cd $CONDA_PREFIX
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/activate.d/env_vars.sh
touch ./etc/conda/deactivate.d/env_vars.sh
echo "export OMP_NUM_THREADS=1" > ./etc/conda/activate.d/env_vars.sh
echo "unset OMP_NUM_THREADS" > ./etc/conda/deactivate.d/env_vars.sh

cd $INSTALL_DIR

# Install tlm_adjoint & checkout relevant branch
git clone https://github.com/jrmaddison/tlm_adjoint.git
cd $INSTALL_DIR/tlm_adjoint
git checkout fenics_ice

# Point the conda env to tlm_adjoint & fenics_ice
conda develop $INSTALL_DIR/tlm_adjoint
conda develop $FENICS_ICE_BASE_DIR
#conda develop $INSTALL_DIR/fice_toolbox  <- system specific but this is how I use fice_toolbox
