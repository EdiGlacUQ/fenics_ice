#!/bin/bash

#==============================================
# INSTALLATION SCRIPT FOR FENICS_ICE
#-----------------------------------
# Installs fenics_ice in a new conda environment 
# called 'fenics_ice'. Also runs an ISMIP-C test
# case (takes several minutes) to check everything
# works.
#
# Run this script where you find it (i.e. top level
# dir of the fenics_ice repository!)
# Be sure to edit CONDA_HOME before running.
#==============================================


# Assumes conda is available on system
# SET THIS!
CONDA_HOME=$HOME/miniconda3/

#Optionally choose a branch: master, joe
BRANCH="master"


#----------------------------------
# Change things below this point at your peril.
#----------------------------------

export FENICS_ICE_BASE_DIR="$PWD"
export INSTALL_DIR=$(dirname "$PWD") #parent directory


source $CONDA_HOME/etc/profile.d/conda.sh

cd $INSTALL_DIR


#Add conda forge
conda config --add channels conda-forge
conda config --set channel_priority strict

#Create the conda env
conda create -y -n fenics_ice -c conda-forge fenics fenics-dijitso fenics-dolfin fenics-ffc fenics-fiat fenics-libdolfin fenics-ufl

#install more packages
conda activate fenics_ice

#Create env variables
cd $CONDA_PREFIX
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/activate.d/env_vars.sh
touch ./etc/conda/deactivate.d/env_vars.sh
echo "export OMP_NUM_THREADS=1" > ./etc/conda/activate.d/env_vars.sh
echo "unset OMP_NUM_THREADS" > ./etc/conda/deactivate.d/env_vars.sh


conda install -y conda-build

pip install --upgrade pip

conda install -y matplotlib numpy ipython scipy seaborn h5py
pip install mpi4py toml gitpython "meshio[all]" pytest pytest-benchmark pytest-mpi pytest-dependency

cd $INSTALL_DIR

#install tlm_adjoint & fenics_ice
git clone https://github.com/jrmaddison/tlm_adjoint.git
cd $INSTALL_DIR/tlm_adjoint
git checkout jtodd/fice_devel

#git clone git@github.com:cpk26/fenics_ice.git

cd $FENICS_ICE_BASE_DIR
git checkout $BRANCH

#PYTHONPATH equiv which doesn't pollute system environment namespace
conda develop $INSTALL_DIR/tlm_adjoint/python/
conda develop $FENICS_ICE_BASE_DIR
#conda develop $INSTALL_DIR/fice_toolbox  <- system specific but this is how I use fice_toolbox

#=================================
#TEST SETUP WITH AN ISMIP-C CASE
#=================================

#Create ismip-c domain
cd $FENICS_ICE_BASE_DIR/aux
python gen_ismipC_domain.py -o ../input/ismipC -L 40000 -nx 100 -ny 100

#Generate velocity 'obs'
cd $FENICS_ICE_BASE_DIR/scripts/ismipc/
./forward_solve.sh

cd $FENICS_ICE_BASE_DIR/aux/
python Uobs_from_momsolve.py -b -L 40000 -d $FENICS_ICE_BASE_DIR/input/ismipC/momsolve

cd $FENICS_ICE_BASE_DIR/input/ismipC/momsolve
cp mask_vel.xml u_*.xml v_*.xml ..

cd $FENICS_ICE_BASE_DIR/scripts/ismipc/

#Run it
bash ./uq_rc_1e6.sh
