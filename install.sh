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
export INSTALL_DIR=$(dirname "$PWD")


source $CONDA_HOME/etc/profile.d/conda.sh

cd $INSTALL_DIR


#Add conda forge
conda config --add channels conda-forge
conda config --set channel_priority strict

#Create the conda env
conda create -y -n fenics_ice -c conda-forge fenics fenics-dijitso fenics-dolfin fenics-ffc fenics-fiat fenics-libdolfin fenics-ufl

#install more packages
conda activate fenics_ice

conda install -y conda-build

pip install --upgrade pip
conda install -y matplotlib numpy ipython scipy seaborn
pip install h5py mpi4py

#get pyrevolve
git clone https://github.com/opesci/pyrevolve.git
cd pyrevolve/
python setup.py install
cd $INSTALL_DIR

#install tlm_adjoint & fenics_ice
git clone https://github.com/jrmaddison/tlm_adjoint.git
#git clone git@github.com:cpk26/fenics_ice.git

cd $FENICS_ICE_BASE_DIR
git checkout $BRANCH

#PYTHONPATH equiv which doesn't pollute system environment namespace
conda develop $INSTALL_DIR/tlm_adjoint/python/
conda develop $INSTALL_DIR/fenics_ice/code

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
