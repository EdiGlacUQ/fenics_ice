#!/bin/bash
set -e

BASE_DIR=/home/ckoziol/Documents/Python/fenics/fenics_ice
RUN_DIR=$BASE_DIR/runs

INPUT_DIR=$BASE_DIR/input/smith_500m_input
OUTPUT_DIR=$BASE_DIR/output/smith/smith_dual_inversion

RC1=1.0
RC2=1e-4
RC3=1e-4
RC4=1e6
RC5=1e4

NX=225
NY=189

AITER=20

source activate dolfinproject_py3
cd $RUN_DIR

python run_inv.py -x $NX -y $NY -m 20 -a $AITER -p 2  -r $RC1 $RC2 $RC3 $RC4 $RC5 -d $INPUT_DIR -o $OUTPUT_DIR
