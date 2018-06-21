#!/bin/bash
set -e

BASE_DIR=/Users/conradkoziol/Documents/Python/fenics/fenics_ice
RUN_DIR=$BASE_DIR/runs

INPUT_DIR=$BASE_DIR/input/ismipC
OUTPUT_DIR=$BASE_DIR/output/ismipC/ismipC_inv5_perbc_20x20_gnhep_prior
EIGENDECOMP_DIR=$OUTPUT_DIR/run_forward
FORWARD_DIR=$OUTPUT_DIR/run_forward

EIGFILE=slepceig_all.p

RC1=1.0
RC2=1e-2
RC3=1e-2
RC4=1e5
RC5=1e5

T=15.0
N=60
S=5

NX=20
NY=20



source activate dolfinproject_py3
cd $RUN_DIR

python run_inv.py -b -x $NX -y $NY -m 200 -p 0  -r $RC1 $RC2 $RC3 $RC4 $RC5 -d $INPUT_DIR -o $OUTPUT_DIR
python run_forward.py -t $T -n $N -s $S -d $OUTPUT_DIR -o $FORWARD_DIR
python run_eigendec.py -s -m -p 0   -d $OUTPUT_DIR -o $EIGENDECOMP_DIR -f $EIGFILE
python run_errorprop.py -p 0 -d $FORWARD_DIR -e $EIGENDECOMP_DIR -l $EIGFILE -o $FORWARD_DIR
