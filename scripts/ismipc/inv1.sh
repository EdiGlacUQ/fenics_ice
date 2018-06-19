#!/bin/bash
set -e

BASE_DIR=/Users/conradkoziol/Documents/Python/fenics/fenics_ice
RUN_DIR=$BASE_DIR/runs

INPUT_DIR=$BASE_DIR/input/ismipC/
OUTPUT_DIR=$BASE_DIR/output/ismipC/ismipC_perbc_inv1
EIGENDECOMP_DIR=$OUTPUT_DIR/run_forward
FORWARD_DIR=$OUTPUT_DIR/run_forward

EIGFILE=slepceig_all.p

RC1=1.0
RC2=1e-6
RC3=1e-6
RC4=1e0
RC5=1e0

T=15.0
N=200
S=5



source activate dolfinproject_py3
cd $RUN_DIR

python run_inv.py -m 25 -p 0 -b -r $RC1 $RC2 $RC3 $RC4 $RC5 -d $INPUT_DIR -o $OUTPUT_DIR
python run_forward.py -t $T -n $N -s $S -d $OUTPUT_DIR -o $FORWARD_DIR
python run_eigendec.py -s -m -p 0   -d $OUTPUT_DIR -o $EIGENDECOMP_DIR -f $EIGFILE
python run_errorprop.py -p 0 -d $FORWARD_DIR -e $EIGENDECOMP_DIR -l $EIGFILE -o $FORWARD_DIR
