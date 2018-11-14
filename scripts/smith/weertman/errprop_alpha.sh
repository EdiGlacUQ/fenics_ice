#!/bin/bash
set -e

BASE_DIR=/home/ckoziol/Documents/Python/fenics/fenics_ice
RUN_DIR=$BASE_DIR/runs

INPUT_DIR=$BASE_DIR/input/smith_500m_input
OUTPUT_DIR=$BASE_DIR/output/smith/smith_dual_inversion

EIGENDECOMP_DIR=$OUTPUT_DIR/run_forward
FORWARD_DIR=$OUTPUT_DIR/run_forward

EIGFILE=slepceig_800.p

RC1=1.0
RC2=1e-4
RC3=1e-4
RC4=1e6
RC5=1e4

NUMEIG=800

T=100.0
N=2400
S=10

NX=20
NY=20



source activate dolfinproject_py3
cd $RUN_DIR

python run_eigendec.py -s -m -p 0 -n $NUMEIG  -d $OUTPUT_DIR -o $EIGENDECOMP_DIR -f $EIGFILE
#python run_forward.py -t $T -n $N -s $S -d $OUTPUT_DIR -o $FORWARD_DIR
#python run_errorprop.py -p 0 -d $FORWARD_DIR -e $EIGENDECOMP_DIR -l $EIGFILE -o $FORWARD_DIR
