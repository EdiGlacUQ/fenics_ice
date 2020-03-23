#!/bin/bash
set -e

BASE_DIR=$FENICS_ICE_BASE_DIR
RUN_DIR=$BASE_DIR/runs

INPUT_DIR=$BASE_DIR/input/ismipC
OUTPUT_DIR=$BASE_DIR/output/ismipC/uq_40x40
EIGENDECOMP_DIR=$OUTPUT_DIR/run_forward
FORWARD_DIR=$OUTPUT_DIR/run_forward

EIGENVALUE_FILE=slepc_eig_all.p
EIGENVECTOR_FILE=$EIGENDECOMP_DIR/vr.h5

RC1=1.0
RC2=1e-2
RC3=1e-9
RC4=1e6
RC5=1e-9


T=15.0
N=60
S=5

NX=40
NY=40

QOI=1

cd $RUN_DIR

python run_inv.py -b -x $NX -y $NY -m 200 -p 0  -r $RC1 $RC2 $RC3 $RC4 $RC5 -d $INPUT_DIR -o $OUTPUT_DIR
python run_forward.py -t $T -n $N -s $S -i $QOI -d $OUTPUT_DIR -o $FORWARD_DIR
python run_eigendec.py -s -m -p 0  -d $OUTPUT_DIR -o $EIGENDECOMP_DIR -f $EIGENVALUE_FILE
python run_errorprop.py -p 0 -d $FORWARD_DIR -e $EIGENDECOMP_DIR -l $EIGENVALUE_FILE -o $FORWARD_DIR
python run_invsigma.py -p 0 -d $FORWARD_DIR -e $EIGENDECOMP_DIR -k $EIGENVECTOR_FILE -l $EIGENVALUE_FILE -d $OUTPUT_DIR -o $FORWARD_DIR
