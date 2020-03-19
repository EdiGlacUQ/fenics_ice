#!/bin/bash
set -e

BASE_DIR=$FENICS_ICE_BASE_DIR
RUN_DIR=$BASE_DIR/runs

INPUT_DIR=$BASE_DIR/input/ismipC
OUTPUT_DIR=$BASE_DIR/output/ismipC/ismipC_inv4_perbc_20x20_gnhep_prior
EIGENDECOMP_DIR=$OUTPUT_DIR/run_forward
FORWARD_DIR=$OUTPUT_DIR/run_forward

EIGFILE=slepceig_all.p

RC1=1.0
RC2=1e-4
RC3=1e-4
RC4=1e4
RC5=1e4

T=30.0
N=120
S=5

NX=20
NY=20

QOI=1

cd $RUN_DIR

python run_inv.py -b -x $NX -y $NY -m 200 -p 0  -r $RC1 $RC2 $RC3 $RC4 $RC5 -d $INPUT_DIR -o $OUTPUT_DIR
#python run_forward.py -t $T -n $N -s $S -i $QOI -d $OUTPUT_DIR -o $FORWARD_DIR
python run_eigendec.py -s -m -p 0   -d $OUTPUT_DIR -o $EIGENDECOMP_DIR -f $EIGFILE
#python run_errorprop.py -p 0 -d $FORWARD_DIR -e $EIGENDECOMP_DIR -l $EIGFILE -o $FORWARD_DIR
