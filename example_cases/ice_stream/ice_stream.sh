#!/bin/bash
set -e

python $FENICS_ICE_BASE_DIR/runs/run_momsolve.py ice_stream.toml
python $FENICS_ICE_BASE_DIR/aux/Uobs_from_momsolve.py -i "U.h5" -o "ice_stream_U_obs.h5" -d ./output -l 750.0

cp output/ice_stream_U_obs.h5 input/

# #Run each phase of the model in turn
RUN_DIR=$FENICS_ICE_BASE_DIR/runs/
mpirun -n 3 python $RUN_DIR/run_inv.py ice_stream.toml
mpirun -n 3 python $RUN_DIR/run_forward.py ice_stream.toml
mpirun -n 3 python $RUN_DIR/run_eigendec.py ice_stream.toml
python $RUN_DIR/run_errorprop.py ice_stream.toml
python $RUN_DIR/run_invsigma.py ice_stream.toml
