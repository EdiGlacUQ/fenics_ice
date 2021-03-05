#!/bin/bash
set -e

# #Run each phase of the model in turn
RUN_DIR=$FENICS_ICE_BASE_DIR/runs/
mpirun -n 3 python $RUN_DIR/run_inv.py ice_stream.toml
mpirun -n 3 python $RUN_DIR/run_forward.py ice_stream.toml
mpirun -n 3 python $RUN_DIR/run_eigendec.py ice_stream.toml
mpirun -n 3 python $RUN_DIR/run_errorprop.py ice_stream.toml
mpirun -n 3 python $RUN_DIR/run_invsigma.py ice_stream.toml
