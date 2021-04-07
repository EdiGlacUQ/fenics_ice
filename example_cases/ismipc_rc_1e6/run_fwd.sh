#!/bin/bash
set -e

#Run each phase of the model in turn
RUN_DIR=$FENICS_ICE_BASE_DIR/runs/
python $RUN_DIR/run_inv.py ismipc_rc_1e6.toml
python $RUN_DIR/run_forward.py ismipc_rc_1e6.toml
python $RUN_DIR/run_eigendec.py ismipc_rc_1e6.toml
python $RUN_DIR/run_errorprop.py ismipc_rc_1e6.toml
python $RUN_DIR/run_invsigma.py ismipc_rc_1e6.toml
