#!/bin/bash
set -e

#Generate the input data (100x100 grid, and use this to generate 'obs_vel')
python $FENICS_ICE_BASE_DIR/aux/gen_ismipC_domain.py -o ./input -b -L 40000 -nx 100 -ny 100 
python $FENICS_ICE_BASE_DIR/runs/run_momsolve.py momsolve.toml
python $FENICS_ICE_BASE_DIR/aux/Uobs_from_momsolve.py -b -L 40000 -d ./output_momsolve

#Copy vel files across
cd output_momsolve
cp u*xml v*xml mask_vel.xml ../input/
cd ..

#Run each phase of the model in turn
RUN_DIR=$FENICS_ICE_BASE_DIR/runs/
python $RUN_DIR/run_inv.py ismipc_rc_1e4.toml
python $RUN_DIR/run_forward.py ismipc_rc_1e4.toml
python $RUN_DIR/run_eigendec.py ismipc_rc_1e4.toml
python $RUN_DIR/run_errorprop.py ismipc_rc_1e4.toml
python $RUN_DIR/run_invsigma.py ismipc_rc_1e4.toml
