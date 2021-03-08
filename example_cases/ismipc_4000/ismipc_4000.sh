#!/bin/bash
set -e

#Generate the input data (100x100 grid, and use this to generate 'obs_vel')
python $FENICS_ICE_BASE_DIR/aux/gen_rect_mesh.py -o ./input/momsolve_mesh.xml -xmax 40000 -ymax 40000 -nx 100 -ny 100
python $FENICS_ICE_BASE_DIR/aux/gen_rect_mesh.py -o ./input/ismip_mesh.xml -xmax 40000 -ymax 40000 -nx 30 -ny 30

python $FENICS_ICE_BASE_DIR/aux/gen_ismipC_domain.py -o ./input/ismipc_input.h5 -L 40000 -nx 100 -ny 100 --reflect
python $FENICS_ICE_BASE_DIR/runs/run_momsolve.py momsolve.toml
python $FENICS_ICE_BASE_DIR/aux/Uobs_from_momsolve.py -i "U.h5" -o "ismipc_U_obs.h5" -L 40000 -d ./output_momsolve --ls=4000.

cp output_momsolve/ismipc_U_obs.h5 input/

#Run each phase of the model in turn
RUN_DIR=$FENICS_ICE_BASE_DIR/runs/
python $RUN_DIR/run_inv.py ismipc_4000.toml
python $RUN_DIR/run_forward.py ismipc_4000.toml
python $RUN_DIR/run_eigendec.py ismipc_4000.toml
python $RUN_DIR/run_errorprop.py ismipc_4000.toml
python $RUN_DIR/run_invsigma.py ismipc_4000.toml
python $RUN_DIR/run_sample_post.py ismipc_4000.toml
