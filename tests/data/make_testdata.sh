#!/bin/bash

#Keeping track of how these datafiles (for test cases) were produced.

mkdir input

cp $FENICS_ICE_BASE_DIR/example_cases/ismipc_rc_1e6/momsolve.toml .

# Create hi-res mesh
python $FENICS_ICE_BASE_DIR/aux/gen_rect_mesh.py -o ./input/momsolve_mesh.xml -xmax 40000 -ymax 40000 -nx 100 -ny 100

# Create the geometry/field data
python $FENICS_ICE_BASE_DIR/aux/gen_ismipC_domain.py -o ./input/ismipc_input.h5 -L 40000 -nx 100 -ny 100 --reflect

# Run the forward solution
python $FENICS_ICE_BASE_DIR/runs/run_momsolve.py momsolve.toml

# Add noise to velocity data & save HDF5
python $FENICS_ICE_BASE_DIR/aux/Uobs_from_momsolve.py -i "U.h5" -o "ismipc_U_obs.h5" -L 40000 -d ./output_momsolve

# Copy files & tidy up
rm momsolve.toml
mv input/ismipc_input.h5 .
mv output_momsolve/ismipc_U_obs.h5 .

rm -r input output_momsolve
