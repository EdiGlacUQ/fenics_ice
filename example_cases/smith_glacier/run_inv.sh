#!/bin/bash
set -e

echo $(date -u) "Run started"
mpirun -n 2 python $FENICS_ICE_BASE_DIR/runs/run_inv.py $RUN_CONFIG_DIR/smith_glacier_test/smith.toml
echo $(date -u) "Done!"

