#!/bin/bash
#Runs each model in turn (after first clearing out any previous results)

for f in ismipc_rc_1e4 ismipc_rc_1e6 ismipc_30x30 ismipc_40x40;
do
cd $f
echo "Running model in $f"
bash clean.sh
bash $f.sh
cd ..
done
