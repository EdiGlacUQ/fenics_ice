#!/bin/bash
#Runs each model in turn (after first clearing out any previous results)

for f in $(ls -d ismipc*corr);
do echo $f
cd $f
echo "Running model in $f"
bash clean.sh
bash $f.sh > out 2> out2
cd ..
done
