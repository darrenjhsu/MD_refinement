#!/bin/bash

cd SDF_match_PDB
counter=0
overall_counter=0
for i in `ls *_0.pdb`;
do
  mkdir -p ../pipeline$counter
  cp $i ../pipeline$counter/${i%_*}.pdb
  counter=$(( counter + 1 ))
  counter=$(( counter % 8 ))
  overall_counter=$(( overall_counter + 1 ))
done
echo $overall_counter

cd ..
for i in {0..8};
do
  cp -r antechamber_pipeline-master pipeline$i/
done

