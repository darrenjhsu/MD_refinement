#!/bin/bash

cd pipeline$1
flist=`ls *.pdb`
cd antechamber_pipeline-master
for j in $flist
do
  sh pipeline.sh ../$j
done
