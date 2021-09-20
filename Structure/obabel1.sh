#!/bin/bsah

mkdir -p SDF_raw
cd PDBQT
counter=0
for i in `ls`;
  # Generate sdf without hydrogen
do
  if [ "$(( counter % 16 ))" -eq "15" ]; then
      obabel -ipdbqt ${i} -osdf -O ../SDF_raw/${i%.*}.sdf -d &&
      echo "Next round!"
      counter=0
  else
      obabel -ipdbqt ${i} -osdf -O ../SDF_raw/${i%.*}.sdf -d &
      counter=$(( counter + 1 ))
  fi
done

echo "PDBQT -> SDF done"
