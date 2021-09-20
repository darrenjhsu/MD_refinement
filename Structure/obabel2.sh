#!/bin/bsah

mkdir -p SDF_match_PDB
cd SDF_match
counter=0
for i in `ls`;
  # Generate sdf without hydrogen
do
  if [ "$(( counter % 16 ))" -eq "15" ]; then
      obabel -isdf ${i} -opdb -O ../SDF_match_PDB/${i%.*}.pdb -h &&
      echo "Next round!"
      counter=0
  else
      obabel -isdf ${i} -opdb -O ../SDF_match_PDB/${i%.*}.pdb -h &
      counter=$(( counter + 1))
  fi
done

echo "SDF -> PDB done"
