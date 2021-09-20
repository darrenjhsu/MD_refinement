#!/bin/bsah

mkdir -p SDF_match
cd SDF_raw
for i in `ls`;
do
  python ../empty_fields.py ./ $i ../SDF_match/
done

echo "SDF cleaning done"
