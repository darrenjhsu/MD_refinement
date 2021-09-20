
import sys, os

if len(sys.argv) < 3:
    print("Usage: python empty_fields.py indir infile outdir")
    exit()

print(f'Input: {sys.argv[2]}')

indir = sys.argv[1]
outdir = sys.argv[3]

with open(indir + '/' + sys.argv[2], 'r') as f:
    cont = f.readlines()

natom = int(cont[3].split()[0])

for ii in range(4, 4+natom):
    cont[ii] = cont[ii][:34] + ' 0  0  0  0  0  0  0  0  0  0  0  0\n'

with open(outdir + '/' + sys.argv[2], 'w') as f:
    f.writelines(cont)
