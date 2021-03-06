
import sys

input_file = sys.argv[1]

with open(input_file,'r') as inp, open('output.mol2','w') as out:
    atom_dict = {}
    switch = 0
    for line in inp:
        if '@<TRIPOS>ATOM' in line:
            switch = 1
            out.write(line)
            continue
        elif '@<TRIPOS>BOND' in line:
            switch = 0
            out.write(line)
            continue
        if switch:
            temp = line.split()
            atom_name = temp[1]
            
            if atom_name in list(atom_dict.keys()):
                atom_dict[atom_name] += 1
            else:
                atom_dict[atom_name] = 1

            atom_string = atom_name + str(atom_dict[atom_name])
            out.write('%7s %5s %10s %10s %10s %5s   1   LIG \n'%(temp[0],atom_string,temp[2],temp[3],temp[4],temp[5]))
        else:
            out.write(line)

