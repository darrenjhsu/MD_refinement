
import os
import sys
import json
import numpy as np

if len(sys.argv) < 2: # print usage
    print("Usage: python Plan_simulation.py <config_file>")
    exit()


#lig_base = 'Z1530724813'
#lig_prefix = lig_base + '_1_T1'
#lig_suffix = '_H_bcc_sim'
#param_dir = lig_base + '_parameter_tests'
inpcrd_dir = '../Structure/inpcrd' #lig_base + '_complexes'
prmtop_dir = '../Structure/prmtop'

inpcrd_listing = os.listdir('../'+inpcrd_dir)
prmtop_listing = os.listdir('../'+prmtop_dir)
inpcrd_listing.sort()
prmtop_listing.sort()

inpcrd_ul = [x.split('_')[0] for x in inpcrd_listing if 'inpcrd' in x] # ul is unique list
prmtop_ul = [x.split('.')[0] for x in prmtop_listing if 'prmtop' in x]
#print(prmtop_ul)

# Check all inpcrd has corresponding prmtop in the folder

for ii in inpcrd_ul:
    if ii not in prmtop_ul:
        print(f'There is no {ii}.prmtop in {prmtop_dir} folder!')
        exit()

print("File check passed. All inpcrd have corresponding prmtop.")

num_sys = len(inpcrd_ul)

# Create a corresponding prmtop_list
prmtop_list = [x + '.prmtop' for x in inpcrd_ul]
inpcrd_list = [x.split('.')[0] for x in inpcrd_listing]
#print(prmtop_list)

# load json config file
with open(sys.argv[1], 'r') as f:
    settings = json.load(f)

try:
    PERF = settings["Perf_assumption"]
except:
    PERF = 250000.0 # steps / minute for each simulation

# Plan simulation - we want to use all cards and minimize rounds of simulations

EM_Nrep = settings["EM"]["N-rep"]
EM_concurrency = 80 - 80 % EM_Nrep
EM_Nsim = num_sys * EM_Nrep
#EM_Nround = np.ceil(EM_Nsim / NGPU / 80)
EM_min_per_sim = settings["EM"]["cntrl"]["nstlim"] / PERF
EM_max_round = 120 / EM_min_per_sim
EM_Nsim_when_max_round = EM_Nsim / EM_max_round
EM_GPU_when_max_round = EM_Nsim_when_max_round / EM_concurrency

if EM_GPU_when_max_round < 6:
    EM_GPU_when_max_round = 6
    EM_max_round = np.ceil(EM_Nsim / EM_GPU_when_max_round / EM_concurrency)
    EM_Nsim_when_max_round = np.ceil(EM_Nsim / EM_GPU_when_max_round / EM_concurrency) * EM_concurrency

if EM_concurrency < 80:
    print(f'\n\n  Notice: Since N-rep is set to {EM_Nrep} and not a factor of 80, the concurrency for EM round is reduced to {EM_concurrency}')
print(f'''
*** EM - Energy minimization:
  There are {num_sys} unique systems to minimize energy.
  We are running {EM_Nrep} independent simulation(s) for each system, totalling {EM_Nsim} simulations.

  Minimal time to complete this round is to run on {np.ceil(EM_Nsim / EM_concurrency / 6) * 6} GPU cards or {np.ceil(EM_Nsim / EM_concurrency / 6)} nodes.
  Each GPU card takes {EM_concurrency} simulations.
  This would take {EM_min_per_sim:.2f} minutes and {np.ceil(EM_Nsim / EM_concurrency / 6) * EM_min_per_sim / 60:.2f} node hours.

  Alternatively, to use the minimal amount of nodes (and keep sim time < 2 hours), 
  you could run on {EM_GPU_when_max_round} GPU for {EM_max_round} rounds, having each GPU run {EM_Nsim_when_max_round} simulations.
  This would take {EM_max_round * EM_min_per_sim:.2f} minutes and {EM_GPU_when_max_round / 6 * EM_max_round * EM_min_per_sim / 60:.2f} node hours.
''')

NGPU = input("How many GPUs do you want to use to complete this task? ")
NGPU = int(NGPU)

# write EM part
systems_assignment = np.linspace(0,num_sys,NGPU+1,dtype=int)

script_mode = "EM"

for ii in range(NGPU):
    with open(f'mdgxGPU_EM_{ii}.in','w') as f:
        f.write(f'&files\n')
        for key in settings[script_mode]["files"]:
          f.write(f'  {key}  {settings[script_mode]["files"][key]}\n')
        f.write(f'&end\n\n')
    
        f.write(f'&cntrl\n')
        for key in settings[script_mode]["cntrl"]:
          f.write(f'  {key} = {settings[script_mode]["cntrl"][key]},\n')
        f.write(f'&end\n\n')
    
        f.write(f'&pptd\n')
        for key in settings[script_mode]["pptd"]:
          f.write(f'  {key} = {settings[script_mode]["pptd"][key]},\n')
        for jj in range(systems_assignment[ii], systems_assignment[ii+1]):
            if EM_Nrep > 1:
                f.write(f'  oligomer -p {prmtop_dir}/{prmtop_list[jj]} -c {inpcrd_dir}/{inpcrd_listing[jj]} -o {inpcrd_list[jj]}/EM -x {inpcrd_list[jj]}/EM -r {inpcrd_list[jj]}/EM N-rep {EM_Nrep}\n')
            else:
                f.write(f'  oligomer -p {prmtop_dir}/{prmtop_list[jj]} -c {inpcrd_dir}/{inpcrd_listing[jj]} -o {inpcrd_list[jj]}/EM_R1 -x {inpcrd_list[jj]}/EM_R1 -r {inpcrd_list[jj]}/EM_R1\n')
            # print(f'blah blah for system {jj}')
    # Loop through the oligmer script
        f.write(f'&end\n\n')
    

with open(f'EM_script.sh', 'w') as f:
    f.write('#!/bin/bash\n')
    for ii in inpcrd_list:
        f.write(f'mkdir -p {ii}\n')

with open(f'Ssubmit_EM.sh','w') as f:
    f.write(f'''#!/bin/bash
#BSUB -P STF006
#BSUB -W 2:00
#BSUB -nnodes {int(np.ceil(NGPU / 6))}
#BSUB -J mdgx_test
module load gcc/9.3.0 cuda/11.0.3 cmake readline zlib bzip2 boost netcdf-c netcdf-cxx netcdf-fortran parallel-netcdf  openblas netlib-lapack fftw

~/miniconda/bin/conda init bash
source ~/.bashrc
conda activate amber

sh EM_script.sh

for i in {{0..{NGPU-1}}};
do
  jsrun -n 1 -g 1 -a 1 -c 1 --smpiargs="off" ~/Tools/amber_build_rhel8/bin/mdgx.cuda -O -i mdgxGPU_EM_${{i}}.in -Reckless &
done
wait
''')
