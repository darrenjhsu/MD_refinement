
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

# Create a corresponding prmtop_list
inpcrd_list = [x.split('.')[0] for x in inpcrd_listing]

# Check EM round is successful
simulation_dir = '../../Simulation'

if os.path.isfile('EM_success.txt'):
    print('Found existing EM tally, use as is!')
    with open('EM_success.txt','r') as f:
        EM_success = f.readlines()[0].split(',')
#    print(EM_success)
else:
    EM_success = []
    for idx, ii in enumerate(inpcrd_list):
        EM_success_this = True
        EM_out_this = os.listdir(f'{simulation_dir}/{ii}')
        EM_out_this = [x for x in EM_out_this if 'EM' in x]
        EM_out_this = [x for x in EM_out_this if 'out' in x]
        for jj in EM_out_this:
            Temp = 0.0
            with open(f'{simulation_dir}/{ii}/{jj}','r') as f:
                cont = f.readlines()
            for line in cont:
                if 'Temperature' in line:
                    Temp = line.split(':')[-1]
            if np.abs(float(Temp)) > 1000:
                EM_success_this = False
                break
        if EM_success_this:
            EM_success.append(ii)
            print(f'{idx:5d}: Ligand {ii} is successfully tested')
        else:
            print(f'{idx:5d}: Ligand {ii} failed test')
    with open('EM_success.txt','w') as f:
        f.writelines(','.join(EM_success))

inpcrd_list = EM_success
inpcrd_ul = [x.split('_')[0] for x in inpcrd_list] # ul is unique list
prmtop_ul = [x.split('.')[0] for x in prmtop_listing if 'prmtop' in x]
prmtop_list = [x + '.prmtop' for x in inpcrd_ul]

#inpcrd_ul = [x.split('_')[0] for x in inpcrd_listing if 'inpcrd' in x] # ul is unique list
#prmtop_ul = [x.split('.')[0] for x in prmtop_listing if 'prmtop' in x]

# Check all inpcrd has corresponding prmtop in the folder

for ii in inpcrd_ul:
    if ii.split('_')[0] not in prmtop_ul:
        print(f'There is no {ii}.prmtop in {prmtop_dir} folder!')
        exit()

print("File check passed. All inpcrd have corresponding prmtop.")

num_sys = len(inpcrd_ul)

# Create a corresponding prmtop_list
#prmtop_list = [x + '.prmtop' for x in inpcrd_ul]
#inpcrd_list = [x.split('.')[0] for x in inpcrd_listing]

# print json
with open(sys.argv[1], 'r') as f:
    settings = json.load(f)

try:
    PERF = settings["Perf_assumption"]
except:
    PERF = 250000.0 # steps / minute for each simulation

# QR round

QR_Nrep = settings["QR"]["N-rep"]
QR_concurrency = 80 - 80 % QR_Nrep
QR_Nsim = num_sys * QR_Nrep
#QR_Nround = np.ceil(QR_Nsim / NGPU / 80)
QR_min_per_sim = settings["QR"]["cntrl"]["nstlim"] / PERF
QR_max_round = 120 / QR_min_per_sim
QR_GPU_when_max_round = np.ceil(QR_Nsim / QR_concurrency / QR_max_round / 6) * 6
QR_Nsim_when_max_round = np.ceil(QR_Nsim / QR_GPU_when_max_round)


if QR_GPU_when_max_round <= 6:
    QR_GPU_when_max_round = 6
    QR_max_round = np.ceil(QR_Nsim / QR_GPU_when_max_round / QR_concurrency)
    QR_Nsim_when_max_round = np.ceil(QR_Nsim / QR_GPU_when_max_round / 6) * 6

try:
    QR_max_wait = settings["QR"]["Max-wait"]
    QR_set_max_wait = True
    QR_max_round_max_wait = np.floor(QR_max_wait / QR_min_per_sim)
    QR_GPU_when_max_wait = np.ceil(QR_Nsim / QR_concurrency / QR_max_round_max_wait / 6) * 6
    QR_Nsim_when_max_wait = np.ceil(QR_Nsim / QR_GPU_when_max_wait)
    QR_max_round_max_wait = np.ceil(QR_Nsim_when_max_wait / QR_concurrency)
except:
    QR_set_max_wait = False

if QR_concurrency < 80:
    print(f'\n\n  Notice: Since N-rep is set to {QR_Nrep} and not a factor of 80, the concurrency for QR round is reduced to {QR_concurrency}')
print(f'''
*** QR - Qualifying round:  
  We will test the stability of the {num_sys} unique systems.
  We are running {QR_Nrep} independent simulation(s) for each system, totalling {QR_Nsim} simulations.

  Minimal time to complete this round is to run on {np.ceil(QR_Nsim / QR_concurrency / 6) * 6} GPU cards or {np.ceil(QR_Nsim / QR_concurrency / 6)} nodes.
  Each GPU card takes {QR_concurrency} simulations.
  This would take {QR_min_per_sim:.2f} minutes and {np.ceil(QR_Nsim / QR_concurrency / 6) * QR_min_per_sim / 60:.2f} node hours.

  Alternatively, to use the minimal amount of nodes (and keep sim time < 2 hours), 
  you could run on {QR_GPU_when_max_round} GPU for {QR_max_round} rounds, having each GPU run {QR_Nsim_when_max_round} simulations.
  This would take {QR_max_round * QR_min_per_sim:.2f} minutes and {QR_GPU_when_max_round / 6 * QR_max_round * QR_min_per_sim / 60:.2f} node hours.''')

if QR_set_max_wait:
    print(f'''
  Finally, to keep wait time less than {QR_max_wait} minutes,
  you could run on {QR_GPU_when_max_wait} GPU for {QR_max_round_max_wait} rounds, having each GPU run {QR_Nsim_when_max_wait} simulations.
  This would take {QR_max_round_max_wait * QR_min_per_sim:.2f} minutes and {QR_GPU_when_max_wait / 6 * QR_max_round_max_wait * QR_min_per_sim / 60:.2f} node hours.
''')

if settings["QR"]["Vary-wait"]:
    perf_table = []
    print('  Here is a plan table for your reference:')
    print(f'    Wait, NGPU, Sim/GPU, Nodes, Node Hours')
    for QR_max_wait in range(2,100):
        QR_set_max_wait = True
        QR_max_round_max_wait = np.floor(QR_max_wait / QR_min_per_sim)
        if QR_max_round_max_wait == 0:
            continue
        QR_GPU_when_max_wait = np.ceil(QR_Nsim / QR_concurrency / QR_max_round_max_wait / 6) * 6
        QR_Nsim_when_max_wait = np.ceil(QR_Nsim / QR_GPU_when_max_wait)
        QR_max_round_max_wait = np.ceil(QR_Nsim_when_max_wait / QR_concurrency)
        QR_node_hours = QR_GPU_when_max_wait / 6 * QR_max_round_max_wait * QR_min_per_sim / 60
        if np.abs(QR_max_wait - QR_max_round_max_wait * QR_min_per_sim) < 1:
            print(f'    {QR_max_wait:4.0f}, {QR_GPU_when_max_wait:4.0f}, {QR_Nsim_when_max_wait:7.0f}, {QR_GPU_when_max_wait/6:5.0f}, {QR_node_hours:10.2f}')



NGPU = input("How many GPUs do you want to use to complete this task? ")
NGPU = int(NGPU)

# write QR part
systems_assignment = np.linspace(0,num_sys,NGPU+1,dtype=int)

script_mode = "QR"
    
for ii in range(NGPU):

    
    with open(f'mdgxGPU_QR_{ii}.in','w') as f:
        f.write(f'&files\n')
        for key in settings[script_mode]["files"]:
          f.write(f'  {key}  {settings[script_mode]["files"][key]}\n')
        f.write(f'&end\n\n')
    
        f.write(f'&cntrl\n')
        for key in settings[script_mode]["cntrl"]:
          f.write(f'  {key} =  {settings[script_mode]["cntrl"][key]},\n')
        f.write(f'&end\n\n')
    
        f.write(f'&pptd\n')
        for key in settings[script_mode]["pptd"]:
          f.write(f'  {key} =  {settings[script_mode]["pptd"][key]},\n')
        for jj in range(systems_assignment[ii], systems_assignment[ii+1]):
            if QR_Nrep > 1:
                f.write(f'  oligomer -p {prmtop_dir}/{prmtop_list[jj]} -c {inpcrd_list[jj]}/EM_R1.rst -o {inpcrd_list[jj]}/QR -x {inpcrd_list[jj]}/QR -r {inpcrd_list[jj]}/QR N-rep {QR_Nrep}\n')
            else:
                f.write(f'  oligomer -p {prmtop_dir}/{prmtop_list[jj]} -c {inpcrd_list[jj]}/EM_R1.rst -o {inpcrd_list[jj]}/QR_R1 -x {inpcrd_list[jj]}/QR_R1 -r {inpcrd_list[jj]}/QR_R1\n')
    # Loop through the oligmer script
        f.write(f'&end\n\n')



with open(f'Ssubmit_QR.sh','w') as f:
    f.write(f'''#!/bin/bash
#BSUB -P STF006
#BSUB -W 2:00
#BSUB -nnodes {int(np.ceil(NGPU / 6))}
#BSUB -J mdgx_test
module load gcc/9.3.0 cuda/11.0.3 cmake readline zlib bzip2 boost netcdf-c netcdf-cxx netcdf-fortran parallel-netcdf  openblas netlib-lapack fftw

~/miniconda/bin/conda init bash
source ~/.bashrc
conda activate amber


for i in {{0..{NGPU-1}}};
do
  jsrun -n 1 -g 1 -a 1 -c 1 --smpiargs="off" ~/Tools/amber_build_rhel8/bin/mdgx.cuda -O -i mdgxGPU_QR_${{i}}.in -Reckless &
done
wait
''')


