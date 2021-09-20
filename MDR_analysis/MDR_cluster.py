import os
import sys
import json
import copy
import mdtraj
import numpy as np
import time
import pandas as pd
import pickle
import mdtraj as md
import multiprocessing as mp
try:
    import cupy as cp
    cudaExists = True
except ImportError as e:
    cudaExists = False
    print("Can't load CuPy, fall back to numba")
from numba import jit, prange
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import nglview
from nglview.player import TrajectoryPlayer




@jit(nopython=True)
def pureRMSD(P, Q):
    # Assume P and Q are aligned first
    diff = P - Q
    N = len(P)
    return np.sqrt((diff * diff).sum() / N)

@jit(nopython=True,parallel=True)
def pureRMSDrow(P, Q):
    # Assume P and Q are aligned first, P is a ref, Q is an array of P-like matrices
    rmsd = np.zeros(len(Q))
    diff = np.zeros_like(Q)
    for ii in prange(len(Q)):
#         diff[ii] = P - Q[ii] 
        N = len(P)
#         rmsd[ii] = np.sqrt((diff[ii] * diff[ii]).sum() / N)
        rmsd[ii] = np.sqrt(((P-Q[ii]) * (P-Q[ii])).sum() / N)
    return rmsd

@jit(nopython=True,parallel=True)
def pureRMSDself(P):
    # Assume P and Q are aligned first, P is a ref, Q is an array of P-like matrices
    rmsd = np.zeros((len(P),len(P)))
    lenP = len(P[0])
    for ii in prange(len(P)):
#         if ii % 50 == 0:
#             print(ii)
        for jj in range(len(P)):
            Q = P[ii] - P[jj]
            rmsd[ii,jj] = np.sqrt(np.sum(Q * Q) / lenP)
    return rmsd

def pureRMSDcupy(P):
    Pcp = cp.array(P)
#     print(P.shape)
    rmsd = cp.zeros((len(P),len(P)))
#     print(rmsd.shape)
    lenP = len(P[0])
    for ii in range(len(P)):
#         if ii % 400 == 0:
#             print(ii)
        Q = Pcp - Pcp[ii]
#         print(Q.shape)
        rmsd[ii] = cp.sqrt(cp.sum(cp.sum(Q * Q, axis=2), axis=1) / lenP)
    rmsd_local = rmsd.get()
    del rmsd
    return rmsd_local
    
# traj = MDR.Ligands['Mpro-x10959'].Poses['3'].traj['MD'][0]
# len_traj = len(traj.ligandTrajectory)
# rmsd_cluster = np.zeros((len_traj,len_traj))
# for ii in range(len_traj):
#     for jj in range(len_traj):
#         rmsd_cluster[ii,jj] = pureRMSD(traj.ligandTrajectory[ii],traj.ligandTrajectory[jj])

def gimme_best_pose(MDR, ligand='Mpro-x10959', metric='vdW', ligand_res=56, filter_dist=False, filter_dist_thres=2.5, 
                    top_select=5, plot=True, cluster_min_samples=3, eps=0.4, min_size_multiplier=1, speed=15000, show_pose=False,
                    simRound='MD',rank=None, outputPDB=False):
        
    simRound = simRound
    ligand_res = str(ligand_res)
    
    if not filter_dist:
        filter_dist = True
        filter_dist_thres = 500
    
    # perform full ligand sample clustering to pick best poses based on the metric
    trajAgg = []
    rmsdAgg = []
    metricAgg = []
#     poseNum = str(poseNum)
    for pp in range(0, MDR.Ligands[ligand].numPoses):

        poseNum = str(pp)
#         print(poseNum)
        if MDR.Ligands[ligand].Poses[poseNum].successQR:
#             print(f'Pose num {poseNum}')
            for traj in MDR.Ligands[ligand].Poses[poseNum].traj[simRound]:
                if traj.hasTrjFile and traj.hasRMSD:
                    # Find out the minimum length recorded
                    minLength = np.min([len(traj.ligandTrajectoryH[10:]), len(traj.output[metric].values[10:])])
                    trajAgg.append(traj.ligandTrajectoryH[10:minLength])
                    rmsdAgg.append(traj.RMSD[10:minLength])
                    metricAgg.append(traj.output[metric].values[10:minLength])
#                     print(len(traj.ligandTrajectoryH[10:]), 
#                           len(traj.RMSD[10:]), 
#                           len(traj.output[metric].values[10:]))
    trajAgg = np.array(np.concatenate(trajAgg))
    rmsdAgg = np.array(np.concatenate(rmsdAgg)).flatten()
    metricAgg = np.array(np.concatenate(metricAgg)).flatten()

    len_traj = len(trajAgg)

    stride = max(len_traj // speed,1)
    print(f'Stride factor is {stride} (number of frames: {len_traj//stride})')
    
    if cudaExists:
        rmsd_cluster = pureRMSDcupy(trajAgg[::stride])
    else:
        rmsd_cluster = pureRMSDself(trajAgg[::stride])
    print(f'RMSD calculation done on {len(trajAgg[::stride])} frames.')
    
    cluster_min_samples = cluster_min_samples
    clustering = DBSCAN(eps=eps, min_samples=cluster_min_samples, metric='precomputed').fit(rmsd_cluster)
    print("Clustering done")


    # for ii in MDR.Ligands['Mpro-x10959'].Poses:
    simLigTraj = []
    simTraj = []
    simOutput = []
    simLocation = []
    simCumLength = [0] # This line needs check
    simCumLength = []
    for pp in range(0, MDR.Ligands[ligand].numPoses):
        poseNum = str(pp)
    #     if ii == poseNum:
        if MDR.Ligands[ligand].Poses[poseNum].successQR:

            for jj in range(0, len(MDR.Ligands[ligand].Poses[poseNum].traj[simRound])):
                for simType in [simRound]:
        #             print(ii,simType,jj)
                    traj = MDR.Ligands[ligand].Poses[poseNum].traj[simType][jj]
                    if traj.hasRMSD and traj.hasTrjFile:
                        minLength = np.min([len(traj.ligandTrajectoryH[10:]), len(traj.output[metric].values[10:])])
                        simLigTraj.append(traj.ligandTrajectoryH[10:minLength])
                        simTraj.append(traj.RMSD[10:minLength])
                        simOutput.append(traj.output[metric].values[10:minLength])
                        simLocation.append(traj)
                        if len(simCumLength) > 0:
                            simCumLength.append(simCumLength[-1]+len(traj.RMSD[10:minLength]))
                        else:
                            simCumLength = [len(traj.RMSD[10:minLength])]
    simLigTraj = np.concatenate(simLigTraj)
    simTraj = np.array(np.concatenate(simTraj)).flatten()
    simOutput = np.array(np.concatenate(simOutput)).flatten()
    simCumLength = np.array(simCumLength)
    # Diagnostics
#     print(simCumLength)
    if plot:
        plt.figure(figsize=(12,8))
        plt.scatter(simTraj[::stride],clustering.labels_,s=5)
        plt.yticks(np.sort(np.unique(clustering.labels_)))
        plt.ylabel('Cluster index',fontsize=15)
        plt.xlabel('RMSD (Å)',fontsize=15)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.xlim([1, 15])
#         plt.ylim([-1, 250])
    results = []
    representative = {}
    
    if filter_dist:
        try:
            crystalComp = mdtraj.load(MDR.Ligands[ligand].crystalPose, top=MDR.Ligands[ligand].prmtop)
            usingCrystalComp = True
        except:
            if rank is not None:
                crystalComp = mdtraj.load(f'{MDR.inpcrdFolder}/{ligand}_{rank}.inpcrd', top=MDR.Ligands[ligand].prmtop)
                print(f'Using ranked reference')
                print(f'No crystal comp found - using pose {rank} instead. RMSD will be meaningless')
            else:
                crystalComp = mdtraj.load(f'{MDR.inpcrdFolder}/{ligand}_0.inpcrd', top=MDR.Ligands[ligand].prmtop)
                print(f'Using rank 0 reference')
                print(f'No crystal comp found - using pose 0 instead. RMSD will be meaningless')
            usingCrystalComp = False

        pro = crystalComp.top.select(f'not residue {ligand_res} and not symbol H')
        close_atoms = np.array([ 45, 106, 107, 167, 168, 170, 175, 176, 177, 178, 179, 180, 181, 182, 232, 259, 381, 386, 387, 388])
        COM_active_site = crystalComp.xyz[0][pro][close_atoms].mean(0)
    
    if plot:
        plt.figure(figsize=(8,6))
    for ii in np.unique(clustering.labels_):

        if np.sum(clustering.labels_ == ii) >= min_size_multiplier*cluster_min_samples and ii >= 0:
            selected = simLigTraj[::stride][clustering.labels_ == ii]
    #         print(selected.shape)
            centroid = np.mean(selected,axis=0)
    #         print(centroid.shape)

            distance_to_centroid = pureRMSDrow(centroid, selected)
            representative[ii] = np.argmin(distance_to_centroid)
            if filter_dist:
#                 ligandActiveSiteDistance = np.sqrt(((simLigTraj[::stride][clustering.labels_ == ii][representative[ii]].mean(0) - COM_active_site*10)**2).sum())
                ligandActiveSiteDistance = np.sqrt(((simLigTraj[::stride][clustering.labels_ == ii].mean(1) - COM_active_site*10)**2).sum(1)).mean(0)
#                 print(ligandActiveSiteDistance)
                if ligandActiveSiteDistance < filter_dist_thres:
#                     print("Trying to append")
                    try:
                        results.append([ii, np.sum(clustering.labels_ == ii), np.mean(simTraj[::stride][clustering.labels_ == ii]), np.mean(simOutput[::stride][clustering.labels_ == ii]), ligandActiveSiteDistance])
                    except:
                        print(f'Error at cluster {ii}, simTraj has the shape {simTraj.shape}, simOutput has the shape {simOutput.shape}')
                        raise
#                     print(f'{ii:3d}, {np.sum(clustering.labels_ == ii):5d}, \
# {np.mean(simTraj[::stride][clustering.labels_ == ii]):.3f}, {np.mean(simOutput[::stride][clustering.labels_ == ii]):.3f}, \
# {simTraj[::stride][clustering.labels_ == ii][representative[ii]]:.3f}, {simOutput[::stride][clustering.labels_ == ii][representative[ii]]:.3f}, {ligandActiveSiteDistance:.3f} Å')
#                 else:
#                     print(f'{ii:3d}, {np.sum(clustering.labels_ == ii):5d}, \
# {np.mean(simTraj[::stride][clustering.labels_ == ii]):.3f}, {np.mean(simOutput[::stride][clustering.labels_ == ii]):.3f}, \
# {simTraj[::stride][clustering.labels_ == ii][representative[ii]]:.3f}, {simOutput[::stride][clustering.labels_ == ii][representative[ii]]:.3f}, {ligandActiveSiteDistance:.3f} Å XX')
                    if plot:
                        plt.scatter(simTraj[::stride][clustering.labels_ == ii],simOutput[::stride][clustering.labels_ == ii],s=2)
                        plt.scatter(np.mean(simTraj[::stride][clustering.labels_ == ii]), np.mean(simOutput[::stride][clustering.labels_ == ii]), 
                                    s = np.sum(clustering.labels_ == ii), color = [0,0,1,0.05], edgecolor='r', linewidths=2.4)
                        
                        plt.xlabel('RMSD (Å)', fontsize=15)
                        plt.ylabel('Raw vdW energy (kcal/mol)', fontsize=15)
                        plt.tick_params(axis='both', which='major', labelsize=15)
            else:
                results.append([ii, np.sum(clustering.labels_ == ii), np.mean(simTraj[::stride][clustering.labels_ == ii]), np.mean(simOutput[::stride][clustering.labels_ == ii])])
#             if plot:
                
#                 plt.scatter(np.mean(simTraj[::stride][clustering.labels_ == ii]), ii, c='r',s=8)
            
    results = np.array(results)
#     print(results)
    if plot:



        plt.figure(figsize=(11,5))
        plt.grid(alpha=0.3)
        plt.scatter(results[:,2], results[:,3], s = results[:,1], color = [0,0,1,0.05], edgecolor='r', linewidths=2.4)
        plt.xlabel('RMSD (Å)', fontsize=15)
        plt.ylabel('Raw vdW energy (kcal/mol)', fontsize=15)
        plt.tick_params(axis='both', which='major', labelsize=15)

        
#         plt.figure(figsize=(12,8))
#         plt.scatter(results[:,4], results[:,3], s = results[:,1], color = [0,0,1,0.02], edgecolor='r', linewidths=1.5)
#         plt.xlabel('Dist (Å)', fontsize=15)
#         plt.ylabel('Raw vdW energy (kcal/mol)', fontsize=15)
#         plt.tick_params(axis='both', which='major', labelsize=15)

#         plt.figure(figsize=(12,8))
#         plt.scatter(results[:,4], results[:,2], s = results[:,1], color = [0,0,1,0.02], edgecolor='r', linewidths=1.5)
#         plt.xlabel('Dist (Å)', fontsize=15)
#         plt.ylabel('RMSD (Å)', fontsize=15)
#         plt.tick_params(axis='both', which='major', labelsize=15)

    
    if top_select == 'all':
        top_select = len(results)
    
    cluster_selection = results[:,0][np.argsort(results[:,3])[:top_select]].astype(int)
    print(f'Selected clusters are: {cluster_selection}')
    
    rmsd_selection = [simTraj[::stride][clustering.labels_ == ii][representative[ii]] for ii in cluster_selection]
    if not usingCrystalComp:
        if rank is not None:
            print(f'RMSD of the clusters : {rmsd_selection} NOTE: these values are meaningless - they are compared to predicted pose {rank}')
        else:
            print(f'RMSD of the clusters : {rmsd_selection} NOTE: these values are meaningless - they are compared to predicted pose 0')
    elif min(rmsd_selection) < 2.5:
        print(f'RMSD of the clusters : {rmsd_selection} PASS')
    else:
        print(f'RMSD of the clusters : {rmsd_selection} ')
    if usingCrystalComp:
        print(f'Lowest possible RMSD : {np.min(results[:,2])}')
    
#     print(simCumLength)
    
    # Set the comps for each selected conformers
    comp = {}
    actualComp = {}
    comp_selection = [simLigTraj[::stride][clustering.labels_ == ii][representative[ii]] for ii in cluster_selection]   
    loc_selection = [np.arange(len(simLigTraj))[::stride][clustering.labels_ == ii][representative[ii]] for ii in cluster_selection]
#     print(loc_selection)
    for idx, ii in enumerate(cluster_selection):
        if usingCrystalComp:
            ref_pose = mdtraj.load(f'reference_structure/{ligand}_0A.inpcrd',top=f'reference_structure/{ligand}.prmtop')
        else:
            if rank is not None:
                ref_pose = mdtraj.load(f'{MDR.inpcrdFolder}/{ligand}_{rank}.inpcrd', top=MDR.Ligands[ligand].prmtop)
            else:
                ref_pose = mdtraj.load(f'{MDR.inpcrdFolder}/{ligand}_0.inpcrd', top=MDR.Ligands[ligand].prmtop)
        if usingCrystalComp:
            comp[ii] = mdtraj.load(f'reference_structure/{ligand}_0A.inpcrd',top=f'reference_structure/{ligand}.prmtop')
        else:
            if rank is not None:
                comp[ii] = mdtraj.load(f'{MDR.inpcrdFolder}/{ligand}_{rank}.inpcrd', top=MDR.Ligands[ligand].prmtop)
            else:
                comp[ii] = mdtraj.load(f'{MDR.inpcrdFolder}/{ligand}_0.inpcrd', top=MDR.Ligands[ligand].prmtop)
        comp[ii].xyz[0][-len(comp_selection[0]):] = simLigTraj[::stride][clustering.labels_ == ii][representative[ii]]/10
        comp[ii].superpose(ref_pose, atom_indices=range(0,700))
        actualLocation = np.where(simCumLength > loc_selection[idx])[0][0]
        actualFile = simLocation[actualLocation]
        actualComp[ii] = mdtraj.load_mdcrd(f'{actualFile.trjFile}',top=MDR.Ligands[ligand].prmtop)
        actualComp[ii].superpose(ref_pose, atom_indices=range(0,700))
        if actualLocation == 0:
            actualFrame = loc_selection[idx] + 10
        else:
            actualFrame = loc_selection[idx] - simCumLength[actualLocation-1] + 10
        actualComp[ii] = actualComp[ii][actualFrame]
        # Diagnostics
#         print(actualLocation)
#         print(actualFile)
#         print(actualFrame)
            
        
    # Output PDB
    if outputPDB:
        try:
            os.mkdir('Refined_Poses')
        except:
            pass
        try:
            os.mkdir(f'Refined_Poses/{ligand}')
        except:
            pass
        for idx, ii in enumerate(cluster_selection):
            # There's some bug with actualComp, so we output comp for now
            actualComp[ii][0].save(f'Refined_Poses/{ligand}/Pose{idx}.pdb') 
#             comp[ii][0].save(f'Refined_Poses/{ligand}/Pose{idx}.pdb')
    if show_pose: # Show selected poses
        view = nglview.NGLWidget() 
        view.camera = 'orthographic'
        


#         print(loc_selection)
        c = {}


#         ac = {}
        colors = ['red','orange','yellow','green','blue','purple']
        for idx, ii in enumerate(cluster_selection):

#             ac[ii] = view.add_trajectory(actualComp[ii][actualFrame])
#             ac[ii].clear()
#             ac[ii].add_cartoon(selection="protein")
#             ac[ii].add_licorice(selection="protein",opacity=0.3,color='white')
# For poster
#             comp[ii].xyz[0][-len(comp_selection[0]):][:,0] -= idx // 3 * 1.5
#             comp[ii].xyz[0][-len(comp_selection[0]):][:,1] -= idx % 3
            
            c[ii] = view.add_trajectory(comp[ii])
            c[ii].clear()
            if idx == 0:

                c[ii].add_cartoon(selection="protein")
                c[ii].add_licorice(selection='33')
                c[ii].add_licorice(selection="protein",opacity=0.25)
#                 c[ii].add_surface(selection="protein", opacity=0.3) # Not often used

            c[ii].add_licorice(selection=ligand_res,color=colors[idx%6],opacity=0.6)
#             c[ii].add_licorice(selection=ligand_res,color=colors[idx%6],opacity=0.8)
#             c[ii].add_ball_and_stick(selection=ligand_res,opacity=1)

    # Reference pose
        c2 = view.add_trajectory(ref_pose[-1])
        c2.clear()
        c2.add_ball_and_stick(selection=ligand_res,width=0.5)

        
        del rmsd_cluster
        return cluster_selection, rmsd_selection, view, comp, actualComp
    else:
        return cluster_selection, rmsd_selection, [], comp, actualComp
    


