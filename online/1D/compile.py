import numpy as np
from utils import *

analysis = 'npc'

if analysis == 'rmax':
    datadir = './data/rmax/'

    rmaxs = [1,2,3,4,5,6,7,8,9,10]
    npcs = [16, 64, 256]
    rszs = [0.01,0.025,0.05,0.1]
    sigmas = [0.01,0.025,0.05,0.1]
    seeds = 10
    episodes = 50000
    trial_len = 10

    latencys = np.zeros([len(npcs),len(sigmas),len(rmaxs),len(rszs), seeds, episodes])
    cumrs = np.zeros_like(latencys)
    dxs = np.zeros([len(npcs),len(sigmas),len(rmaxs),len(rszs), seeds, trial_len, 1001])
    delta_dxrs = np.zeros([len(npcs),len(sigmas),len(rmaxs),len(rszs), seeds, trial_len])

    for n,npc in enumerate(npcs):
        for sig, sigma in enumerate(sigmas):
            for r, rmax in enumerate(rmaxs):
                for z,rsz in enumerate(rszs):
                    for s in range(seeds):
                        exptname = f"both_1D_td_online_0.0ba_0.0ns_0p_{npc}n_0.01plr_0.01clr_0.0001llr_0.0001alr_0.0001slr_uni_0.5a_{sigma}s_2a_{s}s_{episodes}e_{rmax}rmax_{rsz}rsz"
                        [latencys[n,sig,r,z, s], cumrs[n,sig,r,z, s], trials, dxs[n,sig,r,z, s], delta_dxrs[n,sig,r,z, s]] = saveload(datadir+exptname, 1, 'load')


    saveload(f"./comp_data/rmax_compile_{episodes}e", [np.mean(latencys,axis=4),np.mean(cumrs,axis=4),np.mean(dxs,axis=4),np.mean(delta_dxrs,axis=4)], 'save')

if analysis == 'npc':
    datadir = './data/'

    rmaxs = [1,5,10]
    npcs = [4,8,16,32,64,128,256,512, 1024]
    rszs = [0.01,0.025,0.05,0.1]
    sigmas = [0.01,0.025,0.05,0.1]
    seeds = 10
    episodes = 50000
    trial_len = 10

    latencys = np.zeros([len(npcs),len(sigmas),len(rmaxs),len(rszs), seeds, episodes])
    cumrs = np.zeros_like(latencys)
    dxs = np.zeros([len(npcs),len(sigmas),len(rmaxs),len(rszs), seeds, trial_len, 1001])
    delta_dxrs = np.zeros([len(npcs),len(sigmas),len(rmaxs),len(rszs), seeds, trial_len])

    for n,npc in enumerate(npcs):
        for sig, sigma in enumerate(sigmas):
            for r, rmax in enumerate(rmaxs):
                for z,rsz in enumerate(rszs):
                    for s in range(seeds):
                        exptname = f"both_1D_td_online_0.0ba_0.0ns_0p_{npc}n_0.01plr_0.01clr_0.0001llr_0.0001alr_0.0001slr_uni_0.5a_{sigma}s_2a_{s}s_{episodes}e_{rmax}rmax_{rsz}rsz"
                        [latencys[n,sig,r,z, s], cumrs[n,sig,r,z, s], trials, dxs[n,sig,r,z, s], delta_dxrs[n,sig,r,z, s]] = saveload(datadir+exptname, 1, 'load')


    saveload(f"./comp_data/npc_compile_{episodes}e", [np.mean(latencys,axis=4),np.mean(cumrs,axis=4),np.mean(dxs,axis=4),np.mean(delta_dxrs,axis=4)], 'save')