#%%

import matplotlib.pyplot as plt
from utils import *
from envs import *
from model import *
import numpy as np
import os
from copy import deepcopy
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, required=False, help='episodes', default=50000)
parser.add_argument('--tmax', type=int, required=False, help='tmax', default=100)
parser.add_argument('--rmax', type=int, required=False, help='rmax', default=5)
parser.add_argument('--plr', type=float, required=False, help='plr', default=0.0001)
parser.add_argument('--clr', type=float, required=False, help='clr', default=0.00025)
parser.add_argument('--llr', type=float, required=False, help='llr', default=0.000) 
parser.add_argument('--alr', type=float, required=False, help='alr', default=0.000) 
parser.add_argument('--slr', type=float, required=False, help='slr', default=0.000)
parser.add_argument('--gamma', type=float, required=False, help='gamma', default=0.9)
parser.add_argument('--npc', type=int, required=False, help='npc', default=64)
parser.add_argument('--alpha', type=float, required=False, help='alpha', default=1.0)
parser.add_argument('--sigma', type=float, required=False, help='sigma', default=0.1)
parser.add_argument('--rsz', type=float, required=False, help='rsz', default=0.1)
parser.add_argument('--seed', type=int, required=False, help='seed', default=2020)
parser.add_argument('--pcinit', type=str, required=False, help='pcinit', default='uni')
parser.add_argument('--balpha', type=float, required=False, help='balpha', default=0.0)
parser.add_argument('--noise', type=float, required=False, help='noise', default=0.000)
parser.add_argument('--nact', type=int, required=False, help='nact', default=2)
parser.add_argument('--paramsindex', type=int,nargs='+', required=False, help='paramsindex', default=[0,1,2])
args, unknown = parser.parse_known_args()


# training params
train_episodes = args.episodes
tmax = args.tmax

# env pararms
envsize = 1
maxspeed = 0.1
goalsize = args.rsz
startcoord = [-0.75]
goalcoords = [0.5, 0.0, 0.75,-0.25, 0.5]
seed = args.seed
initvelocity = 0.0
max_reward = args.rmax

#agent params
npc = args.npc
sigma = args.sigma
alpha = args.alpha
nact = args.nact

# noise params
noise = args.noise
paramsindex = args.paramsindex
piname = ''.join(map(str, paramsindex))
pcinit = args.pcinit

actor_eta = args.plr
critic_eta = args.clr
pc_eta = args.llr
sigma_eta = args.slr
constant_eta = args.alr
etas = [pc_eta, sigma_eta,constant_eta, actor_eta,critic_eta]
gamma = args.gamma
balpha = args.balpha

savevar = False
savefig = False
savegif = False

exptname = f'1D_td_online_{balpha}ba_{noise}ns_{piname}p_{npc}n_{actor_eta}plr_{critic_eta}clr_{pc_eta}llr_{constant_eta}alr_{sigma_eta}slr_{pcinit}_{nact}a_{seed}s_{train_episodes}e_{max_reward}rmax_{goalsize}rsz'
figdir = './fig/'
datadir = './data/'
print(exptname)

if pcinit=='uni':
    params = uniform_pc_weights(npc, nact, seed, sigma=sigma, alpha=alpha, envsize=envsize)
elif pcinit == 'rand':
        params = random_pc_weights(npc, nact, seed, sigma=sigma, alpha=alpha, envsize=envsize)

initparams = deepcopy(params)
initpcacts = plot_place_cells(initparams, startcoord=startcoord, goalcoord=[goalcoords[0]],goalsize=goalsize, title='Fields before learning',envsize=envsize)

# inner loop training loop
def run_trial(params, env, trial):
    coords = []
    actions = []
    rewards = []
    tds = []

    state, goal, eucdist, done = env.reset()
    totR = 0
    
    for t in range(tmax):

        pcact = predict_placecell(params, state)

        aprob = predict_action_prob(params, pcact)

        onehotg = get_onehot_action(aprob, nact=nact)

        newstate, reward, done = env.step(onehotg) 

        params, td = learn(params, reward, newstate, state, onehotg,aprob, gamma, etas,balpha, noise, paramsindex)

        coords.append(state)
        actions.append(onehotg)
        rewards.append(reward)
        tds.append(td**2)

        state = newstate.copy()

        totR += reward

        if done:
            break

    return np.array(coords), np.array(rewards), np.array(actions),np.sum(tds), t, params


#%%
losses = []
latencys = []
allcoords = []
logparams = []
logparams.append(initparams)
cum_rewards = []

for goalcoord in goalcoords:
    env = OneDimNav(startcoord=startcoord, goalcoord=[goalcoord], goalsize=goalsize, tmax=tmax, 
                    maxspeed=maxspeed,envsize=envsize, nact=nact, initvelocity=initvelocity, max_reward=max_reward)

    for episode in range(train_episodes):

        coords, rewards, actions,tds, latency, params = run_trial(params, env, episode)

        discount_rewards = get_discounted_rewards(rewards, gamma)

        allcoords.append(coords)
        logparams.append(deepcopy(params))
        latencys.append(latency)
        losses.append(tds)
        cum_rewards.append(np.sum(discount_rewards))

        print(f'Goal {goalcoord}, Trial {episode+1}, G {np.sum(discount_rewards):.3f}, t {latency}, L {tds:.3f}, a {np.linalg.norm(params[2],ord=1):.3f}')



#%%
env.plot_trajectory()

f,score, drift = plot_analysis(logparams, latencys,cum_rewards, allcoords, stable_perf=0, exptname=exptname, rsz=goalsize)
print(score, drift)

for g,goal in enumerate(goalcoords):
    plot_pc(logparams, train_episodes*(g+1), goalcoord=[goal])

if savefig and seed == 0:
    f.savefig(figdir+exptname+'.svg')

if savevar:
    saveload(datadir+exptname, [logparams, latencys, allcoords], 'save')

# %%
if savegif:
    plot_gif(logparams, gif_name=f'{noise}ns_{piname}p_{balpha}ba_{pcinit}pc.gif', num_frames=250, duration=10)

