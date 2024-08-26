#%%

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"
import jax.numpy as jnp
import matplotlib.pyplot as plt
from backend import *
import numpy as np
from jax import config
config.update('jax_platform_name', 'cpu')
from jax.lib import xla_bridge
device = xla_bridge.get_backend().platform
print(device)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, required=False, help='episodes', default=10000)
parser.add_argument('--plr', type=float, required=False, help='plr', default=0.001)
parser.add_argument('--llr', type=float, required=False, help='llr', default=0.001)
parser.add_argument('--alr', type=float, required=False, help='alr', default=0.001)
parser.add_argument('--slr', type=float, required=False, help='slr', default=0.001)
parser.add_argument('--gamma', type=float, required=False, help='gamma', default=0.9)
parser.add_argument('--npc', type=int, required=False, help='npc', default=16)
parser.add_argument('--seed', type=int, required=False, help='seed', default=2020)
parser.add_argument('--pcinit', type=str, required=False, help='pcinit', default='uni')
parser.add_argument('--balpha', type=float, required=False, help='balpha', default=0.0)
parser.add_argument('--noise', type=float, required=False, help='noise', default=0.000)
parser.add_argument('--paramsindex', type=int,nargs='+', required=False, help='paramsindex', default=[])
args, unknown = parser.parse_known_args()


# training params
train_episodes = args.episodes
tmax = 100

# env pararms
envsize = 1
maxspeed = 0.1
goalsize = 0.05
startcoord = [-0.75]
goalcoord = [0.5]
seed = args.seed
initvelocity = 0.0
max_reward = 5

#agent params
npc = args.npc
sigma = 0.1
alpha = 1.0
nact = 2

# noise params
noise = args.noise
paramsindex = args.paramsindex
piname = ''.join(map(str, paramsindex))
pcinit = args.pcinit

actor_eta = args.plr
critic_eta = 0.0
pc_eta = args.llr
sigma_eta = args.slr
constant_eta = args.alr
etas = [pc_eta, sigma_eta,constant_eta, actor_eta,critic_eta]
betas = [0.0, args.balpha]
gamma = args.gamma

savevar = False
savefig = False
exptname = f'1D_pg__{pcinit}_{noise}ns_{piname}p_{nact}a_{npc}n_{seed}s_{train_episodes}e_{goalsize}gs_{actor_eta}plr_{pc_eta}llr_{constant_eta}alr_{sigma_eta}slr_{args.balpha}ba'
figdir = './fig/'
datadir = './data/'

if not os.path.exists(datadir):
    os.makedirs(datadir)

if not os.path.exists(figdir):
    os.makedirs(figdir)


if pcinit=='uni':
    params = uniform_pc_weights(npc, nact, seed, sigma=sigma, alpha=alpha, envsize=envsize)
elif pcinit == 'rand':
        params = random_pc_weights(npc, nact, seed, sigma=sigma, alpha=alpha, envsize=envsize)

initparams = params.copy()
initpcacts = plot_place_cells(initparams, startcoord=startcoord, goalcoord=goalcoord,goalsize=goalsize, title='Fields before learning',envsize=envsize)

# inner loop training loop
def run_trial(params, env, trial):
    coords = []
    actions = []
    rewards = []

    state, goal, eucdist, done = env.reset()
    totR = 0
    
    for t in range(tmax):

        pcact = predict_placecell(params, state)

        aprob = predict_action(params, pcact)

        onehotg = get_onehot_action(aprob, nact=nact)

        newstate, reward, done = env.step(onehotg) 

        coords.append(state)
        actions.append(onehotg)
        rewards.append(reward)

        state = newstate.copy()

        totR += reward

        if done:
            break

    return jnp.array(coords), jnp.array(rewards), jnp.array(actions), t


#%%
losses = []
latencys = []
allcoords = []
logparams = []
logparams.append(initparams)

env = OneDimNav(startcoord=startcoord, goalcoord=goalcoord, goalsize=goalsize, tmax=tmax, 
                maxspeed=maxspeed,envsize=envsize, nact=nact, initvelocity=initvelocity, max_reward=max_reward)

for episode in range(train_episodes):

    coords, rewards, actions, latency = run_trial(params, env, episode)
        
    discount_rewards = get_discounted_rewards(rewards, gamma=gamma)
    
    params, grads,loss = update_params(params, coords, actions, discount_rewards, etas, betas)

    allcoords.append(coords)
    logparams.append(params)
    latencys.append(latency)
    losses.append(loss)

    print(f'Trial {episode+1}, G {np.sum(discount_rewards):.3f}, t {latency}, L {loss:.3f}, a {np.min(params[2]):.3f}')


if savevar:
    saveload(datadir+exptname, [logparams, allcoords, latencys, losses], 'save')


#%%
env.plot_trajectory()

f = plot_analysis(logparams, latencys, allcoords, stable_perf=10000)

if savefig:
    f.savefig(figdir+'analysis_'+exptname+'.png')


# %%
