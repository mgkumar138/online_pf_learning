#%%

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"
import jax.numpy as jnp
from jax import grad, jit, vmap, random, nn, lax, config
import matplotlib.pyplot as plt
from backend import *
import numpy as np
config.update('jax_platform_name', 'cpu')
from jax.lib import xla_bridge
device = xla_bridge.get_backend().platform
print(device)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, required=False, help='episodes', default=50000)
parser.add_argument('--plr', type=float, required=False, help='plr', default=0.0001)
parser.add_argument('--llr', type=float, required=False, help='llr', default=0.0001)
parser.add_argument('--alr', type=float, required=False, help='alr', default=0.0001)
parser.add_argument('--slr', type=float, required=False, help='slr', default=0.0)
parser.add_argument('--gamma', type=float, required=False, help='gamma', default=0.9)
parser.add_argument('--npc', type=int, required=False, help='npc', default=13)
parser.add_argument('--seed', type=int, required=False, help='seed', default=0)
parser.add_argument('--balpha', type=int, required=False, help='balpha', default=0)
parser.add_argument('--bsigma', type=int, required=False, help='bsigma', default=0)
parser.add_argument('--rloc', type=float, required=False, help='rloc', default=0.0)
args, unknown = parser.parse_known_args()


# training params
train_episodes = args.episodes
tmax = 100

# env pararms
envsize = 1
maxspeed = 0.1
goalsize = 0.025
startcoord = [-0.75]
goalcoord = [args.rloc]
seed = args.seed
initvelocity = 0.0
max_reward = 5

#agent params
npc = args.npc
sigma = 0.1
alpha = 1.0
nact = 2

actor_eta = args.plr
critic_eta = args.plr
pc_eta = args.llr
sigma_eta = args.slr
constant_eta = args.alr
etas = [pc_eta, sigma_eta,constant_eta, actor_eta,critic_eta]
betas = [0.1,args.balpha, args.bsigma]
gamma = args.gamma

savevar = True

# load data
load_exptname = f'1D_a2c_{npc}n_{seed}s_{50000}e_{goalsize}gs_{actor_eta}plr_{pc_eta}llr_{constant_eta}alr_{sigma_eta}slr_{args.balpha}ba_{args.bsigma}bs'
figdir = './fig/'
datadir = './data/'

if not os.path.exists(datadir):
    os.makedirs(datadir)

if not os.path.exists(figdir):
    os.makedirs(figdir)

[load_logparams, _, load_latencys, _] = saveload(datadir+load_exptname, 1, 'load')
params = load_logparams[-1]

exptname = f'new{goalcoord[0]}_1D_a2c_{npc}n_{seed}s_{train_episodes}e_{goalsize}gs_{actor_eta}plr_{pc_eta}llr_{constant_eta}alr_{sigma_eta}slr_{args.balpha}ba_{args.bsigma}bs'

initparams = params.copy()
initpcacts = plot_place_cells(initparams, startcoord=startcoord, goalcoord=goalcoord,goalsize=goalsize, title='Fields before learning',envsize=envsize)

lr_rates = get_learning_rate(1.0, 1.0,train_episodes)

#%%
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

        #onehotg = jnp.array([0,1,0])

        newstate, reward, done = env.step(onehotg) 

        coords.append(state)
        actions.append(onehotg)
        rewards.append(reward)

        state = newstate.copy()

        totR += reward

        if done:
            break

    return jnp.array(coords), jnp.array(rewards), jnp.array(actions), t

losses = []
latencys = []
allcoords = []
logparams = []
logparams.append(params)

env = OneDimNav(startcoord=startcoord, goalcoord=goalcoord, goalsize=goalsize, tmax=tmax, 
                maxspeed=maxspeed,envsize=envsize, nact=nact, initvelocity=initvelocity, max_reward=max_reward)

for episode in range(train_episodes):

    coords, rewards, actions, latency = run_trial(params, env, episode)
        
    discount_rewards = get_discounted_rewards(rewards, gamma=gamma)

    etas = [eta*lr_rates[episode] for eta in etas]
    
    params, grads,loss = update_a2c_params(params, coords, actions, discount_rewards, etas, betas)

    allcoords.append(coords)
    logparams.append(params)
    latencys.append(latency)
    losses.append(loss)

    print(f'Trial {episode+1}, G {np.sum(discount_rewards):.3f}, t {latency}, L {loss:.3f}')


if savevar:
    saveload(datadir+exptname, [logparams, allcoords, latencys, losses], 'save')


#%%
env.plot_trajectory()

initpcacts = plot_place_cells(initparams, startcoord=startcoord, goalcoord=goalcoord,goalsize=goalsize, title='Fields before learning',envsize=envsize)
# plt.savefig(figdir+'initpc_'+exptname+'.png')
pcacts = plot_place_cells(params, startcoord=startcoord, goalcoord=goalcoord,goalsize=goalsize, title='Fields after learning',envsize=envsize)
# plt.savefig(figdir+'trainpc_'+exptname+'.png')


plt.figure(figsize=(8,4))
plt.subplot(231)
plt.plot(latencys)
ma = moving_average(latencys, 20)
plt.plot(ma)
plt.xlabel('Trial')
plt.ylabel('Latency (Steps)')
#plt.xscale('log')
print(ma[-1])


plt.subplot(232)
plt.plot(params[4])
plt.xlabel('Place Cell')
plt.ylabel('Value')
plt.subplot(233)
plt.imshow(params[3],aspect='auto')
plt.xlabel('Action')
plt.xticks(np.arange(nact),env.onehot2dirmat)
plt.ylabel('Place Cell')
plt.yticks(np.arange(npc),np.arange(npc,dtype=int))
plt.colorbar()
plt.tight_layout()

lambdas = []
for e in range(train_episodes):
    lambdas.append(logparams[e][0])
lambdas = np.array(lambdas)
plt.subplot(234)
for n in range(npc):
    plt.plot(lambdas[:,n], label=f'PF{n+1}')
plt.title(f'Field center with learning')
plt.xlabel('Trial')
plt.ylabel('$\lambda$')
plt.axhline(startcoord[0], color='g', linestyle='--')
plt.axhline(goalcoord[0], color='orange', linestyle='--')


# %%
