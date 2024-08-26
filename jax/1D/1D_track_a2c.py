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
parser.add_argument('--episodes', type=int, required=False, help='episodes', default=5000)
parser.add_argument('--plr', type=float, required=False, help='plr', default=0.001)
parser.add_argument('--llr', type=float, required=False, help='llr', default=0.001)
parser.add_argument('--alr', type=float, required=False, help='alr', default=0.001)
parser.add_argument('--slr', type=float, required=False, help='slr', default=0.00)
parser.add_argument('--gamma', type=float, required=False, help='gamma', default=0.9)
parser.add_argument('--npc', type=int, required=False, help='npc', default=13)
parser.add_argument('--seed', type=int, required=False, help='seed', default=2020)
parser.add_argument('--balpha', type=float, required=False, help='balpha', default=0.0)
parser.add_argument('--bsigma', type=float, required=False, help='bsigma', default=0.0)
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
nact = 3

actor_eta = args.plr
critic_eta = args.plr
pc_eta = args.llr
sigma_eta = args.slr
constant_eta = args.alr
etas = [pc_eta, sigma_eta,constant_eta, actor_eta,critic_eta]
betas = [0.1,args.balpha, args.bsigma]
gamma = args.gamma

savevar = False
savefig = False
exptname = f'1D_a2c_{nact}a_{npc}n_{seed}s_{train_episodes}e_{goalsize}gs_{actor_eta}plr_{pc_eta}llr_{constant_eta}alr_{sigma_eta}slr_{args.balpha}ba_{args.bsigma}bs'
figdir = './fig/'
datadir = './data/'

if not os.path.exists(datadir):
    os.makedirs(datadir)

if not os.path.exists(figdir):
    os.makedirs(figdir)


params = uniform_pc_weights(npc, nact, seed, sigma=sigma, alpha=alpha, envsize=envsize)
# ridx = 9
# indx = np.zeros(npc)
# indx[ridx] = 1
# params[2] += indx

initparams = params.copy()
initpcacts = plot_place_cells(initparams, startcoord=startcoord, goalcoord=goalcoord,goalsize=goalsize, title='Fields before learning',envsize=envsize)

lr_rates = get_learning_rate(1.0, 1.0,train_episodes)

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


#%%
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
if savefig:
    plt.savefig(figdir+'initpc_'+exptname+'.png')
pcacts = plot_place_cells(params, startcoord=startcoord, goalcoord=goalcoord,goalsize=goalsize, title='Fields after learning',envsize=envsize)
if savefig:
    plt.savefig(figdir+'trainpc_'+exptname+'.png')


plt.figure(figsize=(8,4))
plt.subplot(231)
plt.plot(latencys)
ma = moving_average(latencys, 20)
plt.plot(ma)
plt.xlabel('Trial')
plt.ylabel('Latency (Steps)')
plt.xscale('log')
plt.title(ma[-1])

# plt.subplot(232)
# plt.plot(losses)
# mal = moving_average(losses, 20)
# plt.plot(mal)
# plt.xlabel('Trial')
# plt.ylabel('Loss')
# print(mal[-1])

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

gap = end = 10
bins = 20

indexs = np.logspace(np.log10(end), np.log10(train_episodes),21,dtype=int)
Rs = []
for i in indexs:
    visits, freq, density, corr = get_1D_freq_density_corr(allcoords, logparams, end=i, gap=gap, bins=bins)
    Rs.append(corr)
    
plt.subplot(236)
plt.plot(indexs, Rs, marker='o',color='gray')
slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(indexs).reshape(-1), np.array(Rs).reshape(-1))
regression_line = slope * np.array(indexs).reshape(-1) + intercept
plt.plot(np.array(indexs).reshape(-1), regression_line, color='red', label=f'R:{np.round(r_value,3)}, P:{np.round(p_value,3)}')
plt.legend(frameon=False, fontsize=6)
plt.title('f(x):d(x)')
plt.xlabel('Trial')
plt.ylabel('Correlation')
plt.xscale('log')

trials, pv_corr = get_pvcorr(logparams, train_episodes//2, train_episodes, num=21)

plt.subplot(235)
plt.plot(trials, pv_corr)
plt.xlabel('Trial')
plt.ylabel('PV Corr')
plt.xscale('log')
plt.tight_layout()
if savefig:
    plt.savefig(figdir+'analysis_'+exptname+'.png')

visits, freq, density, corr = get_1D_freq_density_corr(allcoords, logparams, end=end, gap=gap, bins=bins)
plot_freq_density_corr(visits, freq, density,title=f'Before learning to navigate in 1D: Trials {end-gap}-{end}')
if savefig:
    plt.savefig(figdir+'init_fxdx_'+exptname+'.png')

visits, freq, density, corr = get_1D_freq_density_corr(allcoords, logparams, end=train_episodes, gap=gap, bins=bins)
plot_freq_density_corr(visits, freq, density,title=f'After learning to navigate in 1D: Trials {train_episodes-gap}-{train_episodes}')
if savefig:
    plt.savefig(figdir+'train_fxdx_'+exptname+'.png')

# %%
