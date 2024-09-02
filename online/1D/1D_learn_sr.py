#%%

import matplotlib.pyplot as plt
from model import *
from envs import *
from utils import *
import numpy as np
import os
from copy import deepcopy
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, required=False, help='episodes', default=250)
parser.add_argument('--plr', type=float, required=False, help='plr', default=0.000)
parser.add_argument('--clr', type=float, required=False, help='clr', default=0.00)
parser.add_argument('--llr', type=float, required=False, help='llr', default=0.000)
parser.add_argument('--alr', type=float, required=False, help='alr', default=0.000)
parser.add_argument('--slr', type=float, required=False, help='slr', default=0.000)
parser.add_argument('--gamma', type=float, required=False, help='gamma', default=0.9)
parser.add_argument('--npc', type=int, required=False, help='npc', default=16)
parser.add_argument('--seed', type=int, required=False, help='seed', default=2020)
parser.add_argument('--pcinit', type=str, required=False, help='pcinit', default='uni')
parser.add_argument('--balpha', type=float, required=False, help='balpha', default=0.0)
parser.add_argument('--noise', type=float, required=False, help='noise', default=0.000)
parser.add_argument('--paramsindex', type=int,nargs='+', required=False, help='paramsindex', default=[])
args, unknown = parser.parse_known_args()

def plot_pcacts(pcacts, title, ax=None):
    if ax is None:
        f,ax = plt.subplots()

    for i in range(pcacts.shape[1]):
        ax.plot(xs, pcacts[:,i])
    ax.plot(xs, np.sum(pcacts,axis=1), color='k',linewidth=3)
    ax.hlines(xmin=-envsize,xmax=envsize, y=0, colors='k')
    ax.axvline(-0.75, color='g',linestyle='--',label='Start', linewidth=2)
    ax.axvline(0.5, color='r',linestyle='--',label='Goal', linewidth=2)
    #plt.ylim([-0.25, 1.25])
    ax.set_title(title)
    ax.set_ylabel('Tuning curves $\phi(x)$')
    ax.set_xlabel('Location (x)')

def relu(x):
    return np.maximum(x,0)

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
sigma = 0.05
alpha = 0.5
nact = 2

savevar = False
savefig = False
savegif = False

# load data
# exptname = f"1D_td_online_uni_0.0ns_p_2a_16n_0s_50000e_0.025gs_0.0001plr_0.0025clr_0.0001llr_0.0001alr_0.0slr"
exptname = f"./data/1D_td_online_0.0ba_0.0ns_01234p_16n_0.01plr_0.01clr_0.0001llr_0.0001alr_0.0001slr_uni_0.5a_0.05s_2a_2020s_50000e_5rmax_0.05rsz"
# [logparams, allcoords, latencys, losses] = saveload(datadir+exptname, 1, 'load')
[logparams, latencys, cumr, allcoords] = saveload(exptname, 1, 'load')

#%%
random_policy = logparams[0]
mid_policy = [logparams[0][0], logparams[0][1], logparams[0][2], logparams[12500-1][3], logparams[12500][4]]
optimal_policy = [logparams[0][0], logparams[0][1], logparams[0][2], logparams[-1][3], logparams[-1][4]]


# choose param 
lr = 1/train_episodes
gamma = 0.9 

# inner loop training loop
def run_trial(params, env, U):
    state, goal, eucdist, done = env.reset()

    
    for t in range(tmax):

        pcact = predict_placecell(params, state)

        aprob = predict_action_prob(params, pcact)

        onehotg = get_onehot_action(aprob, nact=nact)

        newstate, reward, done = env.step(onehotg) 

        # learn SR
        nextpcact = predict_placecell(params, newstate)

        pcact = pcact[:,None]
        nextpcact = nextpcact[:,None]

        M = relu(U) @ pcact
        M1 = relu(U) @ nextpcact

        td = pcact.T + gamma * M1 - M 
        delu = td * pcact

        U += lr * delu

        state = newstate.copy()

        if done:
            break

    return U

Us = []
ca1s = []

for params in [random_policy, mid_policy, optimal_policy]:
    env = OneDimNav(startcoord=startcoord, goalcoord=goalcoord, goalsize=goalsize, tmax=tmax, 
                    maxspeed=maxspeed,envsize=envsize, nact=nact, initvelocity=initvelocity, max_reward=max_reward)

    U = np.zeros([npc, npc])
    for episode in range(train_episodes):

        U = run_trial(params, env, U)

        print(f'Trial {episode+1}')
    
    Us.append(U)

    xs = np.linspace(-1,1,1001)
    ca3 = predict_batch_placecell(params, xs)

    ca1_sr = []
    for i in range(ca3.shape[0]):
        ca1_sr.append(relu(U) @ ca3[i])
    ca1_sr = np.array(ca1_sr)
    ca1s.append(ca1_sr)


#%%
f,axs = plt.subplots(2,3,figsize=(12,6))

# plot_pcacts(ca3, 'ca3', ax=axs[0,0])

plot_field_area(logparams, np.linspace(0, 50000, num=51, dtype=int), ax=axs[1,1])
axs[1,1].set_title('Field size increase during learning')

# change in field location
plot_field_center(logparams, np.linspace(0, 50000, num=51, dtype=int), ax=axs[1,2])
axs[1,2].set_title('Fields shift backward during learning')

plot_frequency(allcoords, [25, 12500, 50000], ax=axs[0,0], bins=21)
axs[0,0].set_title('Frequency dynamics')

plot_density(logparams, [25, 12500, 50000], ax=axs[0,2])
axs[0,2].set_title('Density learnd using RL')

plot_fxdx_trials(allcoords, logparams,np.linspace(25, 50000,dtype=int, num=51), ax=axs[1,0], gap=25)
axs[1,0].set_title('$f(x):d(x)$ correlation with learning')

ax = axs[0,1]
maxval = 0
for t, trial in enumerate([0,12500, 50000]):
    pcacts = ca1s[t]
    dx = np.sum(pcacts,axis=1)
    ax.plot(xs, dx, label=f'T={trial}')
    maxval = max(maxval, np.max(dx) * 1.1)

ax.set_xlabel('Location (x)')
ax.set_ylabel('Density $d(x)$')
ax.legend(frameon=False, fontsize=6)
ax.set_title('Density learnd using SR')
ax.fill_betweenx(np.linspace(0,maxval), goalcoord[0]-goalsize, goalcoord[0]+goalsize, color='r', alpha=0.25, label='Target')
ax.axvline(startcoord[0],ymin=0, ymax=maxval, color='g',linestyle='--',label='Start', linewidth=2)
ax.hlines(xmin=-envsize,xmax=envsize, y=0, colors='k')



# plot_pcacts(ca1s[0], 'ca1_sr_random', ax=axs[1,0])
# plot_pcacts(ca1s[1], 'ca1_sr_mid', ax=axs[1,1])
# plot_pcacts(ca1s[2], 'ca1_sr_optimal',ax=axs[1,2])

# field area
# areac = np.trapz(ca3,axis=0)
# area0 = np.trapz(ca1s[0],axis=0)
# area1 = np.trapz(ca1s[1],axis=0)

# estimate parameters
# fit_params = []
# fit_params.append(logparams[0])
# fit_param = [[],[],[], [], []]
# init_params = logparams[0]
# for n in range(npc):
#     initial_guess = [init_params[2][n], init_params[0][n], init_params[1][n]]
#     popt, pcov = curve_fit(gaussian_rbf, xs, np.array(ca1s[1])[:,n], p0=initial_guess)
#     fit_param[0].append(popt[1])
#     fit_param[1].append(popt[2])
#     fit_param[2].append(popt[0])

# for p in fit_param:
#     fit_param.append(np.array(p))
# fit_params.append(fit_param)

# f,axs = plt.subplots(1,2,figsize=(7, 2.5))
# plot_field_area(fit_params, np.linspace(0, 1, num=2, dtype=int), ax=axs[0])

# # change in field location
# plot_field_center(fit_params, np.linspace(0, 1, num=2, dtype=int), ax=axs[1])



# exptname = "1D_td_online_uni_0.0ns_p_2a_16n_2020s_50000e_0.025gs_0.0001plr_0.0025clr_0.0001llr_0.0001alr_0.0slr"
# datadir = './data/'
# [logparams, allcoords, latencys, losses] = saveload(datadir+exptname, 1, 'load')
# random_params = logparams[0]
# mid_params = logparams[12500]
# optimal_params = logparams[-1]

# plot_pcacts(predict_batch_placecell(random_params, xs), 'rl_init', ax=axs[2,0])
# plot_pcacts(predict_batch_placecell(mid_params, xs), 'rl_mid', ax=axs[2,1])
# plot_pcacts(predict_batch_placecell(optimal_params, xs), 'rl_learned', ax=axs[2,2])

# plot_field_area(logparams, np.linspace(0, train_episodes, num=51, dtype=int), ax=axs[3,0])

# # change in field location
# plot_field_center(logparams, np.linspace(0, train_episodes, num=51, dtype=int), ax=axs[3,1])

f.tight_layout()
# f.savefig('rl_vs_sr.svg')
# %%
