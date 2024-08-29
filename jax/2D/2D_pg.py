#%%

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"
import jax.numpy as jnp
import matplotlib.pyplot as plt
from backend import *
import numpy as np

# inner loop
train_episodes = 1000
tmax = 300

actor_eta = 0.01
critic_eta = 0.00
pc_eta = 0.0001
sigma_eta = 0.0001
constant_eta = 0.0001
normreward = False
etas = [pc_eta, sigma_eta,constant_eta, actor_eta,critic_eta]

gamma = 0.95
goalsize = 0.05
startcoord = [[-0.75,-0.75]]
goalcoord = [0.5,0.5]
seed = 2021
obstacles = False

# outer loop
npc = 8**2
sigma = 0.1
alpha = 0.5
nact = 4
pctype = 'uni'

exptname = f'2D_PG_{npc}n_{seed}'

savevar = False

params = uniform_2D_pc_weights(npc, nact, seed, sigma=sigma,alpha=alpha)
initparams = params.copy()

print(params[1])
# print(jnp.linalg.det(params[1]))
# print(jnp.linalg.inv(params[1])[0])

#%%
# plot_place_cells(initparams, num=np.arange(npc), title='Fields before training',goalcoord=goalcoord, obstacles=obstacles, goalsize=goalsize)
# plot_2D_density(initparams, title='Fields before training')

# inner loop training loop
def run_trial(params, env, trial):
    coords = []
    actions = []
    rewards = []

    state, goal, reward, done = env.reset()

    for i in range(tmax):

        pcact = predict_placecell(params, state)

        aprob = predict_action(params, pcact)

        onehotg = get_onehot_action(aprob)

        newstate, reward, done = env.step(onehotg) 
        
        coords.append(state)
        actions.append(onehotg)
        rewards.append(reward)

        state = newstate.copy()

        if done:
            #print(f'Target reached at {i} step')
            break

    return jnp.array(coords), jnp.array(rewards), jnp.array(actions), i

latencys = []
allcoords = []
logparams = []
logparams.append(params)
env = NDimNav(startcoord=startcoord, goalcoord=goalcoord, goalsize=goalsize, tmax=tmax, obstacles=obstacles)

for episode in range(train_episodes):

    coords, rewards, actions, latency = run_trial(params, env, episode)
        
    discount_rewards = get_discounted_rewards(rewards, gamma=gamma)
    
    params, grads = update_params(params, coords, actions, discount_rewards, etas)

    allcoords.append(coords)
    logparams.append(params)
    latencys.append(latency)

    print(f'Trial {episode+1}, G {np.sum(discount_rewards)}, t {latency}, s {params[1][3]}')

env.plot_trajectory()

plt.figure(figsize=(4,2))
plt.plot(latencys)
plt.plot(moving_average(latencys, 20))
plt.xlabel('Trial')
plt.ylabel('Latency (Steps)')
perf = np.round(np.mean(latencys[-int(train_episodes*0.9):]),1)
print(perf)

plot_maps(params[3],params[4],env, npc)

# plot_place_cells(initparams, num=np.arange(npc), title='Fields before training',goalcoord=goalcoord, obstacles=obstacles, goalsize=goalsize)
# plot_2D_density(initparams, title='Density before training')
# plot_place_cells(params, num=np.arange(npc), title='Fields after training',goalcoord=goalcoord, obstacles=obstacles,goalsize=goalsize)
# plot_2D_density(params, title='Density after training')