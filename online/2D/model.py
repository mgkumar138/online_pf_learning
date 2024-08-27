import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import os
import csv
import matplotlib.cm as cm


# main agent description
def uniform_2D_pc_weights(npc, nact,seed=0,sigma=0.1, alpha=1,envsize=1):
    x = np.linspace(-envsize,envsize,int(npc**0.5))
    xx,yy = np.meshgrid(x,x)
    pc_cent = np.concatenate([xx.reshape(-1)[:,None],yy.reshape(-1)[:,None]],axis=1)
    pc_sigma = np.tile(np.eye(2),(npc,1,1))*sigma
    pc_constant = np.ones(npc) * alpha
    return [np.array(pc_cent), np.array(pc_sigma), np.array(pc_constant), 
    1e-5 * np.random.normal(size=(npc,nact)), 1e-5 * np.random.normal(size=(npc,1))]

def random_pc_weights(npc, nact,seed,sigma=0.1, alpha=1,envsize=1):
    x = np.linspace(-envsize,envsize,int(npc**0.5))
    xx,yy = np.meshgrid(x,x)
    pc_cent = np.concatenate([xx.reshape(-1)[:,None],yy.reshape(-1)[:,None]],axis=1)
    pc_sigma = np.tile(np.eye(2),(npc,1,1))*sigma

    np.random.seed(seed)
    pc_constant = np.random.uniform(0, alpha,size=npc)
    
    return [np.array(pc_cent), np.array(pc_sigma), np.array(pc_constant), 
    1e-5 * np.random.normal(size=(npc,nact)), 1e-5 * np.random.normal(size=(npc,1))]


def predict_placecell(params, x):
    pc_centers, pc_sigmas, pc_constant, actor_weights,critic_weights = params
    npcs = params[0].shape[0]
    pcacts = []
    inv_sigma = np.linalg.inv(pc_sigmas)
    for n in range(npcs):
        diff = (x[:,None]-pc_centers[n][:,None])
        exponent = diff.T @ inv_sigma[n] @ diff
        pcact = np.exp(-0.5*exponent) * pc_constant[n]**2
        pcacts.append(pcact)
    return np.vstack(pcacts)[:,0]

def predict_batch_placecell(params, xs):  
    pcacts = []  
    for x in xs:
        pcacts.append(predict_placecell(params, x))
    pcacts = np.array(pcacts)
    return pcacts


def predict_value(params, pcact):
    pc_centers, pc_sigmas, pc_constant, actor_weights,critic_weights = params
    value = np.matmul(pcact, critic_weights)
    return value

def softmax(x):
    x_max = np.max(x, axis=-1, keepdims=True)
    unnormalized = np.exp(x - x_max)
    return unnormalized/np.sum(unnormalized, axis=-1, keepdims=True)

def predict_action_prob(params, pcact, beta=1):
    pc_centers, pc_sigmas, pc_constant, actor_weights,critic_weights = params
    actout = np.matmul(pcact, actor_weights)
    aprob = softmax(beta * actout)
    return aprob

def get_onehot_action(prob, nact=4):
    A = np.random.choice(a=np.arange(nact), p=np.array(prob))
    onehotg = np.zeros(nact)
    onehotg[A] = 1
    return onehotg

def learn(params, reward, newstate,state, onehotg,aprob, gamma, etas,balpha=0.0, noise=0.0, paramsindex=[], beta=1):
    pc_centers, pc_sigmas, pc_constant, actor_weights, critic_weights = params
    pcact = predict_placecell(params, state)
    newpcact = predict_placecell(params, newstate)

    value = predict_value(params, pcact)
    newvalue = predict_value(params, newpcact)
    td = (reward + gamma * newvalue - value)[0]

    l1_grad = balpha * np.sign(params[2])

    # Critic grads
    dcri = pcact[:, None] * td

    # Actor grads
    decay = beta * (onehotg[:, None] - aprob[:, None])  # derived from softmax grads
    dact = (pcact[:, None] @ decay.T) * td

    # Grads for field parameters
    post_td = (actor_weights @ decay + critic_weights) * td
    dpcc = np.zeros_like(pc_centers)
    dpcs = np.zeros_like(pc_sigmas)
    dpca = np.zeros_like(pc_constant)

    inv_sigma = np.linalg.inv(pc_sigmas)
    for n in range(params[0].shape[0]):
        diff = state - pc_centers[n]

        exp_term = np.exp(-0.5 * diff.T @ inv_sigma[n] @ diff)
        
        # Gradient wrt. pc_cent
        dpcc[n] = post_td[n] * pc_constant[n]**2 * exp_term * inv_sigma[n] @ diff
        
        # Gradient wrt. pc_sigma
        diff_outer = np.outer(diff, diff)
        dpcs[n] = 0.5 * post_td[n] * pc_constant[n]**2 * exp_term * inv_sigma[n] @ diff_outer @ inv_sigma[n]
        
        # Gradient wrt. pc_constant
        dpca[n] = 2 * pc_constant[n] * exp_term * post_td[n] - l1_grad[n]
    
    grads = [dpcc, dpcs, dpca, dact, dcri]

    for p in range(len(params)):
        params[p] += etas[p] * grads[p]

    for p in paramsindex:
        ns = np.random.normal(size=params[p].shape) * noise
        params[p] += ns

    return params, grads, td


def get_discounted_rewards(rewards, gamma=0.9, norm=False):
    discounted_rewards = []
    cumulative = 0
    for reward in rewards[::-1]:
        cumulative = reward + gamma * cumulative  # discounted reward with gamma
        discounted_rewards.append(cumulative)
    discounted_rewards.reverse()
    if norm:
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-9)
    return discounted_rewards