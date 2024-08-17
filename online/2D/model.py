import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import os
import csv
import matplotlib.cm as cm


# main agent description
def uniform_pc_weights(npc, nact,seed,sigma=0.1, alpha=1,envsize=1):
    pc_cent =  np.linspace(-envsize,envsize,npc) 
    pc_sigma = np.ones(npc)*sigma
    pc_constant = np.ones(npc)*alpha 
    
    return [np.array(pc_cent), np.array(pc_sigma), np.array(pc_constant), 
    1e-5 * np.random.normal(size=(npc,nact)), 1e-5 * np.random.normal(size=(npc,1))]

def random_pc_weights(npc, nact,seed,sigma=0.1, alpha=1,envsize=1):
    pc_cent =  np.linspace(-envsize,envsize,npc) 
    pc_sigma = np.ones(npc)*sigma
    np.random.seed(seed)
    pc_constant = np.random.uniform(0, alpha,size=npc)
    #pc_constant /= np.max(pc_constant)*alpha
    
    return [np.array(pc_cent), np.array(pc_sigma), np.array(pc_constant), 
    1e-5 * np.random.normal(size=(npc,nact)), 1e-5 * np.random.normal(size=(npc,1))]


def predict_placecell(params, x):
    pc_centers, pc_sigmas, pc_constant, actor_weights,critic_weights = params
    exponent = ((x-pc_centers)/pc_sigmas)**2
    pcact = np.exp(-0.5*exponent) * pc_constant**2
    return pcact

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

def predict_action_prob(params, pcact, beta=2):
    pc_centers, pc_sigmas, pc_constant, actor_weights,critic_weights = params
    actout = np.matmul(pcact, actor_weights)
    aprob = softmax(beta * actout)
    return aprob

def get_onehot_action(prob, nact=2):
    A = np.random.choice(a=np.arange(nact), p=np.array(prob))
    onehotg = np.zeros(nact)
    onehotg[A] = 1
    return onehotg

def learn(params, reward, newstate,state, onehotg,aprob, gamma, etas,balpha=0.0, noise=0.0, paramsindex=[], beta=2):
    
    pcact = predict_placecell(params, state)
    newpcact = predict_placecell(params, newstate)
    td = (reward + gamma * predict_value(params, newpcact) - predict_value(params, pcact))[0]
    l1_grad = balpha * np.sign(params[2])

    # get critic grads
    dcri = pcact[:,None] * td

    # get actor grads
    decay = beta * (onehotg[:,None]- aprob[:,None])  # derived from softmax grads
    # decay = onehotg[:,None]  # foster et al. 2000 rule, simplified form of the derivative
    dact = (pcact[:,None] @ decay.T) * td

    # get phi grads: dp = phi' (W^actor @ act + W^critic) * td
    post_td = (params[3] @ decay + params[4]) * td

    dpcc = (post_td * (pcact[:,None]) * ((state - params[0])/params[1]**2)[:,None])[:,0]
    dpcs = (post_td * (pcact[:,None]) * ((state - params[0])**2/params[1]**3)[:,None])[:,0]
    dpca = (post_td * (pcact[:,None]) * (2 / params[2][:,None]) - l1_grad)[:,0]
    
    grads = [dpcc, dpcs, dpca, dact, dcri]

    #update weights
    for p in range(len(params)):
        params[p] += etas[p] * grads[p]

    for p in paramsindex:
        ns = np.random.normal(size=params[p].shape) * noise
        params[p] += ns

    return params, td




# other PF agents descriptions

def predict_placecell_sigma(params, x):
    pc_centers, pc_sigmas, pc_constant, actor_weights,critic_weights = params
    exponent = ((x-pc_centers)/pc_sigmas)**2
    pcact = np.exp(-0.5*exponent) * (1/np.sqrt(2*np.pi*pc_sigmas**2))
    return pcact




def learn_sigma(params, reward, newstate,state, onehotg,aprob, gamma, etas,balpha=0.0, noise=0.0, paramsindex=[], beta=2, clip_sigma=[0.1, 2]):
    
    pcact = predict_placecell_sigma(params, state)
    newpcact = predict_placecell_sigma(params, newstate)
    td = (reward + gamma * predict_value(params, newpcact) - predict_value(params, pcact))[0]
    l1_grad = balpha * np.sign(params[2])

    # get critic grads
    dcri =  pcact[:,None] * td

    # get actor grads
    decay = beta * (onehotg[:,None]- aprob[:,None])  # derived from softmax grads
    # decay = onehotg[:,None]  # foster et al. 2000 rule, simplified form of the derivative
    dact = (pcact[:,None] @ decay.T) * td

    # get phi grads: dp = phi' (W^actor @ act + W^critic) * td
    post_td = (params[3] @ decay + params[4]) * td

    dpcc = (post_td * (pcact[:,None]) * ((state - params[0])/params[1]**2)[:,None])[:,0]
    dpcs = (post_td * (pcact[:,None]) * (((state - params[0])**2/params[1]**3) - (1/params[1]))[:,None])[:,0]
    # dpca = (post_td * (pcact[:,None]) * (2 / params[2][:,None]) - l1_grad)[:,0]
    
    grads = [dpcc, dpcs, 0, dact, dcri]

    #update weights
    for p in range(len(params)):
        params[p] += etas[p] * grads[p]

    for p in paramsindex:
        ns = np.random.normal(size=params[p].shape) * noise
        params[p] += ns

    # params[1] = np.clip(params[1],clip_sigma[0]/clip_sigma[1], clip_sigma[0]*clip_sigma[1])

    return params, td


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