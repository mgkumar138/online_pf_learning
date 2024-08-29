import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import os
import csv
import matplotlib.cm as cm


# main agent description
def uniform_2D_pc_weights_(npc, nact,seed=0,sigma=0.1, alpha=1,envsize=1):
    x = np.linspace(-envsize,envsize,int(npc**0.5))
    xx,yy = np.meshgrid(x,x)
    pc_cent = np.concatenate([xx.reshape(-1)[:,None],yy.reshape(-1)[:,None]],axis=1)
    pc_sigma = np.tile(np.eye(2),(npc,1,1))*sigma
    pc_constant = np.ones(npc) * alpha
    return [np.array(pc_cent), np.array(pc_sigma), np.array(pc_constant), 
    1e-5 * np.random.normal(size=(npc,nact)), 1e-5 * np.random.normal(size=(npc,1))]

def uniform_2D_pc_weights(npc, nact,seed=0,sigma=0.1, alpha=1,envsize=1):
    x = np.linspace(-envsize,envsize,int(npc**0.5))
    xx,yy = np.meshgrid(x,x)
    pc_cent = np.concatenate([xx.reshape(-1)[:,None],yy.reshape(-1)[:,None]],axis=1)
    inv_sigma = np.linalg.inv(np.tile(np.eye(2),(npc,1,1))*sigma)
    # pc_sigma = np.tile(np.ones([2,2]),(npc,1,1))*sigma
    pc_constant = np.ones(npc) * alpha
    return [np.array(pc_cent), np.array(inv_sigma), np.array(pc_constant), 
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
    pc_centers, inv_sigma, pc_constant, actor_weights, critic_weights = params
    diff = x - pc_centers  # Shape: (npc, dim)
    exponent = np.einsum('ni,nij,nj->n', diff, inv_sigma, diff)
    pcacts = np.exp(-0.5 * exponent) * pc_constant**2
    return pcacts

def predict_placecell_(params, x):
    pc_centers, pc_sigmas, pc_constant, actor_weights, critic_weights = params
    diff = x - pc_centers  # Shape: (npc, dim)
    inv_sigma = np.linalg.inv(pc_sigmas)  # Shape: (npc, dim, dim)
    exponent = np.einsum('ni,nij,nj->n', diff, inv_sigma, diff)
    pcacts = np.exp(-0.5 * exponent) * pc_constant**2
    return pcacts

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

def learn(params, reward, newstate, state, onehotg, aprob, gamma, etas, balpha=0.0, noise=0.0, paramsindex=[], beta=1):
    pc_centers, inv_sigma, pc_constant, actor_weights, critic_weights = params
    
    # Predict place cell activations
    pcact = predict_placecell(params, state)
    newpcact = predict_placecell(params, newstate)
    
    # Predict values
    value = np.dot(pcact, critic_weights)
    newvalue = np.dot(newpcact, critic_weights)
    td = reward + gamma * newvalue - value

    l1_grad = balpha * np.sign(pc_constant)
    
    # Critic grads
    dcri = pcact[:, None] * td
    
    # Actor grads
    decay = beta * (onehotg[:, None] - aprob[:, None])
    dact = np.dot(pcact[:, None], decay.T) * td
    
    # Grads for field parameters
    post_td = (actor_weights @ decay + critic_weights) * td

    df = state - pc_centers

    dpcc = post_td * pcact[:,None] * np.einsum('nji,nj->ni', inv_sigma, df)
    outer = np.einsum('nj,nk->njk',df,df)
    dpcs = -0.5 * (post_td * pcact[:,None])[:,:,None] * outer
    dpca = (post_td * pcact[:,None] * (2/pc_constant[:,None]) - l1_grad)[:,0]

    grads = [dpcc, dpcs, dpca, dact, dcri]  # dpcc needs to be transposed back
    
    for p in range(len(params)):
        params[p] += etas[p] * grads[p]
    
    for p in paramsindex:
        ns = np.random.normal(size=params[p].shape) * noise
        params[p] += ns
    
    return params, grads, td

def learn_(params, reward, newstate, state, onehotg, aprob, gamma, etas, balpha=0.0, noise=0.0, paramsindex=[], beta=1):
    pc_centers, pc_sigmas, pc_constant, actor_weights, critic_weights = params
    
    # Predict place cell activations
    pcact = predict_placecell(params, state)
    newpcact = predict_placecell(params, newstate)
    
    # Predict values
    value = np.dot(pcact, critic_weights)
    newvalue = np.dot(newpcact, critic_weights)
    td = reward + gamma * newvalue - value

    l1_grad = balpha * np.sign(pc_constant)
    
    # Critic grads
    dcri = pcact[:, None] * td
    
    # Actor grads
    decay = beta * (onehotg[:, None] - aprob[:, None])
    dact = np.dot(pcact[:, None], decay.T) * td
    
    # Grads for field parameters
    post_td = (actor_weights @ decay + critic_weights) * td

    df = state - pc_centers
    inv_sigma = np.linalg.inv(pc_sigmas)

    dpcc = post_td * pcact[:,None] * np.einsum('nji,nj->ni', inv_sigma, df)
    outer = np.einsum('nj,nk->njk',df,df)
    dpcs = 0.5 * (post_td * pcact[:,None])[:,:,None] * np.einsum('njl,njk,nik->nji',inv_sigma, outer, inv_sigma)
    dpca = (post_td * pcact[:,None] * (2/pc_constant[:,None]) - l1_grad)[:,0]

    grads = [dpcc, dpcs, dpca, dact, dcri]  # dpcc needs to be transposed back
    
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