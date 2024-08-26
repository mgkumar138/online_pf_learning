#%%

import numpy as np
import matplotlib.pyplot as plt
from backend import *
import pickle

def convert_to_numpy(param):
    if isinstance(param, (np.ndarray, jnp.ndarray)):
        return np.array(param)
    elif isinstance(param, list):
        return [convert_to_numpy(p) for p in param]
    else:
        return np.array(param)

def convert_to_jax(param):
    if isinstance(param, np.ndarray):
        return jnp.array(param)
    elif isinstance(param, list):
        return [convert_to_jax(p) for p in param]
    else:
        return jnp.array(param)

def saveload_jax(filename, variables, opt):
    if opt == 'save':
        # Convert each variable in the list to NumPy parameters
        variables_np = [convert_to_numpy(variable) for variable in variables]
        with open(f"{filename}.pickle", "wb") as file:
            pickle.dump(variables_np, file)
        print('file saved')
    else:
        with open(f"{filename}.pickle", "rb") as file:
            variables_np = pickle.load(file)
        # Convert each variable in the list back to Jax parameters
        variables_jax = [convert_to_jax(variable) for variable in variables_np]
        return variables_jax
    

# import data

npc = 13
seed = 0
train_episodes = 100000
exptname = f'1D_pg_{npc}n_{seed}s_{train_episodes}e'

[logparams, allcoords, latencys] = saveload('./data/jax_'+exptname, 1, 'load')

# %%
