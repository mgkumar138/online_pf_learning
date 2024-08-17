#%%
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# True function r(x)
def r(x, glambda, galpha=1.0,gsigma=0.05):
    return galpha * np.exp(-0.5 * ((x - glambda) / gsigma) ** 2)

# Model function phi(x) with parameters
def phi(x, alphas, lambdas, sigmas):
    phi_vals = np.zeros_like(x)
    for alpha, lambda_, sigma in zip(alphas, lambdas, sigmas):
        phi_vals += (alpha**2) * np.exp(-0.5 * ((x - lambda_) / sigma) ** 2)
    return phi_vals

def phi_ind(x, alphas, lambdas, sigmas):
    phi_vals = []
    for alpha, lambda_, sigma in zip(alphas, lambdas, sigmas):
        phi_vals.append((alpha**2) * np.exp(-0.5 * ((x - lambda_) / sigma) ** 2))
    return np.array(phi_vals)

def get_pvcorr(params_history, start, end, num):
    alphas_start = params_history[start, :N]
    lambdas_start = params_history[start, N:2*N]
    sigmas_start = params_history[start, 2*N:]

    xs = np.linspace(-1, 1, 1001)
    startpcs = phi_ind(xs, alphas_start, lambdas_start, sigmas_start).T
    startvec = startpcs.flatten()
    trials = np.linspace(start, end-1, num, dtype=int)
    startxcor = startpcs @ startpcs.T

    pv_corr = []
    rep_corr = []
    for i in trials:
        alphas_start = params_history[i, :N]
        lambdas_start = params_history[i, N:2*N]
        sigmas_start = params_history[i, 2*N:]

        endpcs = phi_ind(xs, alphas_start, lambdas_start, sigmas_start).T
        endvec = endpcs.flatten()
        R = np.corrcoef(startvec, endvec)[0, 1]
        pv_corr.append(R)

        endxcor = endpcs @ endpcs.T
        R_rep = np.corrcoef(startxcor.flatten(), endxcor.flatten())[0, 1]
        rep_corr.append(R_rep)
    return trials, pv_corr, rep_corr, startxcor, endxcor

# Objective function to minimize (negative log likelihood)
def objective(optim_params, x_data, r_data, optim_mask, full_params):
    full_params[optim_mask] = optim_params
    N = len(full_params) // 3
    alphas = full_params[:N]
    lambdas = full_params[N:2*N]
    sigmas = full_params[2*N:]
    phi_vals = phi(x_data, alphas, lambdas, sigmas)
    return np.sum((r_data - phi_vals) ** 2)

# Set optimization flags
optim_alpha = True
optim_lambda = True
optim_sigma = False

# True function
x_data_full = np.linspace(-1, 1, 1001)
r_data_full = r(x_data_full, glambda=0.5)

# Number of terms in the model
N = 64
ns = 0

# Initial guesses for alpha_i, lambda_i, sigma_i
initial_alphas = np.ones(N) * 0.5
initial_lambdas = np.linspace(-1, 1, N)
initial_sigmas = np.ones(N) * 0.05
initial_guess = np.concatenate([initial_alphas, initial_lambdas, initial_sigmas])

# Create optimization mask
optim_mask = np.concatenate([
    np.ones(N) if optim_alpha else np.zeros(N),
    np.ones(N) if optim_lambda else np.zeros(N),
    np.ones(N) if optim_sigma else np.zeros(N)
]).astype(bool)

# Filter initial_guess to include only parameters that are optimized
optim_initial_guess = initial_guess[optim_mask]

# Lists to store parameter updates and loss values
params_history = [initial_guess]  # Include initial guess
loss_history = []

# Number of iterations
iterations = 200
current_params = initial_guess


#%%
glambdas = [0.75, 0.0]

# Perform optimization iteratively
for i in range(iterations):


    # Generate random data points
    # x_data = np.random.uniform(-1, 1, 100)
    x_data = np.linspace(-1, 1, 1001)

    if i > iterations//2:
        glambda = glambdas[1]
    else:
        glambda = glambdas[0]
    r_data = r(x_data, glambda=glambda)
    
    # Perform a single optimization step
    result = minimize(objective, optim_initial_guess, args=(x_data, r_data, optim_mask, current_params),
                      options={'maxiter': 1, 'disp': False})
    optim_initial_guess = result.x
    
    # Update current_params with optimized values
    current_params[optim_mask] = optim_initial_guess
    
    # Store updated parameters and loss value
    params_history.append(current_params.copy())
    loss_history.append(objective(current_params[optim_mask], x_data, r_data, optim_mask, current_params))

    current_params += np.random.normal(0, ns, size=current_params.shape)
    print(i)

# Extract final parameter values
final_alphas = current_params[:N]
final_lambdas = current_params[N:2*N]
final_sigmas = current_params[2*N:]

print(f"Estimated alphas: {final_alphas}")
print(f"Estimated lambdas: {final_lambdas}")
print(f"Estimated sigmas: {final_sigmas}")
print(f"Number of iterations: {iterations}")

params_history = np.array(params_history)
alphas_history = params_history[:, :N]
lambdas_history = params_history[:, N:2*N]
sigmas_history = params_history[:, 2*N:]

plt.figure()
plt.hlines(xmin=-1, xmax=1, y=0, colors='k')
plt.plot(x_data_full, r_data_full, label='True function r(x)')
plt.plot(x_data_full, phi(x_data_full, initial_alphas, initial_lambdas, initial_sigmas), label='Init model phi(x)')
init_pcact = phi_ind(x_data_full, initial_alphas, initial_lambdas, initial_sigmas)
for n in range(N):
    plt.plot(x_data_full, init_pcact[n])
plt.legend()
plt.tight_layout()
plt.title('True function and Initial model')

# Plot the true function and the fitted model using the final parameters
plt.figure()
plt.hlines(xmin=-1, xmax=1, y=0, colors='k')
plt.plot(x_data_full, r_data_full, label='True function r(x)')
plt.plot(x_data_full, phi(x_data_full, final_alphas, final_lambdas, final_sigmas), label='Fitted model phi(x)')
final_pcact = phi_ind(x_data_full, final_alphas, final_lambdas, final_sigmas)
for n in range(N):
    plt.plot(x_data_full, final_pcact[n])
plt.legend()
plt.tight_layout()
plt.title('True function and Fitted model')

# Plot the parameter updates
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
for i in range(N):
    if optim_alpha:
        ax1.plot(alphas_history[:, i], label=f'Alpha {i+1}')
    if optim_lambda:
        ax2.plot(lambdas_history[:, i], label=f'Lambda {i+1}')
    if optim_sigma:
        ax3.plot(sigmas_history[:, i], label=f'Sigma {i+1}')
if optim_alpha:
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Alpha values')
    ax1.set_title('Alpha convergence')
    ax1.legend(loc='upper right')
if optim_lambda:
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Lambda values')
    ax2.set_title('Lambda convergence')
    ax2.legend(loc='upper right')
if optim_sigma:
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Sigma values')
    ax3.set_title('Sigma convergence')
    ax3.legend(loc='upper right')
plt.tight_layout()

# Plot the loss values
plt.figure()
plt.plot(loss_history, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss value')
plt.title('Loss convergence')
plt.legend()
plt.tight_layout()

trials, pv_corr, rep_corr, startxcor, endxcor = get_pvcorr(params_history, 100, iterations, num=100)

f, ax = plt.subplots()
ax.plot(trials, pv_corr, label='$\phi(t)$')
ax.plot(trials, rep_corr, label=r'$\phi(t)^\top\phi(t)$')
ax.set_xlabel('Trial')
ax.set_ylabel('Correlation')
ax.legend(frameon=False, fontsize=6)

f,ax = plt.subplots()
x = params_history[iterations//2-1, N:2*N]
y = params_history[iterations-1, N:2*N]
ax.plot(np.linspace(np.min(x),np.max(x),1000),np.linspace(np.min(y),np.max(y),1000), color='k')
indexes = np.where((x >= glambdas[0]-0.1) & (x <= glambdas[0]+0.1))[0]
values_in_x = x[indexes]
values_in_y = y[indexes]
ax.scatter(x, y)
ax.scatter(values_in_x, values_in_y, color='g')
ax.axvline(glambdas[0], color='r')
ax.axhline(glambdas[1], color='r')


plt.show()
# %%
