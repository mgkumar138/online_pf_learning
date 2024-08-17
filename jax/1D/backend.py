import jax.numpy as jnp
from jax import grad, jit, vmap, random, nn, lax, value_and_grad
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

# Define the functions to fit
def linear(x, a, b):
    return a * x + b

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

def sigmoid(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))

# Function to fit the model
def fit_model(x, y, func_type='linear'):
    if func_type == 'linear':
        func = linear
    elif func_type == 'exp':
        func = exponential
    elif func_type == 'sigmoid':
        func = sigmoid
    else:
        raise ValueError("Unsupported function type. Choose from 'linear', 'exponential', or 'sigmoid'.")

    popt, _ = curve_fit(func, x, y)
    return popt, func

def plot_model_fit(x, y, func_type):
    plt.scatter(x, y)
    popt, func = fit_model(x, y, func_type)
    plt.plot(x, func(x, *popt), label=f'Fitted: {func_type}\nParams: {np.round(popt, 3)}', color='red')
    plt.legend(frameon=False, fontsize=6)
    plt.show()

def get_1D_fva_density_corr(allcoords, logparams, end, gap=25, bins=31, delta_t=1, end2=None):
    bins = np.linspace(-1, 1, bins)
    fx = []
    dx = []
    xs = []
    vx = []
    ax = []
    for g in range(gap-1):
        coord = allcoords[end-1-g].flatten()
        velocity = (coord[1:] - coord[:-1]) / delta_t
        coord = coord[:-1]  # Exclude the last coordinate because it doesn't have a corresponding velocity
        acceleration = (velocity[1:] - velocity[:-1]) / delta_t
        velocity = velocity[:-1]  # Exclude the last velocity because it doesn't have a corresponding acceleration
        coord = coord[:-1]  # Exclude the last coordinate because it doesn't have a corresponding acceleration

        frequency, x = np.histogram(coord, bins=bins)
        visits = x[:-1] + (x[1] - x[0]) / 2

        if end2 is None:
            end2 = end

        param = logparams[end2-1-g]
        pcacts = predict_batch_placecell(param, visits)
        density = np.sum(pcacts, axis=1)
        
        # Bin the velocities and accelerations
        bin_indices = np.digitize(coord, bins) - 1
        sum_velocity = np.zeros(len(bins) - 1)
        sum_acceleration = np.zeros(len(bins) - 1)
        counts = np.zeros(len(bins) - 1)

        for i in range(len(velocity)):
            if 0 <= bin_indices[i] < len(sum_velocity):
                sum_velocity[bin_indices[i]] += velocity[i]
                sum_acceleration[bin_indices[i]] += acceleration[i]
                counts[bin_indices[i]] += 1

        avg_velocity = np.zeros(len(sum_velocity))
        avg_acceleration = np.zeros(len(sum_acceleration))
        nonzero_bins = counts != 0
        avg_velocity[nonzero_bins] = sum_velocity[nonzero_bins] / counts[nonzero_bins]
        avg_acceleration[nonzero_bins] = sum_acceleration[nonzero_bins] / counts[nonzero_bins]
        
        fx.append(frequency)
        dx.append(density)
        xs.append(visits)
        vx.append(avg_velocity)
        ax.append(avg_acceleration)
    
    fx = np.array(fx)
    dx = np.array(dx)
    xs = np.array(xs)
    vx = np.array(vx)
    ax = np.array(ax)

    xs = np.mean(xs, axis=0)
    dx = np.mean(dx, axis=0)
    fx = np.mean(fx, axis=0)
    vx = np.mean(vx, axis=0)
    ax = np.mean(ax, axis=0)

    R_fd, pval_fd = stats.pearsonr(fx, dx)
    R_vd, pval_vd = stats.pearsonr(vx, dx)
    R_ad, pval_ad = stats.pearsonr(ax, dx)
    return xs, fx, dx, vx, ax, R_fd, pval_fd, R_vd, pval_vd, R_ad, pval_ad


def get_1D_freq_density_corr(allcoords, logparams, trial, gap=25, bins=31):
    bins = np.linspace(-1,1,bins)
    fx = []
    dx = []
    xs = []

    for g in range(gap):
        coord = allcoords[trial-g-1]
        #x = np.linspace(-1.0,1.0,bins+1)
        #coord = np.concatenate([coord[:,0], x],axis=0)
        frequency,x = np.histogram(coord, bins=bins)
        visits = x[:-1] + (x[1] - x[0])/2

        param = logparams[trial-g-1]
        pcacts = predict_batch_placecell(param, visits)
        density = np.sum(pcacts,axis=1)
    
        fx.append(frequency)
        dx.append(density)
        xs.append(visits)
    
    fx = np.array(fx)
    dx = np.array(dx)
    xs = np.array(xs)

    xs = np.mean(xs,axis=0)
    dx = np.mean(dx,axis=0)
    fx = np.mean(fx,axis=0)
    R,pval = stats.pearsonr(fx, dx)
    return xs, fx, dx, R, pval

def plot_fxdx_trials(allcoords, logparams, trials,gap, ax=None):
    if ax is None:
        f,ax = plt.subplots()
    
    Rs = []
    for trial in trials:
        visits, frequency, density, R, pval = get_1D_freq_density_corr(allcoords, logparams, trial, gap=gap)
        Rs.append(R)
    ax.plot(trials, Rs, marker='o')
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(trials).reshape(-1), np.array(Rs).reshape(-1))
    regression_line = slope * np.array(trials).reshape(-1) + intercept
    ax.plot(np.array(trials).reshape(-1), regression_line, color='red', label=f'R:{np.round(r_value, 3)}, P:{np.round(p_value, 3)}')
    ax.legend(frameon=False, fontsize=6)
    ax.set_title('Correlation with learning')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Correlation')


def plot_fx_dx(allcoords, logparams, trial, title,gap,ax=None):
    if ax is None:
        f,ax = plt.subplots()
    
    visits, frequency, density, R, pval = get_1D_freq_density_corr(allcoords, logparams, trial, gap=gap)
    ax.scatter(frequency, density)
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(frequency).reshape(-1), np.array(density).reshape(-1))
    regression_line = slope * np.array(frequency).reshape(-1) + intercept
    ax.plot(np.array(frequency).reshape(-1), regression_line, color='red', label=f'R:{np.round(r_value, 3)}, P:{np.round(p_value, 3)}')
    ax.legend(frameon=False, fontsize=6)
    ax.set_title(title)
    ax.set_xlabel('Frequency $f(x)$')
    ax.set_ylabel('Density $d(x)$')
    print(R, pval)
    print(r_value, p_value)

def plot_freq_density_corr(visits, frequency, density, title):
    plt.figure(figsize=(6, 2))
    plt.suptitle(title)
    plt.subplot(121)
    plt.plot(visits, (frequency - np.min(frequency)) / (np.max(frequency) - np.min(frequency)), label='Frequency')
    plt.plot(visits, (density - np.min(density)) / (np.max(density) - np.min(density)), label='Density')
    plt.xlabel('Location (x)')
    plt.ylabel('Norm value')
    plt.axvline(-0.75, color='g', linestyle='--', label='Start')
    plt.axvline(0.5, color='r', linestyle='--', label='Goal')
    plt.legend(frameon=False, fontsize=6)

    plt.subplot(122)
    plt.scatter(frequency, density)
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(frequency).reshape(-1), np.array(density).reshape(-1))
    regression_line = slope * np.array(frequency).reshape(-1) + intercept
    plt.plot(np.array(frequency).reshape(-1), regression_line, color='red', label=f'R:{np.round(r_value, 3)}, P:{np.round(p_value, 3)}')
    plt.legend(frameon=False, fontsize=6)
    plt.title('Correlation')
    plt.xlabel('Frequency f(x)')
    plt.ylabel('Density d(x)')
    plt.tight_layout()

def plot_metric_density_corr(visits, metric, density, title, metricname):

    plt.figure(figsize=(6,2))
    plt.suptitle(title)
    plt.subplot(121)
    plt.plot(visits, (metric-np.min(metric))/(np.max(metric)-np.min(metric)),label=f'{metricname}(x)')
    plt.plot(visits, (density-np.min(density))/(np.max(density)-np.min(density)), label='d(x)')
    plt.xlabel('Location (x)')
    plt.ylabel('Norm value')
    plt.axvline(-0.75, color='g',linestyle='--',label='Start')
    plt.axvline(0.5, color='r',linestyle='--',label='Goal')
    plt.legend(frameon=False, fontsize=6)

    plt.subplot(122)
    plt.scatter(metric, density)
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(metric).reshape(-1), np.array(density).reshape(-1))
    regression_line = slope * np.array(metric).reshape(-1) + intercept
    plt.plot(np.array(metric).reshape(-1), regression_line, color='red', label=f'R:{np.round(r_value,3)}, P:{np.round(p_value,3)}')
    plt.legend(frameon=False, fontsize=8)
    plt.title('Correlation')
    plt.xlabel(f'{metricname}(x)')
    plt.ylabel('d(x)')
    plt.tight_layout()

def flatten(xss):
    return np.array([x for xs in xss for x in xs],dtype=np.float32)

def random_pc_weights(npc, nact,seed,sigma=0.1, alpha=1,envsize=1):
    pc_cent =  jnp.linspace(-envsize,envsize,npc) 
    pc_sigma = jnp.ones(npc)*sigma
    actor_key, critic_key, constant_key = random.split(random.PRNGKey(seed), num=3)
    pc_constant = abs(random.normal(constant_key, (npc,)))
    pc_constant /= jnp.max(pc_constant)*alpha
    
    return [jnp.array(pc_cent), jnp.array(pc_sigma), jnp.array(pc_constant), 
    1e-5 * random.normal(actor_key, (npc,nact)), 1e-5 * random.normal(critic_key, (npc,1))]

def uniform_pc_weights(npc, nact,seed,sigma=0.1, alpha=1,envsize=1):
    pc_cent =  jnp.linspace(-envsize,envsize,npc) 
    pc_sigma = jnp.ones(npc)*sigma
    pc_constant = jnp.ones(npc)*alpha 
    
    actor_key, critic_key = random.split(random.PRNGKey(seed), num=2)
    return [jnp.array(pc_cent), jnp.array(pc_sigma), jnp.array(pc_constant), 
    1e-5 * random.normal(actor_key, (npc,nact)), 1e-5 * random.normal(critic_key, (npc,1))]

def predict_placecell(params, x):
    pc_centers, pc_sigmas, pc_constant, actor_weights,critic_weights = params
    exponent = ((x-pc_centers)/pc_sigmas)**2
    pcact = jnp.exp(-0.5*exponent) * pc_constant**2 #1/jnp.sqrt(2*jnp.pi*pc_sigmas**2)
    return pcact


def predict_value(params, pcact):
    pc_centers, pc_sigmas, pc_constant, actor_weights,critic_weights = params
    value = jnp.matmul(pcact, critic_weights)
    return value

def predict_action(params, pcact, beta=2):
    pc_centers, pc_sigmas, pc_constant, actor_weights,critic_weights = params
    actout = jnp.matmul(pcact, actor_weights)
    aprob = nn.softmax(beta * actout)
    return aprob

def pg_loss(params, coords, actions, discount_rewards, betas):
    aprobs = []
    for coord in coords:
        pcact = predict_placecell(params, coord)
        aprob = predict_action(params, pcact)
        aprobs.append(aprob)
    aprobs = jnp.array(aprobs)
    neg_log_likelihood = jnp.log(aprobs) * actions  # log probability of action as policy
    weighted_rewards = lax.stop_gradient(jnp.array(discount_rewards)[:,None])
    actor_loss = jnp.sum(jnp.array(neg_log_likelihood * weighted_rewards))  # log policy * discounted reward

    alpha_reg = -jnp.linalg.norm(params[2], ord=1) * (1/len(params[2]))
    sigma_reg = -jnp.linalg.norm(params[1], ord=2)**2 / (len(params[1])*0.1**2)
    tot_loss = actor_loss + betas[1] * alpha_reg + betas[2] * sigma_reg
    return tot_loss

@jit
def update_params(params, coords, actions, discount_rewards, etas, betas):
    loss, grads = value_and_grad(pg_loss)(params, coords,actions, discount_rewards, betas)
    pc_centers, pc_sigmas, pc_constant, actor_weights,critic_weights = params
    dpcc, dpcs, dpca, dact, dcri = grads

    # + for gradient ascent
    pc_eta, sigma_eta,constant_eta, actor_eta, critic_eta = etas
    newpc_centers = pc_centers + pc_eta * dpcc
    newpc_sigma = pc_sigmas + sigma_eta * dpcs
    newpc_const = pc_constant + constant_eta * dpca
    newactor_weights = actor_weights + actor_eta * dact
    newcritic_weights = critic_weights + critic_eta * dcri  # gradient descent
    return [newpc_centers, newpc_sigma,newpc_const, newactor_weights,newcritic_weights], grads, loss

def a2c_loss(params, coords, actions, discount_rewards, betas):
    aprobs = []
    values = []
    for coord in coords:
        pcact = predict_placecell(params, coord)
        aprob = predict_action(params, pcact)
        value = predict_value(params, pcact)
        aprobs.append(aprob)
        values.append(value)
    aprobs = jnp.array(aprobs)
    values = jnp.array(values)

    log_likelihood = jnp.log(aprobs) * actions  # log probability of action as policy
    advantage = jnp.array(discount_rewards)[:,None] - values

    actor_loss = jnp.sum(log_likelihood * lax.stop_gradient(advantage))  # log policy * discounted reward
    critic_loss = -jnp.sum(advantage ** 2) # grad decent
    alpha_reg = -jnp.linalg.norm(params[2], ord=1) * (1/len(params[2]))
    sigma_reg = -jnp.linalg.norm(params[1], ord=2)**2 / (len(params[1])*0.1**2)
    tot_loss = actor_loss + betas[0] * critic_loss + betas[1] * alpha_reg + betas[2] * sigma_reg
    return tot_loss

@jit
def update_a2c_params(params, coords, actions, discount_rewards, etas, betas):
    loss, grads = value_and_grad(a2c_loss)(params, coords,actions, discount_rewards, betas)
    pc_centers, pc_sigmas, pc_constant, actor_weights,critic_weights = params
    dpcc, dpcs, dpca, dact, dcri = grads

    # + for gradient ascent
    pc_eta, sigma_eta,constant_eta, actor_eta, critic_eta = etas
    newpc_centers = pc_centers + pc_eta * dpcc
    newpc_sigma = pc_sigmas + sigma_eta * dpcs
    newpc_const = pc_constant + constant_eta * dpca
    newactor_weights = actor_weights + actor_eta * dact
    newcritic_weights = critic_weights + critic_eta * dcri  # gradient descent
    return [newpc_centers, newpc_sigma,newpc_const, newactor_weights,newcritic_weights], grads, loss

def td_loss(params, coords, actions, rewards, gamma, betas):
    aprobs = []
    values = []
    for coord in coords:
        pcact = predict_placecell(params, coord)
        aprob = predict_action(params, pcact)
        value = predict_value(params, pcact)
        aprobs.append(aprob)
        values.append(value)
    aprobs = jnp.array(aprobs)
    values = jnp.array(values)

    log_likelihood = jnp.log(aprobs) * actions  # log probability of action as policy
    tde = jnp.array(compute_reward_prediction_error(rewards[:,None], values, gamma))

    actor_loss = jnp.sum(log_likelihood * lax.stop_gradient(tde))  # log policy * discounted reward
    critic_loss = -jnp.sum(tde ** 2) # grad decent
    alpha_reg = -jnp.linalg.norm(params[2], ord=1) * (1/len(params[2]))
    sigma_reg = -jnp.linalg.norm(params[1], ord=2)**2 / (len(params[1])*0.1**2)
    tot_loss = actor_loss + betas[0] * critic_loss + betas[1] * alpha_reg + betas[2] * sigma_reg
    return tot_loss

@jit
def update_td_params(params, coords, actions, rewards, etas, gamma, betas):
    loss, grads = value_and_grad(td_loss)(params, coords,actions, rewards, gamma, betas)
    pc_centers, pc_sigmas, pc_constant, actor_weights,critic_weights = params
    dpcc, dpcs, dpca, dact, dcri = grads

    # + for gradient ascent
    pc_eta, sigma_eta,constant_eta, actor_eta, critic_eta = etas
    newpc_centers = pc_centers + pc_eta * dpcc
    newpc_sigma = pc_sigmas + sigma_eta * dpcs
    newpc_const = pc_constant + constant_eta * dpca
    newactor_weights = actor_weights + actor_eta * dact
    newcritic_weights = critic_weights + critic_eta * dcri  # gradient descent
    return [newpc_centers, newpc_sigma,newpc_const, newactor_weights,newcritic_weights], grads, loss

def get_onehot_action(prob, nact=3):
    A = np.random.choice(a=np.arange(nact), p=np.array(prob))
    onehotg = np.zeros(nact)
    onehotg[A] = 1
    return onehotg

def get_discounted_rewards(rewards, gamma=0.95, norm=False):
    discounted_rewards = []
    cumulative = 0
    for reward in rewards[::-1]:
        cumulative = reward + gamma * cumulative  # discounted reward with gamma
        discounted_rewards.append(cumulative)
    discounted_rewards.reverse()
    if norm:
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-9)
    return discounted_rewards

def compute_reward_prediction_error(rewards, values, gamma=0.95):
    new_values = jnp.concatenate([values[1:], jnp.array([[0]])])
    td = rewards + gamma * new_values - values
    return td

def predict_batch_placecell(params, xs):  
    pcacts = []  
    for x in xs:
        pcacts.append(predict_placecell(params, x))
    pcacts = np.array(pcacts)
    return pcacts

def plot_analysis(logparams,latencys, allcoords, stable_perf):
    f, axs = plt.subplots(6,3,figsize=(10,15))
    total_trials = len(latencys)
    gap = 25

    #latency 
    plot_latency(latencys, ax=axs[0,0])

    plot_pc(logparams, 0,ax=axs[0,1], title='Before Learning')

    plot_pc(logparams, total_trials,ax=axs[0,2], title='After Learning')


    plot_value(logparams, [gap,total_trials//4, total_trials], ax=axs[1,0])

    plot_velocity(logparams,  [gap,total_trials//4,total_trials],ax=axs[1,1])



    ## high d at reward
    plot_density(logparams,  [gap,total_trials//4, total_trials], ax=axs[1,2])

    plot_frequency(allcoords,  [gap,total_trials//4,total_trials], ax=axs[2,2], gap=gap)


    plot_fx_dx(allcoords, logparams, gap,'Before Learning', ax=axs[2,0], gap=gap)

    plot_fx_dx(allcoords, logparams, total_trials,'After Learning', ax=axs[2,1], gap=gap)

    plot_fxdx_trials(allcoords, logparams, np.linspace(gap, total_trials,dtype=int, num=31), ax=axs[3,0], gap=gap)

    # change in field area
    plot_field_area(logparams, np.linspace(0, total_trials, num=51, dtype=int), ax=axs[3,1])

    # change in field location
    plot_field_center(logparams, np.linspace(0, total_trials, num=51, dtype=int), ax=axs[3,2])


    ## drift
    trials, pv_corr,rep_corr, startxcor, endxcor = get_pvcorr(logparams, stable_perf, total_trials, num=101)

    plot_rep_sim(startxcor, stable_perf, ax=axs[4,0])

    plot_rep_sim(endxcor, total_trials, ax=axs[4,1])
    
    plot_pv_rep_corr(trials, pv_corr, rep_corr,ax=axs[4,2])

    param_delta = get_param_changes(logparams, total_trials)
    plot_param_variance(param_delta, total_trials, stable_perf,axs=axs[5])

    f.tight_layout()
    return f


def get_param_changes(logparams, total_trials):

    lambdas = []
    sigmas = []
    alphas = []
    episodes = np.arange(0, total_trials)
    for e in episodes:
        lambdas.append(logparams[e][0])
        sigmas.append(logparams[e][1])
        alphas.append(logparams[e][2])
    lambdas = np.array(lambdas)
    sigmas = np.array(sigmas)
    alphas = np.array(alphas)
    return [lambdas, sigmas, alphas]

def plot_param_variance(param_change, total_trials, stable_perf,axs=None):
    if axs is None:
        f,axs = plt.subplots(nrows=1, ncols=3)
    [lambdas, sigmas, alphas] = param_change
    # Assuming `lambdas` is your T x N matrix
    variances = np.var(alphas[stable_perf:], axis=0)
    # Get indices of the top 10 variances
    top_indices = np.argsort(variances)[-10:][::-1]
    episodes = np.arange(0, total_trials)

    labels = [r'$\lambda$', r'$\sigma$',r'$\alpha$']
    for i, param in enumerate([lambdas, sigmas, alphas]):
        for n in top_indices:
            axs[i].plot(episodes[stable_perf:], param[stable_perf:,n])
        axs[i].set_xlabel('Trial')
        axs[i].set_ylabel(labels[i])

def plot_pv_rep_corr(trials, pv_corr, rep_corr,ax=None):
    if ax is None:
        f,ax = plt.subplots()
    ax.plot(trials, pv_corr,label='$\phi(t)$')
    ax.plot(trials, rep_corr,label=r'$\phi(t)^\top\phi(t)$')
    ax.set_xlabel('Trial')
    ax.set_ylabel('PV Corr')
    ax.legend(frameon=False, fontsize=6)

def plot_latency(latencys,ax=None, window=20):
    if ax is None:
        f,ax = plt.subplots()
    ax.plot(latencys)
    ma = moving_average(latencys, window)
    ax.plot(ma)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Latency (Steps)')
    #plt.xscale('log')
    ax.set_title(f'Last Latency: {np.round(ma[-1]):.0f}')

def plot_rep_sim(xcor,trial, ax=None):
    if ax is None:
        f,ax = plt.subplots()
    im = ax.imshow(xcor)
    plt.colorbar(im)
    ax.set_xlabel('Location (x)')
    ax.set_ylabel('Location (x)')
    idx = np.array([0,500,1000])
    ax.set_xticks(np.arange(1001)[idx], np.linspace(-1,1,1001)[idx])
    ax.set_title(f'T={trial}')

def plot_value(logparams, trials, goalcoord=[0.5], startcoord=[-0.75], goalsize=0.025, envsize=1, ax=None):
    if ax is None:
        f,ax = plt.subplots()
    xs = np.linspace(-1,1,1001)
    maxval  = 0 
    for trial in trials:
        pcacts = predict_batch_placecell(logparams[trial], xs)
        value = pcacts @ logparams[trial][4] 
        ax.plot(xs, value, label=f'T={trial}')
        
        maxval = max(maxval, np.max(value) * 1.1)
    ax.set_xlabel('Location (x)')
    ax.set_ylabel('Value v(x)')
    ax.legend(frameon=False, fontsize=6)
    ax.fill_betweenx(np.linspace(0,maxval), goalcoord[0]-goalsize, goalcoord[0]+goalsize, color='r', alpha=0.25, label='Target')
    ax.axvline(startcoord[0],ymin=0, ymax=maxval.item(), color='g',linestyle='--',label='Start', linewidth=2)
    ax.hlines(xmin=-envsize,xmax=envsize, y=0, colors='k')

def plot_field_area(logparams, trials,ax):
    if ax is None:
        f,ax = plt.subplots()
    areas = []
    for trial in trials:
        area = np.trapz(predict_batch_placecell(logparams[trial], np.linspace(-1,1,1001)),axis=0)
        areas.append(area)
    areas = np.array(areas)
    norm_area = areas/areas[0]

    ax.errorbar(trials, np.mean(norm_area,axis=1), np.std(norm_area,axis=1)/np.sqrt(len(logparams[0][0])), marker='o')
    ax.set_ylabel('Norm Field Area')
    ax.set_xlabel('Trial')

def plot_field_center(logparams, trials,ax):
    lambdas = []
    for trial in trials:
        lambdas.append(logparams[trial][0])
    lambdas = np.array(lambdas)
    norm_lambdas = lambdas-lambdas[0]
    ax.errorbar(trials, np.mean(norm_lambdas,axis=1), np.std(norm_lambdas,axis=1)/np.sqrt(len(logparams[0][0])), marker='o')
    ax.set_ylabel('Centered Field Center')
    ax.set_xlabel('Trial')



def plot_velocity(logparams, trials, goalcoord=[0.5], startcoord=[-0.75], goalsize=0.025, envsize=1, ax=None):
    if ax is None:
        f,ax = plt.subplots()

    xs = np.linspace(-1,1,1001)
    maxval  = 0 
    for trial in trials:
        pcacts = predict_batch_placecell(logparams[trial], xs)
        actout = pcacts @ logparams[trial][3] 
        aprob = nn.softmax(2 * actout)
        if logparams[0][3].shape[1] == 3:
            vel = np.matmul(aprob, np.array([[-1], [1], [0]]))
        else:
            vel = np.matmul(aprob, np.array([[-1], [1]]))
        vel = np.clip(vel, -1,1) * 0.1 

        ax.plot(xs, vel, label=f'T={trial}')
        maxval = max(maxval, np.max(vel) * 1.1)

    ax.set_xlabel('Location (x)')
    ax.set_ylabel(r'Velocity $\rho(x)$')
    ax.legend(frameon=False, fontsize=6)
    ax.fill_betweenx(np.linspace(0,maxval), goalcoord[0]-goalsize, goalcoord[0]+goalsize, color='r', alpha=0.25, label='Target')
    ax.axvline(startcoord[0],ymin=0, ymax=-0.1, color='g',linestyle='--',label='Start', linewidth=2)
    ax.hlines(xmin=-envsize,xmax=envsize, y=0, colors='k')
    

def plot_pc(logparams, trial,title='', ax=None, goalcoord=[0.5], startcoord=[-0.75], goalsize=0.025, envsize=1, ):
    if ax is None:
        f,ax = plt.subplots()

    xs = np.linspace(-1,1,1001)
    pcacts = predict_batch_placecell(logparams[trial], xs)
    maxval = 0
    for i in range(pcacts.shape[1]):
        ax.plot(xs, pcacts[:,i])
    ax.set_xlabel('Location (x)')
    ax.set_ylabel('Tuning curves $\phi(x)$')
    ax.set_title(title)
    maxval = max(maxval, np.max(pcacts) * 1.1)
    ax.fill_betweenx(np.linspace(0,maxval), goalcoord[0]-goalsize, goalcoord[0]+goalsize, color='r', alpha=0.25, label='Target')
    ax.axvline(startcoord[0],ymin=0, ymax=maxval, color='g',linestyle='--',label='Start', linewidth=2)
    ax.hlines(xmin=-envsize,xmax=envsize, y=0, colors='k')
    # plt.legend(frameon=False, fontsize=6)


def plot_density(logparams, trials, ax=None, goalcoord=[0.5], startcoord=[-0.75], goalsize=0.025, envsize=1, ):
    if ax is None:
        f,ax = plt.subplots()
    maxval  = 0 
    xs = np.linspace(-1,1,1001)

    for trial in trials:
        pcacts = predict_batch_placecell(logparams[trial], xs)
        dx = np.sum(pcacts,axis=1)
        ax.plot(xs, dx, label=f'T={trial}')
        maxval = max(maxval, np.max(dx) * 1.1)

    ax.set_xlabel('Location (x)')
    ax.set_ylabel('Density $d(x)$')
    ax.legend(frameon=False, fontsize=6)

    ax.fill_betweenx(np.linspace(0,maxval), goalcoord[0]-goalsize, goalcoord[0]+goalsize, color='r', alpha=0.25, label='Target')
    ax.axvline(startcoord[0],ymin=0, ymax=maxval, color='g',linestyle='--',label='Start', linewidth=2)
    ax.hlines(xmin=-envsize,xmax=envsize, y=0, colors='k')


def plot_frequency(allcoords, trials,ax=None, goalcoord=[0.5], startcoord=[-0.75], goalsize=0.025, envsize=1, gap=25):
    if ax is None:
        f,ax = plt.subplots()
    maxval  = 0 
    bins = 31
    bins = np.linspace(-1,1,bins)

    for trial in trials:
        fx = []
        xx = []
        for g in range(gap):

            f, x = np.histogram(allcoords[trial-g-1], bins=bins)
            x = x[:-1] + (x[1] - x[0]) / 2
            fx.append(f)
            xx.append(x)

        fx = np.array(fx)
        xx = np.array(xx)
        xx = np.mean(xx, axis=0)
        fx = np.mean(fx, axis=0)

        ax.plot(xx, fx, label=f'T={trial-gap}-{trial}')
        maxval = max(maxval, np.max(fx) * 1.1)

    ax.set_xlabel('Location (x)')
    ax.set_ylabel('Freqency $f(x)$')
    ax.legend(frameon=False, fontsize=6)

    ax.fill_betweenx(np.linspace(0,maxval), goalcoord[0]-goalsize, goalcoord[0]+goalsize, color='r', alpha=0.25, label='Target')
    ax.axvline(startcoord[0],ymin=0, ymax=maxval, color='g',linestyle='--',label='Start', linewidth=2)
    ax.hlines(xmin=-envsize,xmax=envsize, y=0, colors='k')



def plot_place_cells(params,startcoord, goalcoord,goalsize, title='', envsize=1):
    xs = np.linspace(-envsize,envsize,1000)
    pcacts = []
    velocity = []
    for x in xs:
        pc = predict_placecell(params, x)
        actout = jnp.matmul(pc, params[3])
        aprob = nn.softmax(2 * actout)
        if params[3].shape[1] == 3:
            vel = np.matmul(aprob, np.array([[-1], [1], [0]]))
        else:
            vel = np.matmul(aprob, np.array([[-1], [1]]))
        pcacts.append(pc)
        velocity.append(np.tanh(vel)*0.1)
    pcacts = np.array(pcacts)
    velocity = np.array(velocity)

    plt.figure(figsize=(4,4))
    plt.subplot(211)
    plt.title(title)
    for i in range(pcacts.shape[1]):
        plt.plot(xs, pcacts[:,i])
    plt.hlines(xmin=-envsize,xmax=envsize, y=0, colors='k')
    plt.axvline(startcoord[0], color='g',linestyle='--',label='Start', linewidth=2)
    #plt.axvline(goalcoord[0], color='r',linestyle='--',label='Goal', linewidth=2)
    plt.fill_betweenx(np.linspace(0,np.max(pcacts)), goalcoord[0]-goalsize, goalcoord[0]+goalsize, color='r', alpha=0.25)
    #plt.ylim([-0.25, 1.25])
    plt.ylabel('Tuning curves $\phi(x)$')
    plt.xlabel('Location (x)')
    plt.tight_layout()

    plt.subplot(212)
    plt.plot(xs, np.sum(pcacts,axis=1), color='red')
    plt.hlines(xmin=-envsize,xmax=envsize, y=0, colors='k')
    plt.axvline(startcoord[0], color='g',linestyle='--',label='Start', linewidth=2)
    plt.fill_betweenx(np.linspace(0,np.max(np.sum(pcacts,axis=1))), goalcoord[0]-goalsize, goalcoord[0]+goalsize, color='r', alpha=0.25)
    plt.ylabel('Field density $d(x)$')
    plt.xlabel('Location (x)')
    ax = plt.twinx()
    ax.plot(xs, velocity, color='k')
    ax.set_ylabel('Avg velocity $V(x)$')
    ax.set_ylim(-0.1,0.1)
    #plt.title('Fixed Rewards: Higher place field density at reward location')
    plt.tight_layout()
    return pcacts

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

class OneDimNav:
    def __init__(self,nact,maxspeed=0.1, envsize=1, goalsize=0.1, tmax=100, goalcoord=[0.8], startcoord=[-0.8], initvelocity=1.0, max_reward=3) -> None:
        self.tmax = tmax  # maximum steps per trial
        self.minsize = -envsize  # arena size
        self.maxsize = envsize
        self.state = 0
        self.done = False
        self.goalsize = goalsize
        self.goals = np.array(goalcoord)
        self.starts = np.array(startcoord)
        self.statesize = 1
        self.actionsize = nact
        self.maxspeed = maxspeed  # max agent speed per step
        self.tauact = 0.25
        self.total_reward = 0
        self.initvelocity = np.array(initvelocity)
        self.max_reward = max_reward

        # convert agent's onehot vector action to direction in the arena
        if self.actionsize ==3:
            self.onehot2dirmat = np.array([[-1], [1], [0]])  # move left, right, lick
        else:
            self.onehot2dirmat = np.array([[-1], [1]])  # move left, right, stay
    
    def action2velocity(self, g):
        # convert onehot action vector from actor to velocity
        return np.matmul(g, self.onehot2dirmat)

    
    def reset(self):
        if len(self.starts) > 1:
            startidx = np.random.choice(np.arange(len(self.starts)),1)
            self.state = self.starts[startidx].copy()
            if len(self.goals)>1:
                self.goal = self.goals[startidx].copy()
            else:
                self.goal = self.goals.copy()
        else:
            self.state = self.starts.copy()
            self.goal = self.goals.copy()

        #self.state = self.starts.copy()
        self.error = self.goal - self.state
        self.eucdist = abs(self.error)
        self.done = False
        self.t = 0
        self.reward = 0
        self.total_reward = 0

        self.track = []
        self.track.append(self.goal.copy())
        self.track.append(self.state.copy())

        self.velocity = np.zeros(self.statesize)
        self.velocity += self.initvelocity

        #print(f"State: {self.state}, Goal: {self.goal}")
        return self.state, self.goal, self.reward, self.done

    
    def step(self, g):
        self.t +=1
        newvelocity = self.action2velocity(g) #* self.maxspeed  # get velocity from agent's onehot action

        # newvelocity = np.clip(newvelocity, 0,1)
        # self.velocity = np.clip(1.0 - newvelocity,0,1)
        # newstate = self.state.copy() + self.velocity * self.maxspeed 

        # self.velocity += self.tauact * (newvelocity)  # smoothen actions so that agent explores the entire arena. From Foster et al. 2000
        self.velocity += self.tauact * (-self.velocity + newvelocity)
        newstate = self.state.copy() + np.clip(self.velocity, -1,1) * self.maxspeed  #np.clip(self.velocity,-self.maxspeed,self.maxspeed)   # update state with action velocity

        self.track.append(self.state.copy())

        # check if new state crosses boundary
        if newstate > self.maxsize or newstate < -self.maxsize:
            newstate = self.state.copy()
            self.velocity = np.zeros(self.statesize)

        # if newstate > self.maxsize:
        #     newstate = newstate -2
        #     #self.velocity = np.zeros(self.statesize)
        #     #self.done = True
        # if newstate < -self.maxsize:
        #     newstate = newstate +2
        #     #self.velocity = np.zeros(self.statesize)
        #     #self.done = True
        
        # if new state does not violate boundary or obstacles, update new state
        self.state = newstate.copy()
        self.error = self.goal - self.state
        self.eucdist = abs(self.error)

        # check if agent is within radius of goal
        self.reward = 0
        if (self.eucdist < self.goalsize).any():
            # if nact = 2, agent needs to be in the vicinty of goal to get a reward
            if self.actionsize == 2:
                self.reward = 1
                self.total_reward +=1 

            # if nact = 3, agent has to lick to get a reward, not merely be in vicinty of goal
            if self.actionsize == 3 and newvelocity == 0:
                self.reward = 1
                self.total_reward +=1 
        
        

        if self.total_reward == self.max_reward:
            self.done = True
        
        if self.t == self.tmax:
            self.done = True
       

        return self.state, self.reward, self.done

    def random_action(self):
        action = np.random.uniform(low=-1, high=1,size=self.actionsize)
        return action 

    def plot_trajectory(self, title=None):
        plt.figure(figsize=(4,2))
        plt.title(f'1D {title}')
        plt.hlines(xmin=self.minsize,xmax=self.maxsize, y=1, colors='k')
        plt.eventplot(self.track[1], color='g', zorder=2)
        plt.eventplot(self.track[0], color='orange', zorder=2)
        for i,s in enumerate(self.track):
            if i == 0:
                plt.eventplot(s, color='orange')
            elif i == 1:
                plt.eventplot(s, color='g') 
            else:
                plt.eventplot(s, color='b', zorder=1) 
        
def softmax_grad(s):
    # input s is softmax value of the original input x. Its shape is (1,n) 
    # i.e.  s = np.array([0.3,0.7]),  x = np.array([0,1])

    # make the matrix whose size is n^2.
    jacobian_m = np.diag(s)

    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                jacobian_m[i][j] = s[j] * (1 - s[i])
            else: 
                jacobian_m[i][j] = s[j] * (0-s[i])
    return jacobian_m


def moving_average(signal, window_size):
    # Pad the signal to handle edges properly
    padded_signal = np.pad(signal, (window_size//2, window_size//2), mode='edge')
    
    # Apply the moving average filter
    weights = np.ones(window_size) / window_size
    smoothed_signal = np.convolve(padded_signal, weights, mode='valid')
    
    return smoothed_signal

def saveload(filename, variable, opt):
    import pickle
    if opt == 'save':
        with open(f"{filename}.pickle", "wb") as file:
            pickle.dump(variable, file)
        print('file saved')
    else:
        with open(f"{filename}.pickle", "rb") as file:
            return pickle.load(file)
    

def get_pvcorr(params, start, end, num):
    xs = np.linspace(-1,1,1001)
    startpcs = predict_batch_placecell(params[start], xs)
    startvec = startpcs.flatten()
    trials = np.linspace(start, end-1, num, dtype=int)
    startxcor = startpcs@startpcs.T

    pv_corr = []
    rep_corr = []
    for i in trials:
        endpcs = predict_batch_placecell(params[i], xs)
        endvec = endpcs.flatten()
        R = np.corrcoef(startvec, endvec)[0, 1]
        pv_corr.append(R)

        endxcor = endpcs@endpcs.T
        R_rep = np.corrcoef(startxcor.flatten(), endxcor.flatten())[0, 1]
        rep_corr.append(R_rep)
    return trials, pv_corr,rep_corr, startxcor, endxcor

def get_learning_rate(initial_lr, final_lr, total_steps):
    steps = np.arange(total_steps + 1)
    decay_rate = (final_lr / initial_lr) ** (1 / total_steps)
    learning_rates = initial_lr * (decay_rate ** steps)
    return learning_rates


import numpy as np
import matplotlib.pyplot as plt
import imageio
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import imageio
from io import BytesIO


def plot_gif(logparams, startcoord=[-0.75], goalcoord=[0.5], goalsize=0.025, envsize=1, gif_name='place_cells.gif', num_frames=100, duration=5):
    frames = []
    xs = np.linspace(-envsize, envsize, 1001)
    
    # Select indices for frames to use
    frames_to_use = np.linspace(0, len(logparams) - 1, num_frames, dtype=int)
    
    for p in frames_to_use:
        params = logparams[p]
        pcacts = []
        velocity = []
        
        for x in xs:
            pc = predict_placecell(params, x)
            actout = np.matmul(pc, params[3])
            aprob = nn.softmax(2 * actout)
            if params[3].shape[1] == 3:
                vel = np.matmul(aprob, np.array([[-1], [1], [0]]))
            else:
                vel = np.matmul(aprob, np.array([[-1], [1]]))
            pcacts.append(pc)
            velocity.append(np.clip(vel,-1,1) * 0.1)
        
        pcacts = np.array(pcacts)
        velocity = np.array(velocity)

        plt.figure(figsize=(4, 4))
        for i in range(pcacts.shape[1]):
            plt.plot(xs, pcacts[:, i])
        plt.hlines(xmin=-envsize, xmax=envsize, y=0, colors='k')
        plt.axvline(startcoord[0],ymin=0, ymax= 6.01, color='g', linestyle='--', label='Start', linewidth=2)
        plt.fill_betweenx(np.linspace(0,  6.01), goalcoord[0] - goalsize, goalcoord[0] + goalsize, color='r', alpha=0.25)
        plt.ylabel('Tuning curves $\phi(x)$')
        plt.xlabel('Location (x)')
        plt.title(f'T={p}')
        plt.ylim(-0.01, 6.01)
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        frames.append(imageio.imread(buf))
        buf.close()

    # Save GIF
    imageio.mimsave(gif_name, frames, duration=duration)

# Example usage:
# plot_gif(logparams, startcoord, goalcoord, goalsize)


# Example usage:
# plot_gif(logparams, startcoord, goalcoord, goalsize)
